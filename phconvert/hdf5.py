#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines the function :func:`photon_hdf5` to save data from a
dictionary to **Photon-HDF5** format. The keys of the dictionary must be
valid field names in the Photon-HDF5 format.

It also provides functions to save free-form dict to HDF5
(:func:`dict_to_group`) and read a HDF5 group into a dict
(:func:`dict_from_group`).

Finally there are utility functions to easily print HDF5 nodes and attributes.
"""

from __future__ import print_function, absolute_import, division

import os
import time
import re
import tables
from collections import OrderedDict

from ._version import get_versions


__version__ = get_versions()['version']


official_fields_descr = OrderedDict([
    ## Root fields
    ('/acquisition_time', 'Measurement duration in seconds.'),
    ('/comment', 'A user defined comment for the data file.'),

    ## Photon data group
    ('/photon_data',
         'Group containing arrays of photon-data (one element per photon)'),
    ('/photon_data/timestamps', 'Array of photon timestamps.'),
    ('/photon_data/detectors', 'Array of detector IDs for each timestamp.'),
    ('/photon_data/nanotimes', 'TCSPC photon arrival time (nanotimes).'),
    ('/photon_data/particles', 'Particle IDs (integer) for each timestamp.'),

    ('/photon_data/timestamps_specs', 'Specifications for timestamps.'),
    ('/photon_data/timestamps_specs/timestamps_unit',
         ('Time in seconds of 1-unit increment in timestamps.')),

    ('/photon_data/nanotimes_specs', 'Group for nanotime-specific data.'),
    ('/photon_data/nanotimes_specs/tcspc_unit',
         'TCSPC time bin duration in seconds (nanotimes unit).'),
    ('/photon_data/nanotimes_specs/tcspc_num_bins',
         'Number of TCSPC bins.'),
    ('/photon_data/nanotimes_specs/tcspc_range',
         'TCSPC full-scale range in seconds.'),
    ('/photon_data/nanotimes_specs/irf_donor_hist',
         ('Instrument Response Function (IRF) histogram for the donor '
          'detection channel.')),
    ('/photon_data/nanotimes_specs/irf_acceptor_hist',
         ('Instrument Response Function (IRF) histogram for the acceptor '
          'detection channel.')),
    ('/photon_data/nanotimes_specs/calibration_hist',
         ('Histogram of uncorrelated counts used to correct the TCSPC '
          'non-linearities.')),

    ('/photon_data/measurement_specs',
         ('Metadata necessary for interpretation of the particular type of '
          'measurement.')),
    ('/photon_data/measurement_specs/measurement_type',
         'Name of the measurement the data represents.'),
    ('/photon_data/measurement_specs/alex_period',
         ('Period of laser alternation in us-ALEX measurements in timestamps '
          'units.')),
    ('/photon_data/measurement_specs/laser_pulse_rate',
         'Repetition rate of the pulsed excitation laser.'),
    ('/photon_data/measurement_specs/alex_period_spectral_ch1',
         ('Value pair identifing the range of spectral_ch1 photons in one '
          'period of laser alternation or interleaved pulses.')),
    ('/photon_data/measurement_specs/alex_period_spectral_ch2',
         ('Value pair identifing the range of spectral_ch2 photons in one '
          'period of laser alternation or interleaved pulses.')),

    ('/photon_data/measurement_specs/detectors_specs',
         'Mapping between the detector IDs and the detection channels.'),

    ('/photon_data/measurement_specs/detectors_specs/spectral_ch1',
         ('Pixel IDs for the first spectral channel (i.e. donor in a '
          '2-color smFRET measurement).')),
    ('/photon_data/measurement_specs/detectors_specs/spectral_ch2',
         ('Pixel IDs for the first spectral channel (i.e. acceptor in a '
          '2-color smFRET measurement).')),

    ('/photon_data/measurement_specs/detectors_specs/polarization_ch1',
         'Pixel IDs for the first polarization channel.'),
    ('/photon_data/measurement_specs/detectors_specs/polarization_ch2',
         'Pixel IDs for the second polarization channel.'),

    ('/photon_data/measurement_specs/detectors_specs/split_ch1',
         ('Pixel IDs for the first channel splitted through a '
          'non-polarizing beam splitter.')),
    ('/photon_data/measurement_specs/detectors_specs/split_ch2',
         ('Pixel IDs for the second channel splitted through a '
          'non-polarizing beam splitter.')),

    ('/photon_data/measurement_specs/detectors_specs/labels',
         ('User defined labels for each pixel IDs. In smFRET it is strongly '
          'suggested to use "donor" and "acceptor" for the respective '
          'pixel IDs.')),

    ## Other root groups
    ('/identity', 'Information about the Photon-HDF5 data file.'),
    ('/provenance', 'Information about the original data file.'),
    ('/setup', 'Information about the experimental setup.'),
    ('/sample', 'Information about the measured sample.'),


])


class H5Writer(object):
    """Helper class for writing items with associated descriptions into HDF5.
    """
    def __init__(self, h5file, comp_filter):
        self.h5file = h5file
        self.comp_filter = comp_filter

    def write_group(self, where, name, descr=None):
        return self.h5file.create_group(where, name, title=descr)

    def _write_data(self, where, name, obj, func, descr=None,
                    **kwargs):
        func(where, name, obj=obj, title=descr, **kwargs)

    def write_array(self, where, name, obj, descr=None, chunked=False):
        if not chunked:
            method = self.h5file.create_array
        else:
            method = self.h5file.create_carray
        self._write_data(where, name, obj=obj, func=method, descr=descr,
                         filters=self.comp_filter)

def _analyze_path(name, prefix_list):
    """
    From a name (string) and a prefix_list (list of strings)

    Returns:
        - the meta_path, that is a string with the full HDF5 path
          with possible trailing digits removed from "/photon_dataNN"
        - whether `name` is a user-defined field
        - whether `name` is a photon_data array, i.e. a direct child of
          photon_data and not a specs group.

    """
    assert name[0] != '/' and name[-1] != '/'
    full_path = '/' + name
    if prefix_list is not None and len(prefix_list) > 0:
        prefix = '/'.join(prefix_list)
        assert prefix[0] != '/' and prefix[-1] != '/'
        full_path = '/' + prefix + full_path
    chunks = full_path.split('/')
    assert len(chunks) >= 2
    assert name == chunks[-1]

    #group_path = '/'.join(chunks[:-1]) + '/'
    is_user = 'user' in chunks

    meta_path = full_path
    is_phdata = False
    if full_path.startswith('/photon_data'):
        if len(chunks) == 3 and not name.endswith('_specs'):
            is_phdata = True
        # Remove eventual digits after /photon_data
        pattern = '/photon_data[0-9]*(.*)'
        meta_path = '/photon_data' + \
                    re.match(pattern, full_path).group(1)

    return meta_path, is_phdata, is_user

def _h5_write_array(group, name, obj, descr=None, chunked=False):
    h5file = group._v_file
    if chunked:
        save = h5file.create_carray
    else:
        save = h5file.create_array
    save(group, name, obj=obj, title=descr)

def _save_photon_hdf5_dict(group, data_dict, fields_descr, prefix_list=None):
    """
    Assumptions:
        data_dict is a hierarchical dict whose values are either arrays or
        sub-dictionaries representing a sub-group.

        fields_descr merges official and user-defined field descriptions
        where the key is always the normalized full path (meta path).
        The meta path is the full path where the string "/photon_dataNN"
        is replaced by "/photon_data".
    """
    h5file = group._v_file
    for name, value in data_dict.items():
        descr_key, is_phdata, is_user = _analyze_path(name, prefix_list)
        # Allow missing description in user fields
        description = fields_descr.get(descr_key, '')
        if not is_user:
            #assert description is not None,
            #       'Name "%s" is not valid.' % descr_key
            if description is None:
                print('WARNING: missing description for "%s"' % descr_key)

        if isinstance(value, dict):
            # Current key is a group, create it and walk through its content
            subgroup = h5file.create_group(group, name, title=description)

            new_prefix_list = [] if prefix_list is None else list(prefix_list)
            new_prefix_list.append(name)
            _save_photon_hdf5_dict(subgroup, value, fields_descr, new_prefix_list)
        else:
            _h5_write_array(group, name, obj=value, descr=description,
                            chunked=is_phdata)

def photon_hdf5(data_dict, compression=dict(complevel=6, complib='zlib'),
                h5_fname=None,
                title="Photon-HDF5: A container for photon data.",
                user_descr=None
                #iter_timestamps=None, iter_detectors=None
                ):
    """
    Saves the dict `d` in the Photon-HDF5 format.

    As a side effect `d` is modified by adding the key 'data_file' that
    contains a reference to the pytables file.

    Arguments:
        data_dict (dict): the dictionary containing the photon data.
            The keys must strings matching valid Photon-HDF5 paths.
            The values must be scalars, arrays or strings.
        compression (dict): a dictionary containing the compression type
            and level. Passed to pytables `tables.Filters()`.
        h5_fname (string or None): if not None, contains the file name
            to be used for the HDF5 file. If None, the file name is
            generated from d['fname'], by replacing the original extension
            with '.hdf5'.
        user_descr (dict or None): dictionary of field descriptions for
            user-defined fields. The keys must be strings representing
            the full HDF5 path of each field.

    For description and specs of the Photon-HDF5 format see:
    http://photon-hdf5.readthedocs.org/
    """
    comp_filter = tables.Filters(**compression)

    if h5_fname is None:
        basename, extension = os.path.splitext(data_dict.pop('filename'))
        if compression['complib'] == 'blosc':
            basename += '_blosc'
        h5_fname = basename + '.hdf5'

    if os.path.isfile(h5_fname):
        basename, extension = os.path.splitext(h5_fname)
        h5_fname = basename + '_new_copy.hdf5'

    print('Saving: %s' % h5_fname)
    data_file = tables.open_file(h5_fname, mode="w", title=title,
                                 filters=comp_filter)
    # Saving a file reference is useful in case of error
    backup = data_dict
    data_dict = data_dict.copy()
    backup.update(data_file=data_file)

    ## Add provenance metadata
    if 'provenance' in data_dict:
        provenance = data_dict['provenance']
        orig_fname = None
        if os.path.isfile(provenance['filename']):
            orig_fname = provenance['filename']
        elif os.path.isfile(provenance['full_filename']):
            orig_fname = provenance['full_filename']
        else:
            print("WARNING: Could not locate original file '%s'" % \
                  provenance['filename'])
        if orig_fname is not None:
            provenance.update(get_file_metadata(orig_fname))

    ## Add identity metadata
    full_h5filename = os.path.abspath(h5_fname)
    h5filename = os.path.basename(full_h5filename)
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    identity = dict(filename=h5filename,
                    full_filename=full_h5filename,
                    creation_time=creation_time,
                    software='phconvert',
                    software_version=__version__,
                    format_name='Photon-HDF5',
                    format_version='0.3',
                    format_url='http://photon-hdf5.readthedocs.org/')
    data_dict['identity'] = identity

    ## Save everything to disk
    fields_descr = official_fields_descr.copy()
    if user_descr is not None:
        fields_descr.update(user_descr)
    _save_photon_hdf5_dict(data_file.root, data_dict,
                           fields_descr=fields_descr)
    data_file.flush()


def get_file_metadata(fname):
    """Return a dict with file metadata.
    """
    assert os.path.isfile(fname)

    full_filename = os.path.abspath(fname)
    filename = os.path.basename(full_filename)

    # Creation and modification time (but not exactly on *NIX)
    # see https://docs.python.org/2/library/os.path.html#os.path.getctime)
    ctime = time.localtime(os.path.getctime(full_filename))
    mtime = time.localtime(os.path.getmtime(full_filename))

    ctime_str = time.strftime("%Y-%m-%d %H:%M:%S", ctime)
    mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", mtime)

    metadata = dict(filename=filename, filename_full=full_filename,
                    creation_time=ctime_str, modification_time=mtime_str)
    return metadata


def dict_from_group(group):
    """Return a dict with the content of a PyTables `group`."""
    out = {}
    for node in group:
        if isinstance(node, tables.Group):
            value = dict_from_group(node)
        else:
            value = node.read()
        out[node._v_name] = value
    return out

def dict_to_group(group, dictionary):
    """Save `dictionary` into HDF5 format in `group`.
    """
    h5file = group._v_file
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subgroup = h5file.create_group(group, key)
            dict_to_group(subgroup, value)
        else:
            h5file.create_array(group, name=key, obj=value)
    h5file.flush()

def print_attrs(data_file, node_name='/', which='user'):
    """Print the HDF5 attributes for `node_name`.

    Parameters:
        data_file (pytables HDF5 file object): the data file to print
        node_name (string): name of the path inside the file to be printed.
            Can be either a group or a leaf-node. Default: '/', the root node.
        which (string): Valid values are 'user' for user-defined attributes,
            'sys' for pytables-specific attributes and 'all' to print both
            groups of attributes. Default 'user'.
    """
    node = data_file.get_node(node_name)
    print('List of attributes for:\n  %s\n' % node)
    for attr in node._v_attrs._f_list(which):
        print('\t%s' % attr)
        print('\t    %s' % repr(node._v_attrs[attr]))

def print_children(data_file, group='/'):
    """Print all the sub-groups in `group` and leaf-nodes children of `group`.

    Parameters:
        data_file (pytables HDF5 file object): the data file to print
        group (string): path name of the group to be printed.
            Default: '/', the root node.
    """
    base = data_file.get_node(group)
    print('Groups in:\n  %s\n' % base)

    for node in base._f_walk_groups():
        if node is not base:
            print('    %s' % node)

    print('\nLeaf-nodes in %s:' % group)
    for node in base._v_leaves.itervalues():
        info = node.shape
        if len(info) == 0:
            info = node.read()
        print('\t%s, %s' % (node.name, info))
        if len(node.title) > 0:
            print('\t    %s' % node.title)

del print_function

