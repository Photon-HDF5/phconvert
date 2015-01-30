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

from __future__ import print_function, absolute_import
from builtins import zip

import os
import time
import tables
from collections import OrderedDict

import phconvert    # To get the version


# Metadata for the HDF5 root node
_root_attributes = OrderedDict([
    ('format_name', 'Photon-HDF5'),
    ('format_title', 'HDF5-based format for time-series of photon data.'),
    ('format_version', '0.2'),
    ('format_url', 'http://photon-hdf5.readthedocs.org/'),
])

# Metadata for different fields (arrays) in the HDF5 format
fields_descr = OrderedDict([
    # Root parameters
    ('num_spots', 'Number of excitation or detection spots.'),
    ('num_spectral_ch', ('Number of different spectral bands in the detection '
                         'channels (i.e. 2 for 2-colors smFRET).')),
    ('num_polariz_ch', ('Number of different polarization in the detection '
                        'channels. The value is 1 if no polarization selection '
                        'is performed and 2 if two orthogonal polarizations '
                        'are recorded.')),
    ('num_detectors', ('Total number of detector pixels used in the '
                       'measurement.')),
    ('acquisition_time', 'Measurement duration in seconds.'),
    ('lifetime', ('If True (or 1) the data contains nanotimes from TCSPC '
                  'hardware')),
    ('alex', 'If True (or 1) the file contains ALternated EXcitation data.'),
    ('alex_period', ('The duration of the us-ALEX excitation alternation '
                     'in the same units as the timestamps.')),
    ('laser_pulse_rate', 'The laser(s) pulse rate in Hertz.'),
    ('alex_period_donor', ('Start and stop values identifying the donor '
                           'emission period.')),
    ('alex_period_acceptor', ('Start and stop values identifying the acceptor '
                              'emission period.')),
    ('timestamps_unit', 'Time in seconds of 1-unit increment in timestamps.'),

    # Photon-data
    ('photon_data', ('Group containing arrays of photon-data (one element per '
                     'photon)')),
    ('timestamps', 'Array of photon timestamps'),

    ('detectors', 'Array of detector numbers for each timestamp'),
    ('detectors_specs', 'Group for detector-specific data.'),
    ('donor', 'Detectors for the donor spectral range'),
    ('acceptor', 'Detectors for the acceptor spectral range'),
    ('polarization1', ('Detectors ID for the "polarization1". By default is '
                       'the polarization parallel to the excitation, '
                       'unless specified differently in the "/setup_specs".')),
    ('polarization2', ('Detectors ID for the "polarization2". By default is '
                       'the polarization perpendicular to the excitation, '
                       'unless specified differently in the "/setup_specs".')),

    ('nanotimes', 'TCSPC photon arrival time (nanotimes)'),
    ('nanotimes_specs', 'Group for nanotime-specific data.'),
    ('tcspc_unit', 'TCSPC time bin duration in seconds (nanotimes unit).'),
    ('tcspc_num_bins', 'Number of TCSPC bins.'),
    ('tcspc_range', 'TCSPC full-scale range in seconds.'),
    ('tau_accept_only', 'Intrinsic Acceptor lifetime (seconds).'),
    ('tau_donor_only', 'Intrinsic Donor lifetime (seconds).'),
    ('tau_fret_donor', 'Donor lifetime in presence of Acceptor (seconds).'),
    ('inverse_fret_rate', ('FRET energy transfer lifetime (seconds). Inverse '
                           'of the rate of D*A -> DA*.')),

    ('particles', 'Particle label (integer) for each timestamp.'),

    ## Setup group
    ('setup', 'Information about the experimental setup.'),
    ('excitation_wavelengths', 'Array of excitation wavelengths (meters).'),
    ('excitation_powers', ('Array of excitation powers (in the same order as '
                           'excitation_wavelengths). Units: Watts.')),
    ('excitation_polarizations', ('Polarization angle (in degrees), one for '
                                  'each laser.')),
    ('detection_polarization1', ('Polarization angle (in degrees) for '
                                 '"polarization1".')),
    ('detection_polarization2', ('Polarization angle (in degrees) for '
                                 '"polarization2".')),

    ## Provenance group
    ('provenance', 'Information about the original data file.'),
    ('filename', 'Original file name.'),
    ('full_filename', 'Original full file name, including the folder.'),
    ('creation_time', 'Original file creation time.'),
    ('modification_time', 'Original file time of last modification.'),

    ## Identity group
    ('identity', 'Information about the Photon-HDF5 data file.'),
    ('identity_filename', 'Photon-HDF5 file name at creation time.'),
    ('identity_full_filename', ('Photon-HDF5 full file name, including the '
                                'folder.')),
    ('identity_creation_time', 'Photon-HDF5 file creation time.'),
    ('identity_software', 'Software used to save the Photon-HDF5 file.'),
    ('identity_software_version', ('Software version used to save the '
                                   'Photon-HDF5 file.')),
    ])

mandatory_root_fields = ['timestamps_unit', 'num_spots', 'num_detectors',
                         'num_spectral_ch', 'num_polariz_ch',
                         'alex', 'lifetime',]

optional_root_fields = ['acquisition_time']

setup_fields = ['excitation_wavelengths', 'excitation_powers',
                'excitation_polarizations', 'detection_polarization1',
                'detection_polarization2']

provenance_fields = ['filename', 'full_filename', 'creation_time',
                     'modification_time']


class H5Writer(object):
    """Helper class for writing items with description into HDF5.

    It uses the global metadata dictionary to retrive the field description
    from the field/key name.
    """
    def __init__(self, h5file, data, comp_filter):
        self.h5file = h5file
        self.data = data
        self.comp_filter = comp_filter

    def _add_data(self, where, name, func, obj=None, strip_prefix=False,
                  **kwargs):
        if obj is None:
            obj = self.data[name]
        h5name = name
        if strip_prefix:
            h5name = '_'.join(h5name.split('_')[1:])
        func(where, h5name, obj=obj, title=fields_descr[name], **kwargs)

    def add_carray(self, where, name, obj=None):
        self._add_data(where, name, self.h5file.create_carray, obj=obj,
                       filters=self.comp_filter)

    def add_array(self, where, name, obj=None, strip_prefix=False):
        self._add_data(where, name, self.h5file.create_array, obj=obj,
                       strip_prefix=strip_prefix)

    def add_group(self, where, name, descr_name=None):
        if descr_name is None:
            descr_name = name
        return self.h5file.create_group(where, name,
                                        title=fields_descr[descr_name])


def photon_hdf5(d, compression=dict(complevel=6, complib='zlib'),
                h5_fname=None, title="Confocal smFRET data",
                iter_timestamps=None, iter_detectors=None):
    """
    Saves the dict `d` in the Photon-HDF5 format.

    As a side effect `d` is modified by adding the key 'data_file' that
    contains a reference to the pytables file.

    Arguments:
        d (dict): the dictionary containing the smFRET measurement.
        compression (dict): a dictionary containing the compression type
            and level. Passed to pytables `tables.Filters()`.
        h5_fname (string or None): if not None, contains the file name
            to be used for the HDF5 file. If None, the file name is
            generated from d['fname'], by replacing the original extension
            with '.hdf5'.

    For description and specs of the Photon-HDF5 format see:
    http://photon-hdf5.readthedocs.org/
    """
    comp_filter = tables.Filters(**compression)

    if h5_fname is None:
        basename, extension = os.path.splitext(d['filename'])
        if compression['complib'] == 'blosc':
            basename += '_blosc'
        h5_fname = basename + '.hdf5'

    if os.path.isfile(h5_fname):
        basename, extension = os.path.splitext(h5_fname)
        h5_fname = basename + '_new_copy.hdf5'

    print('Saving: %s' % h5_fname)
    data_file = tables.open_file(h5_fname, mode="w", title=title)
    # Saving a file reference is usefull in case of error
    d.update(data_file=data_file)
    writer = H5Writer(data_file, d, comp_filter)

    ## Save the root-node metadata
    for name, value in _root_attributes.items():
        data_file.root._f_setattr(name, value)

    ## Save the mandatory parameters
    for field in mandatory_root_fields:
        writer.add_array('/', field)

    ## Save optional parameters
    for field in optional_root_fields:
        if field in d:
            writer.add_array('/', field)

    if d['alex']:
        if not d['lifetime']:
            writer.add_array('/', 'alex_period')

        for field in ['alex_period_donor', 'alex_period_acceptor']:
            if field in d:
                writer.add_array('/', field)

    ## Save the photon-data
    if d['num_spots'] == 1:
         _save_photon_data(writer, d)
    else:
        for ich, (timest, det) in enumerate(zip(iter_timestamps,
                                                iter_detectors)):
            ph_group = writer.add_group('/', 'photon_data_%d' % ich,
                                        descr_name='photon_data')
            _save_photon_data(writer, d, ph_group,
                              timestamps=timest, detectors=det)

    ## Add setup info, if present in d
    setup_group = writer.add_group('/', 'setup')
    for field in setup_fields:
        if field in d:
            writer.add_array(setup_group, field)

    ## Add provenance metadata
    orig_file_metadata = dict(filename=d['filename'])
    if os.path.isfile(d['filename']):
        orig_file_metadata = get_file_metadata(d['filename'])
    else:
        print("WARNING: Could locate original file '%s'\n" % d.fname)
        print("         Provenance info not saved.\n")

    # A user provided `provenance` dict overrides pre-computes values
    if 'provenance' in d:
        orig_file_metadata.update(d['provenance'])

    prov_group = writer.add_group('/', 'provenance')
    for field, value in orig_file_metadata.items():
        assert field in provenance_fields
        writer.add_array(prov_group, field, obj=value.encode('latin-1'))

    ## Add identity metadata
    full_h5filename = os.path.abspath(h5_fname)
    h5filename = os.path.basename(full_h5filename)
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    identity_metadata = dict(identity_filename=h5filename,
                             identity_full_filename=full_h5filename,
                             identity_creation_time=creation_time,
                             identity_software='phconvert',
                             identity_software_version=phconvert.__version__)
    identity_group = writer.add_group('/', 'identity')
    for field, value in identity_metadata.items():
        writer.add_array(identity_group, field, obj=value.encode('latin-1'),
                         strip_prefix=True)
    data_file.flush()


def _save_photon_data(writer, d, ph_group=None, timestamps=None,
                      detectors=None):
    if ph_group is None:
        ph_group = writer.add_group('/', 'photon_data')
    if timestamps is None:
        timestamps = d['timestamps']
    if detectors is None:
        detectors = d.get('detectors', 'unspecified')

    writer.add_carray(ph_group, 'timestamps', obj=timestamps)
    if detectors != 'unspecified':
        writer.add_carray(ph_group, 'detectors', obj=detectors)
        det_group = writer.add_group(ph_group, 'detectors_specs')
        writer.add_array(det_group, 'donor')
        writer.add_array(det_group, 'acceptor')

    if d['lifetime']:
        if 'laser_pulse_rate' in d:
            writer.add_array('/', 'laser_pulse_rate')

        writer.add_carray(ph_group, 'nanotimes')
        nt_group = writer.add_group(ph_group, 'nanotimes_specs')

        # Mandatory specs
        nanotimes_specs = ['tcspc_unit', 'tcspc_num_bins', 'tcspc_range']
        for spec in nanotimes_specs:
            writer.add_array(nt_group, spec)

        # Optional specs
        nanotimes_specs = ['tau_accept_only', 'tau_donor_only',
                           'tau_fret_donor', 'inverse_fret_rate']
        for spec in nanotimes_specs:
            if spec in d:
                writer.add_array(nt_group, spec)

    if 'particles' in d:
        writer.add_carray(ph_group, 'particles')


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

    metadata = dict(filename=filename, full_filename=full_filename,
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

