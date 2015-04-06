#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
The module `hdf5` defines functions to save and load Photon-HDF5 files

:func:`save_photon_hdf5` saves data from a dictionary whose keys are
Photon-HDF5 field names.

:func:`load_photon_hdf5` opens a HDF5 file, verifies that is
in a valid Photon-HDF5 format and return the root node.

This module also provides functions to save free-form dict to HDF5
(:func:`dict_to_group`) and read a HDF5 group into a dict
(:func:`dict_from_group`).

Finally there are utility functions to easily print HDF5 nodes and attributes.
"""

from __future__ import print_function, absolute_import, division

import os
import time
import re
import tables

from .metadata import official_fields_descr
from ._version import get_versions


__version__ = get_versions()['version']


def _analyze_path(name, prefix_list):
    """
    Analyze an HDF5 path.

    Arguments:
        name (string): name of the HDF5 node.
        prefix_list (list of strings): list of group names.

    Returns:
        A dictionary containing:
        - full_path: string representing the full HDF5 path.
        - group_path: string representing the full HDF5 path of the group
            containing `name`. Always ends with '/'.
        - meta_path: string representing the full HDF5 path
          with possible trailing digits removed from "/photon_dataNN"
        - is_phdata: (bool) True if `name` is a photon_data array,
            i.e. a direct child of photon_data and not a specs group.
        - is_user: (bool) True if `name` is a user-defined field.
    """
    assert name[0] != '/' and name[-1] != '/'

    group_path = '/'
    if prefix_list is not None and len(prefix_list) > 0:
        group_path += '/'.join(prefix_list) + '/'
    assert group_path[0] == '/' and group_path[-1] == '/'
    full_path = group_path + name

    chunks = full_path.split('/')
    assert len(chunks) >= 2
    assert name == chunks[-1]

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

    return dict(full_path=full_path, group_path=group_path,
                meta_path=meta_path, is_phdata=is_phdata, is_user=is_user)


def _h5_write_array(group, name, obj, descr=None, chunked=False, h5file=None):
    if isinstance(group, str):
        assert h5file is not None
    else:
        h5file = group._v_file
    if chunked:
        if obj.size == 0:
            save = h5file.create_earray
        else:
            save = h5file.create_carray
    else:
        save = h5file.create_array
    if isinstance(obj, str):
        obj = obj.encode()
    save(group, name, obj=obj, title=descr)

def _iter_hdf5_dict(data_dict, fields_descr, prefix_list=None, debug=False):
    for name, value in data_dict.items():
        if debug:
            print('Item "%s", prefix_list %s ' % (name, prefix_list))

        item = _analyze_path(name, prefix_list)
        item['description'] = fields_descr.get(item['meta_path'], '')
        item.update(name=name, value=value)
        yield item

        if isinstance(value, dict):
            if debug:
                print('Start Group "%s"' % (item['full_path']))
            new_prefix = [] if prefix_list is None else list(prefix_list)
            new_prefix.append(name)
            for sub_item in _iter_hdf5_dict(value, fields_descr, new_prefix,
                                            debug=debug):
                yield sub_item
            if debug:
                print('End Group "%s"' % (item['full_path']))

def _save_photon_hdf5_dict(group, data_dict, fields_descr, prefix_list=None,
                           debug=False):
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
    for item in _iter_hdf5_dict(data_dict, fields_descr, prefix_list, debug):
        if not item['is_user']:
            if item['description'] is '':
                print('WARNING: missing description for "%s"' % \
                      item['meta_path'])

        if isinstance(item['value'], dict):
            h5file.create_group(item['group_path'], item['name'],
                                title=item['description'])
        else:
            _h5_write_array(item['group_path'], item['name'],
                            obj=item['value'], descr=item['description'],
                            chunked=item['is_phdata'], h5file=group._v_file)

def save_photon_hdf5(data_dict,
                     h5_fname=None,
                     compression=dict(complevel=6, complib='zlib'),
                     user_descr=None,
                     debug=False):
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
            generated from d['filenamename'], by replacing the original
            extension with '.hdf5'.
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
    title = official_fields_descr['/']
    data_file = tables.open_file(h5_fname, mode="w", title=title,
                                 filters=comp_filter)
    # Saving a file reference is useful in case of error
    orig_data_dict = data_dict
    data_dict = data_dict.copy()
    orig_data_dict.update(data_file=data_file)

    ## Add provenance metadata
    if 'provenance' in data_dict:
        provenance = data_dict['provenance']
        orig_fname = None
        if os.path.isfile(provenance['filename']):
            orig_fname = provenance['filename']
        elif os.path.isfile(provenance['filename_full']):
            orig_fname = provenance['filename_full']
        else:
            print("WARNING: Could not locate original file '%s'" % \
                  provenance['filename'])
        if orig_fname is not None:
            provenance.update(_get_file_metadata(orig_fname))

    ## Add identity metadata
    identity = get_identity(data_file)
    identity.update(software='phconvert',
                    software_version=__version__)
    data_dict['identity'] = identity

    ## Save everything to disk
    fields_descr = official_fields_descr.copy()
    if user_descr is not None:
        fields_descr.update(user_descr)
    _save_photon_hdf5_dict(data_file.root, data_dict,
                           fields_descr=fields_descr, debug=debug)
    data_file.flush()


def get_identity(h5file, format_version='0.3'):
    full_h5filename = os.path.abspath(h5file.filename)
    h5filename = os.path.basename(full_h5filename)
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    identity = dict(filename=h5filename,
                    filename_full=full_h5filename,
                    creation_time=creation_time,
                    format_name='Photon-HDF5',
                    format_version=format_version,
                    format_url='http://photon-hdf5.readthedocs.org/')
    return identity

def _get_file_metadata(fname):
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


def dict_from_group(group, read=True):
    """Return a dict with the content of a PyTables `group`."""
    out = {}
    for node in group:
        if isinstance(node, tables.Group):
            value = dict_from_group(node, read=read)
        else:
            if read:
                value = node.read()
            else:
                value = node
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

def load_photon_hdf5(filename, strict=True):
    assert os.path.isfile(filename)
    h5file = tables.open_file(filename)
    assert_valid_photon_hdf5(h5file.root, strict=strict)
    return h5file.root


class Invalid_PhotonHDF5(Exception):
    """Error raised when a file is not a valid Photon-HDF5 file.
    """
    pass

def _raise_invalid_file(msg, strict=True, norepeat=False, pool=None):
    """Raise Invalid_PhotonHDF5 if strict is True, print a warning otherwise.
    """
    if norepeat:
        if msg in pool:
            return
    if strict:
        raise Invalid_PhotonHDF5(msg)
    else:
        print('Photon-HDF5 WARNING: %s' % msg)
    if norepeat:
        pool.append(msg)

def _check_has_field(name, group, strict=True):
    msg = 'Missing "%s" in "%s".'
    if name not in group:
        _raise_invalid_file(msg % (name, group._v_pathname), strict)

def _check_path(path, strict=True):
    if '/user' in path:
        return

    if path.startswith('/photon_data'):
        # Remove eventual digits after /photon_data
        pattern = '/photon_data[0-9]*(.*)'
        path = '/photon_data' + re.match(pattern, path).group(1)

    if path not in official_fields_descr:
        msg = ('Unknown field "%s". '
               'Custom fields must be inside a "user" group.' % path)
        if strict:
            raise Invalid_PhotonHDF5(msg)
        else:
            print('Photon-HDF5 WARNING: %s' % msg)

def _check_valid_names(data, strict=True):
    already_verified = []
    for group in data._f_walk_groups():
        path = group._v_pathname
        _check_path(path, strict=strict)
        already_verified.append(path)
        for node in group._f_iter_nodes():
            path = node._v_pathname
            if path not in already_verified:
                _check_path(path, strict=strict)
                already_verified.append(path)


def assert_valid_photon_hdf5(data, strict=True):
    """
    Validate the structure of a Photon-HDF5 file.

    Raise an error when missing photon_data group, timestamps array and
    timestamps_unit.

    When `strict` is True, raise an error if
    """
    _check_valid_names(data, strict=strict)
    _check_has_field('acquisition_time', data, strict=strict)
    _check_has_field('comment', data, strict=strict)

    if 'photon_data' in data:
        ph_data_m = [data.photon_data]
    elif 'photon_data0' in data:
        ph_data_m = [data._f_get_child(k) for k in data._v_groups.keys()
                     if k.startswith('photon_data')]
        ph_data_m.sort()
    else:
        msg = 'Invalid Photon-HDF5: missing "photon_data" group.'
        raise Invalid_PhotonHDF5(msg)

    pool = []
    for ph_data in ph_data_m:
        _check_photon_data(ph_data, strict=strict, norepeat=True, pool=pool)

    if 'setup' in data:
        _check_setup(data.setup, strict=strict)
    else:
        _raise_invalid_file('Invalid Photon-HDF5: Missing /setup group.',
                            strict)

def _check_setup(setup, strict=True):
    mantatory_fields = ['num_pixels', 'num_spots', 'num_spectral_ch',
                        'num_polarization_ch', 'num_split_ch',
                        'modulated_excitation', 'lifetime']
    for name in mantatory_fields:
        if name not in setup:
            _raise_invalid_file('Missing "/setup/%s".' % name, strict)

def _check_photon_data(ph_data, strict=True, norepeat=False, pool=None):

    def _assert_has_field(name, group):
        msg = 'Missing "%s" in "%s".'
        if name not in group:
            raise Invalid_PhotonHDF5(msg % (name, group._v_pathname))

    _assert_has_field('timestamps', ph_data)
    _assert_has_field('timestamps_specs', ph_data)
    _assert_has_field('timestamps_unit', ph_data.timestamps_specs)

    spectral_meas_types = ['smFRET',
                           'smFRET-usALEX', 'smFRET-usALEX-3c',
                           'smFRET-nsALEX']
    if 'measurement_specs' not in ph_data:
        _raise_invalid_file('Missing "measurement_specs".',
                            strict, norepeat, pool)
        return

    measurement_specs = ph_data.measurement_specs
    if 'measurement_type' not in measurement_specs:
        _raise_invalid_file('Missing "measurement_type"',
                            strict, norepeat, pool)
        return

    measurement_type = measurement_specs.measurement_type.read()
    if measurement_type not in spectral_meas_types:
        raise Invalid_PhotonHDF5('Unkwnown measurement type "%s"' % \
                                 measurement_type)

    # At this point we have a valid measurement_type
    # Any missing field will raise an error (regardless of `strict`).
    def _assert_has_field_mtype(name, group):
        msg = 'Missing "%s" in "%s".\nThis field is mandatory for "%s" data.'
        if name not in group:
            raise Invalid_PhotonHDF5(msg % (name, group._v_pathname,
                                            measurement_type))

    detectors_specs = measurement_specs.detectors_specs
    _assert_has_field_mtype('spectral_ch1', detectors_specs)
    _assert_has_field_mtype('spectral_ch2', detectors_specs)

    if measurement_type in ['smFRET-usALEX', 'smFRET-usALEX-3c']:
        _assert_has_field_mtype('alex_period', measurement_specs)

    if measurement_type == 'smFRET-nsALEX':
        _assert_has_field_mtype('laser_pulse_rate', measurement_specs)
        _assert_has_field_mtype('nantotimes', ph_data)
        _assert_has_field_mtype('nantotimes_specs', ph_data)
        for name in ['tcspc_unit', 'tcspc_range', 'tcspc_num_bins',
                     'time_reversed']:
             _assert_has_field_mtype(name, ph_data.nantotimes_specs)


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
