#
# phconvert - Reference library to read and save Photon-HDF5 files
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

Finally there are utility functions to easily print HDF5 nodes and attributes
(:func:`print_children`, :func:`print_attrs`).
"""

from __future__ import print_function, absolute_import, division

import os
import time
import re
import tables
import numpy as np


from .metadata import (official_fields_specs, root_attributes,
                       LATEST_FORMAT_VERSION)
from ._version import get_versions


__version__ = get_versions()['version']

_EMPTY = ' '


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

def _is_structured_array(obj):
    if hasattr(obj, 'dtype') and obj.dtype.kind == 'V':
        return True
    else:
        return False

def _h5_write_array(group, name, obj, descr=None, chunked=False, h5file=None):
    """Writes `obj` in the pytables HDF5 `group` with name `name`.
    """
    if isinstance(group, str):
        assert h5file is not None
    else:
        h5file = group._v_file
    if chunked:
        if obj.size == 0:
            save = h5file.create_earray
        else:
            save = h5file.create_carray
    elif _is_structured_array(obj):
        save = h5file.create_table
    else:
        save = h5file.create_array
        if isinstance(obj, str):
            obj = obj.encode()

    save(group, name, obj=obj)
    # Set title through property access to work around pytable issue
    # under python 3 (https://github.com/PyTables/PyTables/issues/469)
    node = h5file.get_node(group)._f_get_child(name)
    node.title = descr.encode()  # saved as binary both on py2 and py3

def _iter_hdf5_dict(data_dict, prefix_list=None, fields_descr=None,
                    debug=False):
    """Recursively iterate over `data_dict` returning a dict for each item.

    This is an iterator returning a dict for each item in `data_dict` (i.e.
    a data-field in HDF5 file) and its sub-dicts (i.e. a group in HDF5 file).

    Each returned dict contains the following keys:
    'full_path', 'group_path', 'meta_path', 'is_phdata', 'is_user',
    'description'.
    """
    if fields_descr is None:
        fields_descr = {}
    for name, value in data_dict.items():
        if name.startswith('_'):
            continue
        if debug:
            print('Item "%s", prefix_list %s ' % (name, prefix_list))

        item = _analyze_path(name, prefix_list)
        item['description'] = fields_descr.get(item['meta_path'], _EMPTY)
        item.update(name=name, value=value, curr_dict=data_dict)
        yield item

        if isinstance(value, dict):
            if debug:
                print('Start Group "%s"' % (item['full_path']))
            new_prefix = [] if prefix_list is None else list(prefix_list)
            new_prefix.append(name)
            for sub_item in _iter_hdf5_dict(value, new_prefix, fields_descr,
                                            debug=debug):
                yield sub_item
            if debug:
                print('End Group "%s"' % (item['full_path']))

def _save_photon_hdf5_dict(group, data_dict, fields_descr, prefix_list=None,
                           debug=False):
    """
    Save a hierarchical structure `data_dict` in a HDF5 `group`.

    Assumptions:
        data_dict is a hierarchical dict whose values are either arrays or
        sub-dictionaries representing a sub-group.

        `fields_descr` merges official and user-defined field descriptions
        where the key is always the normalized full path (meta path).
        The meta path is the full path where the string "/photon_dataNN"
        is replaced by "/photon_data".
    """
    h5file = group._v_file
    for item in _iter_hdf5_dict(data_dict, prefix_list, fields_descr, debug):
        if not item['is_user']:
            if item['description'] == _EMPTY:
                print('WARNING: missing description for "%s"' % \
                      item['meta_path'])

        if isinstance(item['value'], dict):
            h5file.create_group(item['group_path'], item['name'],
                                title=item['description'].encode())
        else:
            _h5_write_array(item['group_path'], item['name'],
                            obj=item['value'], descr=item['description'],
                            chunked=item['is_phdata'], h5file=group._v_file)

def save_photon_hdf5(data_dict,
                     strict = True,
                     h5_fname = None,
                     compression = dict(complevel=6, complib='zlib'),
                     user_descr = None,
                     debug = False,
                     close = True,
                     overwrite = False,
                     validate = True):
    """
    Saves the dict `d` in the Photon-HDF5 format.

    As a side effect `d` is modified by adding the key '_data_file' that
    contains a reference to the pytables file.

    Arguments:
        data_dict (dict): the dictionary containing the photon data.
            The keys must strings matching valid Photon-HDF5 paths.
            The values must be scalars, arrays or strings.
        strict (bool): if True, raises an error when not following the specs.
            If False, does not rais an error but print a warning.
        compression (dict): a dictionary containing the compression type
            and level. Passed to pytables `tables.Filters()`.
        h5_fname (string or None): if not None, contains the file name
            to be used for the HDF5 file. If None, the file name is
            generated from d['filename'], by replacing the original
            extension with '.hdf5'.
        user_descr (dict or None): dictionary of field descriptions for
            user-defined fields. The keys must be strings representing
            the full HDF5 path of each field. The values must be
            binary (i.e. encoded) strings restricted to the ASCII set.
        debug (bool): if True prints addition debug information.
        close (bool): If True (default) the HDF5 file is closed before
            returning. If False the file is left open.
        overwrite (bool): if True, when an HDF5 file with the same name is
            found, overwrite it. If False, save the new file by adding the
            suffix "new_copy" (and if a "_new_copy" file is already present
            overwrites it).

    For description and specs of the Photon-HDF5 format see:
    http://photon-hdf5.readthedocs.org/
    """
    comp_filter = tables.Filters(**compression)

    if h5_fname is None:
        basename, extension = os.path.splitext(data_dict['_filename'])
        if compression['complib'] == 'blosc':
            basename += '_blosc'
        h5_fname = basename + '.hdf5'

    if os.path.isfile(h5_fname) and not overwrite:
        basename, extension = os.path.splitext(h5_fname)
        h5_fname = basename + '_new_copy.hdf5'

    print('Saving: %s' % h5_fname)
    title = official_fields_specs['/'][0].encode()
    h5file = tables.open_file(h5_fname, mode="w", title=title,
                              filters=comp_filter)
    # Saving a file reference is useful in case of error
    data_dict.update(_data_file=h5file)

    ## Add root attributes
    for name, value in root_attributes.items():
        h5file.root._f_setattr(name, value)

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
            # Use metadata from the file except for creation time if
            # already present in `provenance`. i.e. the user-provided
            # creation time has priority over the filesystem one.
            orig_creation_time = provenance.get('creation_time', None)
            provenance.update(_get_file_metadata(orig_fname))
            if orig_creation_time is not None:
                provenance['creation_time'] = orig_creation_time

    ## Add identity metadata
    identity = get_identity(h5file)
    identity.update(software='phconvert',
                    software_version=__version__)
    data_dict['identity'].update(identity)

    ## Save everything to disk
    fields_descr = {k: v[0] for k, v in official_fields_specs.items()}
    if user_descr is not None:
        fields_descr.update(user_descr)
    _save_photon_hdf5_dict(h5file.root, data_dict,
                           fields_descr=fields_descr, debug=debug)
    h5file.flush()
    if validate:
        assert_valid_photon_hdf5_tables(h5file, strict=strict)
    if close:
        h5file.close()


def get_identity(h5file):
    full_h5filename = os.path.abspath(h5file.filename)
    h5filename = os.path.basename(full_h5filename)
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S")
    identity = dict(filename=h5filename,
                    filename_full=full_h5filename,
                    creation_time=creation_time,
                    format_name=root_attributes['format_name'],
                    format_version=LATEST_FORMAT_VERSION,
                    format_url=root_attributes['format_url'])
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
                # Load strings as native strings
                if isinstance(value, bytes) and not isinstance(value, str):
                    # value is a binary string and we are in python 3
                    value = value.decode('utf8')
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
            subgroup = h5file.create_group(group, key, title=_EMPTY.encode())
            dict_to_group(subgroup, value)
        else:
            if isinstance(value, str):
                # Save strings as binary strings
                # no-op on py2, convert to binary on py3
                value = value.encode()
            h5file.create_array(group, name=key, obj=value)
            # Set title through property access to work around pytable issue
            # under python 3 (https://github.com/PyTables/PyTables/issues/469)
            node = group._f_get_child(key)
            # Save a single space to workaround h5labview bug (see issue #4)
            node.title = _EMPTY.encode()  # saved as binary both on py2 and py3
    h5file.flush()

def load_photon_hdf5(filename, strict=True):
    assert os.path.isfile(filename)
    h5file = tables.open_file(filename)
    assert_valid_photon_hdf5_tables(h5file, strict=strict)
    return h5file.root

##
# Utility functions
#

def _get_version(h5file):
    """Return file format version string (unicode on both py2 and py3).

    Arguments:
        h5file (pytables File): pytables File object.
    """
    version = None
    format_name = root_attributes['format_name']

    # Check the root attributes first
    if 'format_name' in h5file.root._v_attrs:
        # All string are saved as binary strings
        assert h5file.root._v_attrs['format_name'] == format_name
        assert 'format_version' in h5file.root._v_attrs
        version = h5file.root._v_attrs['format_version'].decode()

    # Fall back to the identity group
    if version is None:
        # String fields are read as binary strings so we convert them
        # to native strings (binary -> unicode -> native)
        fformat = str(h5file.root.identity.format_name.read().decode())
        assert fformat == format_name
        version = h5file.root.identity.format_version.read().decode()

    if version is None:
        raise Invalid_PhotonHDF5('No version identification.')
    return version

def _check_version(filename):
    """Return file format version string (unicode on both py2 and py3).

    Arguments:
        filename (string): path of the data file.s
    """
    assert os.path.isfile(filename)
    with tables.open_file(filename) as h5file:
        version = _get_version(h5file)
    return version

def _sorted_photon_data_tables(h5file):
    """Return a sorted list of keys "photon_dataN", sorted by N.

    If there is only one "photon_data" (with no N) it returns the list
    ['photon_data'].
    """
    prefix = 'photon_data'
    ph_datas = [n for n in h5file.root._f_iter_nodes()
                if n._v_name.startswith(prefix)]

    ph_datas.sort(key=lambda x: x._v_name[len(prefix):])
    return ph_datas

def _sorted_photon_data(data_dict):
    """Return a sorted list of keys "photon_dataN", sorted by N.

    If there is only one "photon_data" key (with no N) it returns the list
    ['photon_data'].
    """
    prefix = 'photon_data'
    keys = [k for k in data_dict.keys() if k.startswith(prefix)]
    if len(keys) > 1:
        sorted_channels = sorted([int(k[len(prefix):]) for k in keys])
        keys = ['%s%d' % (prefix, ch) for ch in sorted_channels]
    return keys

def photon_data_mapping(h5file, name='timestamps'):
    """Return a mapping (OrderedDict) between ch and photon_data array.
    """
    from collections import OrderedDict
    mapping = OrderedDict()
    prefix = 'photon_data'
    for ph_data in _sorted_photon_data_tables(h5file):
        ph = ph_data._f_get_child(name)
        if ph.shape[-1] > 0:
            ch = int(ph_data._v_name[len(prefix):])
            mapping[ch] = ph
    return mapping

def _sanitize_data(data_dict):
    """Perform type conversions to strictly conform to Photon-HDF5 specs.

    Conversions implemented:

    - assure that fields in detectors_specs have same dtype as detectors
    - convert scalar fields that are array of size == 1 to scalars
    """
    ## detectors_specs conversions
    ph_data = data_dict[_sorted_photon_data(data_dict)[0]]
    dtype = ph_data['detectors'].dtype

    base = '/photon_data/measurement_specs/detectors_specs/'
    names = ['spectral_ch1', 'spectral_ch2', 'split_ch1', 'split_ch2',
             'polarization_ch1', 'polarization_ch2']
    cast_fields = [base + name for name in names]
    for item in _iter_hdf5_dict(data_dict):
        if item['meta_path'] in cast_fields:
            cdict = item['curr_dict']
            cdict[item['name']] = np.array(item['value'], dtype=dtype)

    ## scalar fields conversions
    for item in _iter_hdf5_dict(data_dict):
        if item['is_user']: continue
        if official_fields_specs[item['meta_path']][1] == 'scalar':
            if not np.isscalar(item['value']):
                try:
                    # sequences are converted to array then to scalar
                    scalar_value = np.asscalar(np.asarray(item['value']))
                except ValueError:
                    raise Invalid_PhotonHDF5('Cannot convert "%s" to scalar.'\
                                             % item['meta_path'])
                cdict = item['curr_dict']
                cdict[item['name']] = scalar_value

##
# Validation functions
#
class Invalid_PhotonHDF5(Exception):
    """Error raised when a file is not a valid Photon-HDF5 file.
    """
    pass

def _assert_valid(condition, msg, strict=True, norepeat=False, pool=None):
    """Assert `condition` and raise Invalid_PhotonHDF5(msg) on fail.

    Arguments:
        condition (bool): must evaluate to True for a valid Photon-HDF5 file.
        msg (string): meassage to be printed in case `condition` is False.
        strict (bool): if True, raise Invalid_PhotonHDF5 when `condition` is
            False. Else, print only a warning.
        norepeat (bool): if True, do not repeat the same message more than
            once. The message is considered printed if present in `pool`.
        pool (list): stores the message that have been printed (to avoid
            repetition). The first time pass an empty list, then keep passing
            the same list to avoid repetitions.

    Returns:
        Boolean, pass-through the input argument `condition`.
    """
    if norepeat:
        if msg in pool:
            return
        else:
            pool.append(msg)

    if not condition:
        if strict:
            raise Invalid_PhotonHDF5(msg)
        else:
            print('Photon-HDF5 WARNING: %s' % msg)
    return condition

def _assert_has_field(name, group, msg=None, msg_add=None, mandatory=True,
                      norepeat=False, pool=None, verbose=False):
    """Assert that field `name` is in `group`.

    Arguments:
        name (string): field name whose existence is being tested.
        group (tables.Group): group which should contain `name`.
        msg (string or None): optional message to be printed in case of
            missing field. When None a default meassage is printed.
        msg_add (string or None): an optional message to be added to the
            default message in case of missing field.
        mandatory (bool): if True, raise and Invalid_PhotonHDF5 error when
            the field is missing. If False, print only a warning message.
        norepeat (bool): if True, do not repeat the same message more than
            once. The message is considered printed if present in `pool`.
        pool (list): stores the message that have been printed (to avoid
            repetition). The first time pass an empty list, then keep passing
            the same list to avoid repetitions.

    Returns:
        Boolean, True if `name` exists otherwise False.
    """
    if verbose:
        print('Checking "%s" in %s.' % (name, group._v_pathname))
    if msg is None:
        msg = 'Missing field "%s" in "%s".' % (name, group._v_pathname)
    if msg_add is not None:
        msg += msg_add
    return _assert_valid(name in group, msg, mandatory, norepeat, pool)


def assert_valid_photon_hdf5_tables(datafile, strict=True, verbose=False,
                                    strict_description=True):
    """
    Assert the an HDF5 file follows the Photon-HDF5 specs.

    Raise an error when missing photon_data group, timestamps array and
    timestamps_unit.

    Arguments:
        strict (bool): if True, raise an error optional groups (i.e. setup
            and identity) are missing or lack mandatory fields. If False,
            print only a warning.
        verbose (bool): if True print details about the performed tests.
    """
    if isinstance(datafile, tables.File):
        h5file = datafile
        filename = h5file.filename
    elif isinstance(datafile, str):
        filename = datafile
        assert os.path.isfile(filename)
        h5file = tables.open_file(filename)
    else:
        msg = 'datafile must be a path (string) or a pytables File.'
        raise ValueError(msg)

    _assert_valid_fields(h5file, strict_description=strict_description,
                         verbose=verbose)
    _assert_mandatory_fields(h5file, verbose=verbose)

    pool = []
    for ph_data in _sorted_photon_data_tables(h5file):
        _check_photon_data_tables(ph_data, strict=strict, norepeat=True,
                                  pool=pool)

    _assert_setup(h5file, strict=strict, verbose=verbose)
    _assert_identity(h5file, strict=strict, verbose=verbose)

def _assert_setup(h5file, strict=True, verbose=False):
    """Assert that setup exists and contains the mandatory fields.
    """
    if _assert_has_field('setup', h5file.root, mandatory=strict,
                         verbose=verbose):
        mantatory_fields = ['num_pixels', 'num_spots', 'num_spectral_ch',
                            'num_polarization_ch', 'num_split_ch',
                            'modulated_excitation', 'lifetime']
        for name in mantatory_fields:
            _assert_has_field(name, h5file.root.setup, mandatory=strict,
                              verbose=verbose)

def _assert_identity(h5file, strict=True, verbose=False):
    """Assert that identity group exists and contains the mandatory fields.
    """
    if _assert_has_field('identity', h5file.root, mandatory=strict,
                         verbose=verbose):
        mantatory_fields = ['format_name', 'format_version', 'format_url',
                            'software', 'software_version', 'creation_time']
        for name in mantatory_fields:
            _assert_has_field(name, h5file.root.identity, mandatory=strict,
                              verbose=verbose)

        optional_fields = ['author', 'author_affiliation']
        for name in optional_fields:
            _assert_has_field(name, h5file.root.identity, mandatory=False,
                              verbose=verbose)


def _assert_mandatory_fields(h5file, verbose=False):
    """Assert that the basic mandatory fields are present.
    """
    _assert_has_field('acquisition_duration', h5file.root, verbose=verbose)
    _assert_has_field('description', h5file.root, verbose=verbose)
    if not 'photon_data0' in h5file.root:
        _assert_has_field('photon_data', h5file.root, verbose=verbose)


def _assert_valid_fields(h5file, strict_description=True, verbose=False):
    """Assert compliance of field names, descriptions and data types.

    Test that all the field names, the descriptions (TITLE attribute) and
    data types are compliant with the Photon-HDF5 specs.
    """
    for node in h5file.root._f_walknodes():
        metaname = pathname = node._v_pathname
        if metaname.startswith('/photon_data'):
            metaname = pathname[:len('/photon_data')]
        title = node._v_title
        if verbose:
            print('- Checking name, description and type: "%s".' % pathname)

        ## Test non empty title string
        msg = 'Empty TITLE attribute for "%s"' % pathname
        _assert_valid(len(title) > 0, msg, strict=strict_description)

        ## Test description is a binary string
        # This depends on how pytbales loads the string and fails for some
        # fields (e.g. user fields in BH file) under python 3.
        # The test is disable for the time being.
        #msg = 'TITLE attribute for "%s" is not a binary string.' % pathname
        #_assert_valid(isinstance(title, bytes), msg, strict=strict_description)

        if pathname.endswith('/user') or '/user/' in pathname:
            pass
        else:
            # Check field names
            msg = 'Wrong field name "%s".' % metaname
            _assert_valid(metaname in official_fields_specs.keys(), msg)

            # Check fields use official description
            msg = 'Description (TITLE) for "%s" not compliant.' % metaname
            _assert_valid(title.decode() == official_fields_specs[metaname][0],
                          msg)

            # Check fields have correct type
            official_type = official_fields_specs[metaname][1]

            if official_type == 'group':
                msg = '"%s" must be a group.' % pathname
                _assert_valid(isinstance(node, tables.Group), msg)
            elif official_type == 'string':
                msg = 'Data in "%s" is not a binary string.' % pathname
                _assert_valid(node.ndim == 0, msg)
                _assert_valid(node.dtype.kind == 'S', msg)
                _assert_valid(isinstance(node.read(), bytes), msg)
            elif official_type == 'scalar':
                msg = '"%s" must be scalar.' % pathname
                _assert_valid(node.ndim == 0, msg)
            elif official_type == 'array':
                msg = '"%s" must be an array.' % pathname
                _assert_valid(node.ndim >= 0, msg)
                # NOTE: some fields saved as numpy "scalar" arrays have
                #       ndim == 0, still they are numpy arrays.
            else:
                raise ValueError('Wrong type in JSON specs.')

def _check_photon_data_tables(ph_data, strict=True, norepeat=False, pool=None,
                              verbose=False):
    """Assert that the photon_data group follows the Photon-HDF5 specs.
    """
    _assert_has_field('timestamps', ph_data, verbose=verbose)
    _assert_has_field('timestamps_specs', ph_data, verbose=verbose)
    _assert_has_field('timestamps_unit', ph_data.timestamps_specs,
                      verbose=verbose)

    if not _assert_has_field('measurement_specs', ph_data, mandatory=False,
                             verbose=verbose, norepeat=norepeat, pool=pool):
        return

    spectral_meas_types = ['smFRET', 'smFRET-usALEX', 'smFRET-usALEX-3c',
                           'smFRET-nsALEX']
    meas_specs = ph_data.measurement_specs
    msg = 'Missing "measurement_type" in "%s".' % meas_specs._v_pathname
    _assert_has_field('measurement_type', meas_specs, msg, verbose=verbose)

    meas_type = meas_specs.measurement_type.read().decode()
    if verbose:
        print('* Measurement type: "%s"' % meas_type)
    _assert_valid(meas_type in spectral_meas_types,
                  msg='Unkwnown measurement type "%s"' % meas_type)

    # At this point we have a valid measurement_type
    # Any missing field will raise an error (regardless of `strict`).
    msg = '\nThis field is mandatory for "%s" data.' % meas_type
    kwargs = dict(msg_add=msg, verbose=verbose)
    _assert_has_field('spectral_ch1', meas_specs.detectors_specs, **kwargs)
    _assert_has_field('spectral_ch2', meas_specs.detectors_specs, **kwargs)

    if meas_type in ['smFRET-usALEX', 'smFRET-usALEX-3c']:
        _assert_has_field('alex_period', meas_specs, **kwargs)

    if meas_type == 'smFRET-nsALEX':
        _assert_has_field('laser_repetition_rate', meas_specs, **kwargs)
        _assert_has_field('nanotimes', ph_data, **kwargs)
        _assert_has_field('nanotimes_specs', ph_data, **kwargs)
        for name in ['tcspc_unit', 'tcspc_range', 'tcspc_num_bins']:
             _assert_has_field(name, ph_data.nanotimes_specs, **kwargs)


def print_attrs(node, which='user'):
    """Print the HDF5 attributes for `node_name`.

    Parameters:
        node (pytables node): node whose attributes will be printed.
            Can be either a group or a leaf-node.
        which (string): Valid values are 'user' for user-defined attributes,
            'sys' for pytables-specific attributes and 'all' to print both
            groups of attributes. Default 'user'.
    """
    print('List of attributes for:\n  %s\n' % node)
    for attr in node._v_attrs._f_list(which):
        print('\t%s' % attr)
        print('\t    %s' % repr(node._v_attrs[attr]))


def print_children(group):
    """Print all the sub-groups in `group` and leaf-nodes children of `group`.

    Parameters:
        group (pytables group): the group to be printed.
    """
    for name, value in group._v_children.items():
        if isinstance(value, tables.Group):
            content = '(Group)'
        else:
            content = value.read()
        title = value._v_title
        if isinstance(title, bytes):
            title = title.decode()
        print(name)
        print('    Content:     %s' % content)
        print('    Description: %s\n' % title)


del print_function
