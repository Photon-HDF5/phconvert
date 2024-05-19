#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2017 Antonino Ingargiola <tritemio@gmail.com>
#
"""

The module `hdf5` defines functions to save and validate Photon-HDF5 files.
The main two functions in this module are:

- :func:`save_photon_hdf5` to saves data from a dictionary to Photon-HDF5.
- :func:`assert_valid_photon_hdf5` to validate if a HDF5 file is valid
  Photon-HDF5.

This module also provides functions to save free-form dict to HDF5
(:func:`dict_to_group`) and read a HDF5 group into a dict
(:func:`dict_from_group`).
Finally there are utility functions to easily print HDF5 nodes and attributes
(:func:`print_children`, :func:`print_attrs`).

For more info see:
`Writing Photon-HDF5 files <http://photon-hdf5.readthedocs.org/en/latest/writing.html>`_.

"""

import os
import time
import re
from textwrap import dedent
import tables
import numpy as np


from .metadata import (official_fields_specs, root_attributes,
                       LATEST_FORMAT_VERSION)

from phconvert._version import version as __version__
# __version__ = '0.9.1.post-release'

# Empty description string (workaround for h5labview)
_EMPTY = ' '


# Names of mandatory fields in the setup group
_setup_mantatory_fields = ['num_pixels', 'num_spots', 'num_spectral_ch',
                           'num_polarization_ch', 'num_split_ch',
                           'modulated_excitation', 'lifetime',
                           'excitation_alternated']

# Names of mandatory fields in the identity group
_identity_mantatory_fields = ['format_name', 'format_version', 'format_url',
                              'software', 'software_version', 'creation_time']

# Names of fields in /setup/detectors
_detectors_group_fields = ('id', 'id_hardware', 'counts', 'dcr', 'afterpulsing',
                           'positions', 'spot', 'module', 'label', 'tcspc_unit',
                           'tcspc_num_bins')

# All valid measurement_type strings
valid_meas_types = ['smFRET', 'smFRET-usALEX', 'smFRET-usALEX-3c',
                    'smFRET-nsALEX', 'generic']



def _metapath(fullpath):
    """Normalize a HDF5 path by removing trailing digits after "photon_data".
    """
    metapath = fullpath
    if fullpath.startswith('/photon_data'):
        # Remove eventual digits after /photon_data
        pattern = '/photon_data[0-9]*(.*)'
        metapath = ('/photon_data' +
                    re.match(pattern, fullpath).group(1))
    return metapath


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

    meta_path = _metapath(full_path)
    is_phdata = False
    if full_path.startswith('/photon_data'):
        if len(chunks) == 3 and not name.endswith('_specs'):
            is_phdata = True

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

    if name in h5file.get_node(group):
        # Remove a pre-existing node with the same name
        h5file.remove_node(group, name)
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
                print('WARNING: missing description for "%s"' %
                      item['meta_path'])

        if isinstance(item['value'], tables.Array):
            # If the data is already a pytable array set only the title
            item['value'].set_attr('TITLE', item['description'].encode())
        elif isinstance(item['value'], dict):
            if item['name'] in h5file.get_node(item['group_path']):
                # If group exists only set TITLE
                grp = h5file.get_node(item['group_path'], item['name'])
                grp._f_setattr('TITLE', item['description'].encode())
            else:
                h5file.create_group(item['group_path'], item['name'],
                                    title=item['description'].encode())
        else:
            # Node is not a tables.Array or a Group, write it
            _h5_write_array(item['group_path'], item['name'],
                            obj=item['value'], descr=item['description'],
                            chunked=item['is_phdata'], h5file=group._v_file)


def save_photon_hdf5(data_dict,
                     h5_fname = None,
                     h5file = None,
                     user_descr = None,
                     overwrite = False,
                     compression = dict(complevel=6, complib='zlib'),
                     close = True,
                     validate = True,
                     warnings = True,
                     skip_measurement_specs = False,
                     require_setup = True,
                     debug = False):
    """
    Saves the dict `data_dict` in the Photon-HDF5 format.

    This function requires the data to be saved as ``data_dict`` argument.
    The data needs to have the hierarchical structure of a Photon-HDF5 file.
    For the purpose, we use a standard python dictionary: each keys is
    a Photon-HDF5 field name and each value contains data (e.g. array,
    string, etc..) or another dictionary (in which case, it represents an
    HDF5 sub-group). Similarly, sub-dictionaries contain data or
    other dictionaries, as needed to represent the hierarchy
    of Photon-HDF5 files.

    Features of this function:

    - Checks that all field names are valid Photon-HDF5 field names.
    - Checks that all field type match the Photon-HDF5 specs (scalar, array,
      or string).
    - Populates automatically the identity group with filename, software,
      version and file creation date.
    - Populates automatically the provenance group with info on the original
      data file (if it can be found on disk): creation and modification date,
      path.
    - Computes field `acquisition_duration` when not provided
      (single-spot data only).

    Minimal fields required to create a Photon-HDF5 file:

    - `/description` (string)
    - `/photon_data/timestamps` (array)
    - `/photon_data/timestamps_specs/timestamps_unit` (scalar float)
    - `/setup/num_pixels` (int): number of detectors
    - `/setup/num_spots` (int): number of excitation/detection spots
    - `/setup/num_spectral_ch` (int): number of detection spectral bands
    - `/setup/num_polarization_ch` (int): number of detected polarization states
    - `/setup/num_split_ch` (int): number of beam split channels
    - `/setup/modulated_excitation` (bool): True if there is any form of intensity
      or polarization modulation or interleaved excitation (PIE or nsALEX).
      This field has become obsolete in version 0.5 and maintained only for
      compatibility.
    - `/setup/excitation_alternated` (array of bool): New in version 0.5.
      Values are True if the respective excitation source is
      intensity-modulated. In us-ALEX both sources are alternated,
      while in PAX measurements only one source is alternated.
    - `/setup/lifetime` (bool): True if dataset contains TCSPC data.

    See also
    `Writing Photon-HDF5 files <http://nbviewer.ipython.org/github/Photon-HDF5/phconvert/blob/master/notebooks/Writing%20Photon-HDF5%20files.ipynb>`__.

    As a side effect `data_dict` is modified by adding the key
    '_data_file' containing a reference to the pytables file.

    Arguments:
        data_dict (dict): the dictionary containing the photon data.
            The keys must strings matching valid Photon-HDF5 paths.
            The values must be scalars, arrays, strings or another dict.
        h5_fname (string or None): file name for the output Photon-HDF5 file.
            If None and h5file is also None, the file name is taken from
            ``data_dict['_filename']`` with extension changed to '.hdf5'.
        h5file (pytables.File or None): an already open and writable HDF5
            file to use as container. This argument can be used to complete
            an HDF5 file already containing some arrays, or to update
            an already existing Photon-HDF5 file in-place.
            For more info see note below.
        user_descr (dict or None): dictionary of descriptions (strings) for
            user-defined fields. The keys must be strings representing
            the full HDF5 path of each field. The values must be
            binary (i.e. encoded) strings restricted to the ASCII set.
        overwrite (bool): if True, a pre-existing HDF5 file with same name is
            overwritten. If False, save the new file by adding the
            suffix "new_copy" (and if a "_new_copy" file is already present
            overwrites it).
        compression (dict): a dictionary containing the compression type
            and level. Passed to pytables `tables.Filters()`.
        close (bool): If True (default) the HDF5 file is closed before
            returning. If False the file is left open.
        validate (bool): if True, after saving perform a validation step
            raising an error if the specs are not followed.
        warnings (bool): if True, print warnings for important optional fields
            that are missing. If False, don't print warnings.
        skip_measurement_specs (bool): if True don't print any warning for
            missing measurement_specs group.
        require_setup (bool): if True, raises an error if some mandatory
            fields in /setup are missing. If False, allows missing setup
            fields (or missing setup altogether). Use False when saving
            only detectors' dark counts.
        debug (bool): if True prints additional debug information.

    For description and specs of the Photon-HDF5 format see:
    http://photon-hdf5.readthedocs.org/

    Note:
        The argument `h5file` accepts an already open HDF5 file for storage.
        This allows completing a partially written file (for example
        containing only photon_data arrays) or correcting and already complete
        Photon-HDF5 file. When using `h5file`, you need to pass a full
        `data_dict` structure as usual. If you don't want update an array,
        put in `data_dict` a reference to the existing pytables array
        (instead of using a numpy array). Fields containing numpy arrays
        will be overwritten. Fields containing pytables Array (including
        CArray or EArray) will be left unmodified. In either cases the TITLE
        attribute is always updated.
    """
    comp_filter = tables.Filters(**compression)

    ## Compute file names
    if h5file is not None:
        _msg = 'Argument `h5file` must be None or a `tables.File` object.'
        assert isinstance(h5file, tables.File), _msg
        h5_fname = h5file.filename
    else:
        if h5_fname is None:
            basename, extension = os.path.splitext(data_dict['_filename'])
            if compression['complib'] == 'blosc':
                basename += '_blosc'
            h5_fname = basename + '.hdf5'

        if os.path.isfile(h5_fname) and not overwrite:
            basename, extension = os.path.splitext(h5_fname)
            h5_fname = basename + '_new_copy.hdf5'

    ## Prefill and fix user-provided data_dict
    _populate_provenance(data_dict)
    if 'setup' in data_dict:
        _populate_setup(data_dict)
    _sanitize_data(data_dict, require_setup)
    _compute_acquisition_duration(data_dict)

    ## Create the HDF5 file
    print('Saving: %s' % h5_fname)
    title = official_fields_specs['/'][0].encode()
    if h5file is None:
        h5file = tables.open_file(str(h5_fname), mode="w", title=title,
                                  filters=comp_filter)
    else:
        # If file already opened set only the root-node TITLE
        h5file.set_node_attr('/', attrname='TITLE', attrvalue=title)

    # Saving a file reference, useful in case of errors
    data_dict.update(_data_file=h5file)

    ## Identity info needs to be added after the file is created
    _populate_identity(data_dict, h5file)

    ## Save root attributes
    for name, value in root_attributes.items():
        h5file.root._f_setattr(name, value)

    ## Save everything else to disk
    fields_descr = {k: v[0] for k, v in official_fields_specs.items()}
    if user_descr is not None:
        fields_descr.update(user_descr)
    _save_photon_hdf5_dict(h5file.root, data_dict,
                           fields_descr=fields_descr, debug=debug)
    h5file.flush()

    ## Validation
    try:
        if validate:
            kwargs = dict(skip_measurement_specs=skip_measurement_specs,
                          warnings=warnings, require_setup=require_setup)
            assert_valid_photon_hdf5(h5file, **kwargs)
    finally:
        if close:
            h5file.close()


def _is_mutispot(h5root_or_dict):
    return 'photon_data' not in h5root_or_dict


def _populate_identity(data_dict, h5file):
    """Populate identity metadata adding info from the newly created file.
    """
    identity = _get_identity(h5file)
    identity.update(software='phconvert',
                    software_version=__version__)
    if 'identity' not in data_dict:
        data_dict['identity'] = {}
    data_dict['identity'].update(identity)


def _populate_provenance(data_dict):
    """Try to find the original data file to fill provenance fields.
    """
    if 'provenance' not in data_dict:
        return

    provenance = data_dict['provenance']
    orig_fname = None
    for fn in ['filename', 'filename_full']:
        if fn in provenance and os.path.isfile(provenance[fn]):
            orig_fname = provenance[fn]
            break

    if orig_fname is None:

        msg = """\
            WARNING: Could not locate original file '%s'.
                     File info in provenance group will not be added.
            """ % provenance['filename']
        print(dedent(msg))
    else:
        # Use metadata from the file except for creation time if
        # already present in `provenance`. i.e. the user-provided
        # creation time has priority over the filesystem one.
        orig_creation_time = provenance.get('creation_time', None)
        provenance.update(_get_file_metadata(orig_fname))
        if orig_creation_time is not None:
            provenance['creation_time'] = orig_creation_time


def _compute_acquisition_duration(data_dict):
    """Compute acquisition_duration if not present. Single-spot only.
    """
    if 'acquisition_duration' in data_dict:
        return
    try:
        ph_data = data_dict['photon_data']
        timestamps = ph_data['timestamps']
        ts_specs = ph_data['timestamps_specs']
        timestamps_unit = ts_specs['timestamps_unit']
    except KeyError:
        # This happens in multi-spot or when some fields are missing
        # Missing fields will later yield an error during validation.
        pass
    else:
        assert np.isreal(timestamps_unit)
        acquisition_duration = ((timestamps.max() - timestamps.min()) *
                                timestamps_unit)
        data_dict['acquisition_duration'] = np.round(acquisition_duration, 1)


def _populate_setup(data_dict):
    _populate_detectors_group(data_dict)
    _autofill_laser_rep_rates(data_dict)


def _autofill_laser_rep_rates(data_dict):
    ph_data = data_dict[_sorted_photon_data(data_dict)[0]]
    if 'measurement_specs' not in ph_data:
        return
    meas_specs = ph_data['measurement_specs']
    if meas_specs['measurement_type'] != 'smFRET-nsALEX':
        return
    laser_rep_rate = 'laser_repetition_rate'
    laser_rep_rates = laser_rep_rate + 's'
    if laser_rep_rate not in meas_specs:
        msg = """\
            Measurement type 'smFRET-nsALEX' is missing the field
            'laser_repetition_rate' in 'measurement_specs'."""
        raise Invalid_PhotonHDF5(dedent(msg))
    setup = data_dict['setup']
    if laser_rep_rates not in setup:
        # Create the array of laser repetition rates
        # Note: all lasers repetition rates are assumed equal
        setup[laser_rep_rates] = [meas_specs[laser_rep_rate]] * 2


def _populate_detectors_group(data):
    det_grp = data['setup'].get('detectors', {})
    if 'id' in det_grp and 'id_hardware' in det_grp and 'counts' in det_grp:
        # nothing to do
        return

    # Note that detectors with 0 counts cannot be detected
    det_val, det_cnts, spot = [], [], []
    for i, ph_data in enumerate(_sorted_photon_data(data)):
        detectors = data[ph_data]['detectors'][:]  # numpy or pytables array
        vals, counts = np.unique(detectors, return_counts=True)
        det_val.extend(vals)
        det_cnts.extend(counts)
        spot.extend([i] * len(vals))

    det_grp.setdefault('id', np.array(det_val))
    det_grp.setdefault('id_hardware', np.array(det_val))
    det_grp['counts'] = np.array(det_cnts)
    if _is_mutispot(data):
        det_grp['spot'] = np.array(spot)
    data['setup']['detectors'] = det_grp


def _get_identity(h5file):
    """Return a dict with identity information for `h5file`.
    """
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
    # see https://docs.python.org/3/library/os.path.html#os.path.getctime)
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


def load_photon_hdf5(filename, **kwargs):
    """Open a Photon-HDF5 file in pytables, validating it.

    Additional arguments are passed to :func:`assert_valid_photon_hdf5`.

    Returns:
        The root group of the HDF5 file.
    """
    assert os.path.isfile(filename)
    h5file = tables.open_file(filename)
    assert_valid_photon_hdf5(h5file, **kwargs)
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
        filename (string): path of the data file.
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

    if len(ph_datas) > 1:
        ph_datas.sort(key=lambda x: int(x._v_name[len(prefix):]))
    return ph_datas


def _sorted_photon_data(data_dict):
    """Return a sorted list of keys "photon_dataN", sorted by N.

    If there is only one "photon_data" key (with no N) it returns the list
    ['photon_data'].
    """
    prefix = 'photon_data'
    keys = [k for k in data_dict.keys() if k.startswith(prefix)]
    if len(keys) > 1:
        keys.sort(key=lambda x: int(x[len(prefix):]))
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


def _is_sequence(obj):
    is_sequence = False
    if isinstance(obj, tuple) or isinstance(obj, list):
        is_sequence = True
    elif isinstance(obj, np.ndarray):
        is_sequence = obj.ndim > 0
    return is_sequence


def _normalize_bools(data_dict):
    """Cast bools (both scalars or in sequences) to integers."""
    for name, value in data_dict.items():
        if isinstance(value, dict):
            _normalize_bools(value)
        else:
            if isinstance(value, bool):
                data_dict[name] = int(value)
            elif _is_sequence(value) and isinstance(value[0], bool):
                data_dict[name] = np.asarray(value, dtype='uint8')


def _normalize_detectors_specs(data_dict):
    base = '/photon_data/measurement_specs/detectors_specs/'
    names = ['spectral_ch1', 'spectral_ch2', 'split_ch1', 'split_ch2',
             'polarization_ch1', 'polarization_ch2']
    cast_fields = [base + name for name in names]

    # Retrive the detectors' dtype (from first photon_data group in multi-spot)
    ph_data = data_dict[_sorted_photon_data(data_dict)[0]]
    if 'detectors' not in ph_data:
        return
    dtype = ph_data['detectors'].dtype

    for item in _iter_hdf5_dict(data_dict):
        if item['meta_path'] in cast_fields:
            cdict = item['curr_dict']
            cdict[item['name']] = np.array(item['value'],
                                           dtype=dtype, ndmin=1)


def _normalize_setup_arrays(data_dict):
    """Make sure arrays of float in setup are arrays of floats."""
    if 'setup' not in data_dict:
        return
    # Convert sequences of strings in 'setup' in arrays of floats
    # Useful when input is from YAML whose parser retrives floats a strings
    setup = data_dict['setup']
    # Arrays of float fields in setup group
    names_aof = ['detection_wavelengths', 'excitation_wavelengths',
                 'excitation_input_powers', 'detection_polarizations',
                 'excitation_intensity', 'detection_split_ch_ratios',
                 'laser_repetition_rates']
    for name in names_aof:
        if name in setup:
            setup[name] = np.array([float(v) for v in setup[name]],
                                   dtype=float)


def _normalize_detectors_group(data_dict):
    """Convert fields in /setup/detectors to `numpy.ndarray`."""
    if 'setup' not in data_dict:
        return
    det_grp = data_dict['setup']['detectors']
    for name in _detectors_group_fields:
        if name in det_grp:
            det_grp[name] = np.asarray(det_grp[name])


def _convert_scalar_item(item):
    """Cast a scalar item (from _iter_hdf5_dict) to scalar."""
    # Special case for scalar fields which are string in data_dict.
    # thus requiring a conversion. This happens when the YAML parser
    # fails to detect floats in exponential form.
    scalar_value = item['value']
    if isinstance(item['value'], str):
        msg = """\
        Wrong data type: field `%s` must be a scalar.
                         Instead it is the string %s
                         which I'm unable to convert to int or float.\
        """ % (item['meta_path'], repr(item['value']))
        try:
            scalar_value = int(item['value'])
        except ValueError:
            pass
            try:
                scalar_value = float(item['value'])
            except ValueError:
                raise Invalid_PhotonHDF5(dedent(msg))

    # If a scalar field is 1-element sequence, convert it to scalar
    if not np.isscalar(item['value']):
        try:
            # sequences are converted to array then to scalar
            scalar_value = np.asscalar(np.asarray(item['value']))
        except ValueError:
            raise Invalid_PhotonHDF5('Cannot convert "%s" to scalar.'
                                     % item['meta_path'])
    return scalar_value


def _normalize_scalars(data_dict):
    """Make sure all scalar fields are scalars."""
    # scalar fields conversions
    for item in _iter_hdf5_dict(data_dict):
        if item['is_user']:
            continue
        if official_fields_specs[item['meta_path']][1] == 'scalar':
            scalar_value = _convert_scalar_item(item)
            curr_dict = item['curr_dict']
            curr_dict[item['name']] = scalar_value


def _sanitize_data(data_dict, require_setup=True):
    """Perform type conversions to strictly conform to Photon-HDF5 specs.

    Conversions implemented:

    - assure that fields in detectors_specs have same dtype as detectors
    - convert scalar fields that are array of size == 1 to scalars
    - cast bools or sequences of bools to integers
    - convert scalar fields which are strings to numbers
    - for some setup fields, convert sequences of strings to arrays of floats
    - convert /setup/detectors fields into numpy arrays.
    """
    def _assert_has_key(dict_, key, dict_name):
        if key not in dict_:
            raise Invalid_PhotonHDF5('missing %s in %s.' % (key, dict_name))

    for ph_data_name in _sorted_photon_data(data_dict):
        ph_data = data_dict[ph_data_name]
        for name in ['timestamps', 'timestamps_specs']:
            _assert_has_key(ph_data, name, ph_data_name)

        ts_specs = ph_data['timestamps_specs']
        _assert_has_key(ts_specs, 'timestamps_unit', 'timestamps_specs')

    if require_setup:
        if 'setup' not in data_dict:
            raise Invalid_PhotonHDF5('missing setup group.')
        setup = data_dict['setup']
        for name in _setup_mantatory_fields:
            if name not in setup:
                raise Invalid_PhotonHDF5('missing "%s" in setup group.' % name)

    # Cast booleans to integers
    _normalize_bools(data_dict)
    # Cast fields in detectors_specs
    _normalize_detectors_specs(data_dict)
    # Cast arrays-of-floats fields in setup group
    _normalize_setup_arrays(data_dict)
    # Cast scalar fields to scalar
    _normalize_scalars(data_dict)
    # Convert fields in /setup/detectors to numpy arrays
    _normalize_detectors_group(data_dict)


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
        msg = 'Missing field "%s" in "%s". ' % (name, group._v_pathname)
    if msg_add is not None:
        msg += msg_add
    return _assert_valid(name in group, msg, mandatory, norepeat, pool)


def _assert_valid_detectors(h5file):
    detectors = h5file.root.setup.detectors
    det_ids = detectors.id.read()
    if 'counts' in detectors:
        det_counts = detectors.counts.read()
    if _is_mutispot(h5file.root):
        spot = detectors.spot.read()
    else:
        spot = np.zeros(len(det_ids), dtype='uint8')

    m = 'detectors/%s length (%d) is not equal to the number of detectors (%d).'
    for field in _detectors_group_fields:
        if field in detectors:
            values = detectors._f_get_child(field)
            _assert_valid(len(values) == len(det_ids),
                          msg=m % (field, len(values), len(det_ids)))

    msg = 'Detector %d in spot %d not found in detectors/id.'
    msgc = 'Wrong counts (%d instead of %d) for detector %d in spot %d.'
    for i, ph_data in enumerate(_sorted_photon_data_tables(h5file)):
        if 'detectors' not in ph_data:
            break
        vals, cnts = np.unique(ph_data.detectors[:], return_counts=True)
        det_ids_spot = det_ids[spot == i]
        for v, c in zip(vals, cnts):
            _assert_valid(v in det_ids_spot, msg=msg % (v, i))
            if 'counts' in detectors:
                csaved = det_counts[spot == i][det_ids_spot == v]
                _assert_valid(c == csaved, msg=msgc % (csaved, c, v, i))


def assert_valid_photon_hdf5(datafile, warnings=True, verbose=False,
                             strict_description=True, require_setup=True,
                             skip_measurement_specs=False):
    """
    Asserts that ``datafile`` follows the Photon-HDF5 specs.

    If the input datafile does not follow the specifications, it raises the
    ``Invalid_PhotonHDF5`` exception, with a message indicating the cause of
    the error.

    This function checks that:

    - all fields are valid Photon-HDF5 names
    - all fields have valid descriptions
    - all mandatory fields are present
    - if /setup/lifetime is True (i.e. 1), assures
      that nanotimes and nanotimes_specs are present

    Arguments:
        datafile (string or tables.File): input data file to be validated
        warnings (bool): if True, print warnings for important optional fields
            that are missing. If False, don't print warnings.
        verbose (bool): if True print details about the performed tests.
        strict_description (bool): if True consider a non-conforming
            description (TITLE) a specs violation.
        require_setup (bool): if True, raises an error if some mandatory
            fields in /setup are missing. If False, allows missing setup
            fields (or missing setup altogether).
        skip_measurement_specs (bool): if True don't print any warning for
            missing measurement_specs group.
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
    _assert_has_field('acquisition_duration', h5file.root, verbose=verbose)
    _assert_has_field('description', h5file.root, verbose=verbose)
    if require_setup:
        _assert_setup(h5file, warnings=warnings, verbose=verbose)
    _assert_identity(h5file, warnings=warnings, verbose=verbose)

    pool = []
    kwargs = dict(setup=h5file.root.setup, pool=pool, norepeat=True,
                  skip_measurement_specs=skip_measurement_specs)
    for ph_data in _sorted_photon_data_tables(h5file):
        _check_photon_data_tables(ph_data, **kwargs)


def _assert_setup(h5file, warnings=True, strict=True, verbose=False):
    """Assert that setup exists and contains the mandatory fields.
    """
    if not _assert_has_field('setup', h5file.root, mandatory=strict,
                             verbose=verbose):
        return
    for name in _setup_mantatory_fields:
        _assert_has_field(name, h5file.root.setup, mandatory=strict,
                          verbose=verbose)
    if 'detectors' in h5file.root.setup:
        _assert_valid_detectors(h5file)
    if warnings:
        optional_fields = ['excitation_wavelengths', 'detection_wavelengths']
        for name in optional_fields:
            _assert_has_field(name, h5file.root.setup, mandatory=False,
                              verbose=verbose)


def _assert_identity(h5file, warnings=True, strict=True, verbose=False):
    """Assert that identity group exists and contains the mandatory fields.
    """
    if _assert_has_field('identity', h5file.root, mandatory=strict,
                         verbose=verbose):
        for name in _identity_mantatory_fields:
            _assert_has_field(name, h5file.root.identity, mandatory=strict,
                              verbose=verbose)
        if not warnings:
            return
        optional_fields = ['author', 'author_affiliation']
        for name in optional_fields:
            _assert_has_field(name, h5file.root.identity, mandatory=False,
                              verbose=verbose)


def _assert_valid_fields(h5file, strict_description=True, verbose=False):
    """Assert compliance of field names, descriptions and data types.

    Test that all the field names, the descriptions (TITLE attribute) and
    data types are compliant with the Photon-HDF5 specs.
    """
    for node in h5file.root._f_walknodes():
        pathname = node._v_pathname
        metaname = _metapath(pathname)
        title = node._v_title
        if verbose:
            print('- Checking name, description and type: "%s".' % pathname)

        ## Test non empty title string
        msg = 'Empty TITLE attribute for "%s"' % pathname
        _assert_valid(len(title) > 0, msg, strict=strict_description)

        ## Test description is a binary string
        # This depends on how pytables loads the string and fails for some
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
                          msg, strict=strict_description)

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
                _assert_valid(node.ndim >= 1, msg)
                # check that photon_data arrays are strictly 1D
                ph_data_arrays = ('timestamps', 'detectors', 'nanotimes',
                                  'particles')
                if node._v_name in ph_data_arrays:
                    msg = ('The array /photon_data/%s must be 1D. '
                           'It is %dD instead.' % (node._v_name, node.ndim))
                    _assert_valid(node.ndim == 1, msg)
            else:
                raise ValueError('Wrong type in JSON specs.')


def _check_photon_data_tables(ph_data, setup, norepeat=False, pool=None,
                              skip_measurement_specs=False, verbose=False):
    """Assert that the photon_data group follows the Photon-HDF5 specs.
    """
    _assert_has_field('timestamps', ph_data, verbose=verbose)
    _assert_has_field('timestamps_specs', ph_data, verbose=verbose)
    _assert_has_field('timestamps_unit', ph_data.timestamps_specs,
                      verbose=verbose)

    if 'measurement_specs' not in ph_data:
        if not skip_measurement_specs:
            # Called to print a warning
            _assert_has_field('measurement_specs', ph_data, mandatory=False,
                              verbose=verbose, norepeat=norepeat, pool=pool)
        return

    meas_specs = ph_data.measurement_specs
    msg = 'Missing "measurement_type" in "%s".' % meas_specs._v_pathname
    _assert_has_field('measurement_type', meas_specs, msg, verbose=verbose)

    meas_type = meas_specs.measurement_type.read().decode()
    if verbose:
        print('* Measurement type: "%s"' % meas_type)
    _assert_valid(meas_type in valid_meas_types,
                  msg='Unknown measurement type "%s"' % meas_type)

    # At this point we have a valid measurement_type
    # We will check (and raise an error) for any missing field.
    msg = '\nThis field is mandatory for "%s" data.' % meas_type
    kwargs = dict(msg_add=msg, verbose=verbose)

    # Read number of channels in each branch
    num_ch = dict(spectral=setup.num_spectral_ch.read(),
                  split=setup.num_split_ch.read(),
                  polarization=setup.num_polarization_ch.read())

    # Check for spectral channels
    if meas_type in ('smFRET', 'smFRET-usALEX', 'smFRET-nsALEX'):
        _msg = ('%s measurement requires /setup/num_spectral_ch = 2 '
                '(not %d).' % (meas_type, num_ch['spectral']))
        _assert_valid(num_ch['spectral'] == 2, msg=_msg)
        for branch in ('split', 'polarization'):
            _msg = ('%s measurement requires /setup/num_%s_ch = 1 '
                    '(not %d).' % (meas_type, branch, num_ch[branch]))
            _assert_valid(num_ch[branch] == 1, msg=_msg)
    if meas_type == 'smFRET-usALEX-3c':
        _msg = ('%s measurement requires /setup/num_spectral_ch = 3 '
                '(not %d).' % (meas_type, num_ch['spectral']))
        _assert_valid(num_ch['spectral'] == 3, msg=_msg)

    # handle case of no detectors array and no detectors_specs
    # which is possible when measurement_type is 'generic'
    if 'detectors' in ph_data:
        det_specs = meas_specs.detectors_specs
        # Check for spectral/split/polarization channels in detectors_specs
        for feature, nch in num_ch.items():
            if nch > 1:
                for i in range(nch):
                    _assert_has_field('%s_ch%d' % (feature, i + 1),
                                      det_specs, **kwargs)

    # Check presences of repetititon rate with pulsed lasers
    msg0 = "According to /setup/excitation_cw some lasers are pulsed.\n"
    if not all(setup.excitation_cw[:]):
        m = "However, the field '/setup/laser_repetition_rates' is missing."
        _assert_has_field('laser_repetition_rates', setup, msg_add=msg0 + m)
        m = ("However, the field 'measurement_specs/laser_repetition_rate' "
             "is missing.")
        _assert_has_field('laser_repetition_rate', meas_specs,
                          msg_add=msg0 + m)
    else:
        # All lasers are CW, check that there is no pulsed laser field
        msg = ('According to /setup/excitation_cw all lasera are CW.\n'
               'However, `%s` has the field `laser_repetition_rate`.')
        _assert_valid('laser_repetition_rate' not in setup, msg % 'setup')
        _assert_valid('laser_repetition_rate' not in meas_specs,
                      msg % 'measurement_specs')

    msg_cw = """
    According to /setup/excitation_alternated this measurement uses
    laser alternation and CW lasers. However, there is no alex_period
    field in measurement_specs. alex_period is mandatory in measurements
    using alternation and CW lasers."""
    if all(setup.excitation_cw[:]):
        if any(setup.excitation_alternated[:]):
            _assert_has_field('alex_period', meas_specs,
                              msg_add=dedent(msg_cw))
        else:
            # No alternated laser, check that no ALEX field is present
            msg0 = ('According to /setup/excitation_alternated '
                    'no laser is alternated.\n')
            msg1 = 'However, measurement_specs has the ALEX field `%s`.'
            fields = ['alex_period', 'alex_offset']
            fields += ['alex_excitation_period%d' % i for i in (1, 2, 3)]
            for field in fields:
                _assert_valid(field not in meas_specs, msg0 + msg1 % field)

    if 'nanotimes' in ph_data and 'lifetime' in setup:
        _assert_valid(setup.lifetime.read(),
                      'Lifetime is False but nanotimes are present.')

    if 'lifetime' in setup and setup.lifetime.read():
        msg = """\
        According to /setup/lifetime (=True) this file should be a
        TCSPC measurement. However /setup/excitation_cw says that all
        the laser sources are CW instead of pulsed. At least one source
        needs to be pulsed."""
        _assert_valid(not all(setup.excitation_cw.read()), msg=dedent(msg))
        _assert_has_field('nanotimes', ph_data, **kwargs)

        if 'nanotimes_specs' in ph_data:
            _assert_has_field('nanotimes_specs', ph_data, **kwargs)
            tcspc_specs_group = ph_data.nanotimes_specs
        else:
            _assert_has_field('detectors', setup, **kwargs)
            tcspc_specs_group = setup.detectors
        for name in ('tcspc_unit', 'tcspc_num_bins'):
            _assert_has_field(name, tcspc_specs_group, **kwargs)

    # us-ALEX fields
    if meas_type in ('smFRET-usALEX', 'smFRET-usALEX-3c'):
        msg = 'All lasers need to be CW in %s measurements.'
        _assert_valid(all(setup.excitation_cw[:]), msg=msg % meas_type)
        msg = 'All lasers need to be alternated in %s measurements.'
        _assert_valid(all(setup.excitation_alternated[:]),
                      msg=msg % meas_type)

    # ns-ALEX / PIE fields
    if meas_type == 'smFRET-nsALEX':
        msg = 'All lasers need to be pulsed in smFRET-nsALEX measurements.'
        _assert_valid(all(~setup.excitation_cw[:]), msg=msg)
        _assert_has_field('lifetime', setup, **kwargs)
        _assert_valid(setup.lifetime.read(),
                      msg='smFRET-nsALEX requires lifetime = True.')


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
