#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of helper functions for identifying and filling out fields in the data
dictionary produced by :func:`phconvert.loader.loadfile_bh` and 
:func:`phconvert.loader.loadfile_ptu` functions.
"""

import re
import numpy as np

_spec_regex = re.compile(r'spectral_ch([1-9]\d*)')
_pol_regex = re.compile(r'polarization_ch([1-9]\d*)')
_split_regex = re.compile(r'split_ch([1-9]\d*)')
_phdata_regex = re.compile(r'photon_data([1-9]\d*)?')
_alex_regex = re.compile(r'alex_excitation_period([1-9]\d*)')


def _get_num_channel(specs, regex, name):
    """
    Determine the number of channels in detectors_specs dictionary for a given
    detections type.

    Parameters
    ----------
    specs : dict
        The dictionary of specs with enumerated groups (i.e. detectors_specs or
        measurement_specs).
    regex : re.Pattern
        Compiled regex for recognizing particular field, group 1 must
        output the number of the group.
    name : str
        name to call group in error message.

    Raises
    ------
    ValueError
        When given channel is not measured sequentialy, e.g. spectral_ch2 does
        not exist when spectral_ch3 does.

    Returns
    -------
    n_ch: int
        Number of channesl withing give detection type.

    """
    channels = sorted(int(regex.fullmatch(key).group(1)) for key in 
                      specs.keys() if regex.fullmatch(key))
    if not channels:
        return 1
    channels = np.array(channels)
    if np.any(channels != np.arange(1, channels[-1]+1, 1)):
        raise ValueError(f"One or more {name} channels skipped")
    n_ch = channels[-1]
    return n_ch
    

def get_num_spectral(detectors_specs):
    """
    Determine number of spectral detection types in a detectors_specs dictionary.
    Used to fill out field of ``/setup/num_spectral_ch`` automatically

    Parameters
    ----------
    detectors_specs : dict
        Dictionary to be converted into 
        ``/photon_data/measurement_specs/detectors_specs``.

    Returns
    -------
    int
        Integer value that should be placed in ``/setup/num_spectral_ch``
        assuming no further changes to the input.

    """
    n_spec = _get_num_channel(detectors_specs, _spec_regex, 'spectral')
    return n_spec


def get_num_polarization(detectors_specs):
    """
    Determine number of spectral detection types in a detectors_specs dictionary.
    Used to fill out field of ``/setup/num_polarization_ch`` automatically

    Parameters
    ----------
    detectors_specs : dict
        Dictionary to be converted into 
        ``/photon_data/measurement_specs/detectors_specs``.

    Returns
    -------
    int
        Integer value that should be placed in ``/setup/num_polarization_ch``
        assuming no further changes to the input.

    """
    n_pol = _get_num_channel(detectors_specs, _split_regex, 'split')
    return n_pol


def get_num_split(detectors_specs):
    """
    Determine number of spectral detection types in a detectors_specs dictionary.
    Used to fill out field of ``/setup/num_split_ch`` automatically

    Parameters
    ----------
    detectors_specs : dict
        Dictionary to be converted into 
        ``/photon_data/measurement_specs/detectors_specs``.

    Returns
    -------
    int
        Integer value that should be placed in ``/setup/num_split_ch``
        assuming no further changes to the input.

    """
    n_split = _get_num_channel(detectors_specs, _split_regex, 'split')
    return n_split

def fill_alex_periods(data, *args):
    """
    Add excitation periods to ``/photon_data/measurement_specs/`` sub-dictionary
    of a dictionary to be converted into a photon-HDF5 file

    Parameters
    ----------
    data : dict
        Data dictionary.
    *args : np.ndarray
        One array per excitation period, each array must be even number of
        elements specifiying the star and stop time(s) for the given period.

    """
    for key, val in data.items():
        if not _phdata_regex.fullmatch(key):
            continue
        for i, alternation in enumerate(args):
            val['measurement_specs'][f'alex_excitation_period{i+1}'] = alternation

def _get_num_lasers(data):
    lasers = [_get_num_channel(val['measurement_specs'], _alex_regex, 
                               'alex_excitation_period') for key, val in 
              data.items() if _phdata_regex.fullmatch(key)]
    if all(l == lasers[0] for l in lasers[1:]):
        return lasers[0]
    else:
        return sum(lasers)
    

def fill_setup(data):
    """
    Fill setup dictioanry based on detectors_specs dictionary

    Parameters
    ----------
    setup : dict
        Dicionary for the ``/setup/`` group.
    detectors_specs : dict
        Dictionary for the ``/photon_data/measurement_specs/detectors_specs``.

    """
    setup = data['setup']
    spec = tuple(data[key]['measurement_specs']['detectors_specs'] for key in 
                 data.keys() if _phdata_regex.fullmatch(key))
    setup['num_spectral_ch'] = sum(get_num_spectral(ds) for ds in spec)
    setup['num_polarization_ch'] = sum(get_num_polarization(ds) for ds in spec)
    setup['num_split_ch'] = sum(get_num_split(ds) for ds in spec)
    nlaser = _get_num_lasers(data)
    nrep = setup.get('laser_repetition_rates', None)
    nrep = 0 if nrep is None else nrep.size
    if nlaser != nrep and nlaser != 0:
        if nrep == 1:
            rep_rates = np.repeat(setup['laser_repetition_rates'], nlaser)
            setup['laser_repetition_rates'] = rep_rates
        else:
            print("Warning: laser_repetition_rates inconsistent size "
                  "compared to alex_excitation_period, must manually "
                  "re-assign to correct number, got {nlaser} from "
                  "alex_excitation_period and {nrep} from "
                  "laser_repetition_rates")
    elif nlaser == 0:
        print("Warning: alex_excitation_periods defined, but no "
              "laser_repetition_rates specified, must manually add")
            


def fill_measurement_type(data, measurement_type):
    """
    Iterate over all photon_dataX groups and set the field
    ``/photon_dataX/measurement_spces/measurement_type`` to 

    Parameters
    ----------
    data : dict
        dictionary that will passed to :func:`phconvert.hdf5.save_photon_hdf5`.
    measurement_type : str
        The specified measurement type. Use "generic" if unsure.

    """
    for key, val in data.items():
        if not _phdata_regex.fullmatch(key):
            continue
        val['measurement_specs']['measurement_type'] = measurement_type



        

def report_nones(data, root=''):
    """
    Identify the fields that must be either removed or specified from a
    dictionary returned by the ``loader.loadfile_`` function

    Parameters
    ----------
    data : dict
        Dictionary or subdictionary of dictionary returned from a 
        ``loader.loadfile_`` function.
    root : str, optional
        Name to print beforoe each key with a None value. The default is ''.

    """
    for key, sub in data.items():
        if isinstance(sub, dict):
            report_nones(sub, root=f'{root}/{key}')
        if sub is None:
            print(f'{root}/{key}')


def pop_nones(data):
    """
    Remove keys with None values in a dictionary from ``loader.loadfile_``
    function. Function should be called only after all other applicable None
    values have been set appropriately.
    
    Use :func:`report_nones` to identify all None values in data dictionary.
    Call this and fill out all relevant fields before calling ``pop_nones``

    Parameters
    ----------
    data : dict
        Dictionary from ``loader.loadfile_`` to be converted to photon-HDF5 

    """
    keys = tuple(data.keys())
    for key in keys:
        if data[key] is None:
            data.pop(key)
        elif isinstance(data[key], dict):
            pop_nones(data[key])
