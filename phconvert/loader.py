#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load each supported data format.

The loader functions are used to load data from a specific format and
return a dictionary with keys names corresponding to the Photon-HDF5
data fields.

These dictionaries can be used by the :func:`phconvert.hdf5.photon_hdf5`
function to save the data in Photon-HDF5 format.

"""

from __future__ import (print_function, absolute_import, division,
                        unicode_literals)
del print_function, absolute_import, division
import os
import time

from . import smreader
from . import bhreader
from . import pqreader


def usalex_sm(
        filename, donor=0, acceptor=1, alex_period=4000,
        alex_period_donor=(2850, 580), alex_period_acceptor=(930, 2580),
        excitation_wavelengths=(532e-9, 635e-9)):
    """Load a .sm us-ALEX file and returns a dictionary.

    This dictionary can be passed to the :func:`phconvert.hdf5.photon_hdf5`
    function to save the data in Photon-HDF5 format.
    """
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, labels = smreader.load_sm(filename,
                                                     return_labels=True)
    print(" [DONE]\n")

    dx = dict(filename=filename,
              alex=True,
              lifetime=False,
              timestamps_unit=12.5e-9,
              num_spots=1,
              num_detectors=2,
              num_polariz_ch=1,
              num_spectral_ch=2,
              alex_period=alex_period,
              alex_period_donor=alex_period_donor,
              alex_period_acceptor=alex_period_acceptor,

              timestamps=timestamps,
              detectors=detectors,
              donor=donor,
              acceptor=acceptor,

              excitation_wavelengths=excitation_wavelengths,
              )
    return dx

def nsalex_bh(
        filename_spc, donor=4, acceptor=6, laser_pulse_rate=40e6,
        tcspc_range=60e-9, timestamps_unit=60e-9,
        alex_period_donor=(10, 1500), alex_period_acceptor=(2000, 3500),
        excitation_wavelengths=(532e-9, 635e-9), allow_missing_set=False):
    """Load a .spc and (optionally) .set files for ns-ALEX and return 2 dict.

    The first dictionary can be passed to the
    :func:`phconvert.hdf5.photon_hdf5` function to save the data in
    Photon-HDF5 format.

    Returns:
        Two dictionaries: the first contains the main photon data (timestamps,
        detectors, nanotime, ...); the second contains the raw data from the
        .set file (it can be saved in a user group in Photon-HDF5).
    """
    # Load .SPC file
    assert os.path.isfile(filename_spc), \
           "File '%s' not found." % filename_spc
    print(" - Loading '%s' ... " % filename_spc)
    timestamps, detectors, nanotimes = bhreader.load_spc(filename_spc)
    print(" [DONE]\n")

    # Load .SET file
    filename_set = filename_spc[:-3] + 'set'
    if os.path.isfile(filename_set):
        metadata = bhreader.load_set(filename_set)
    elif allow_missing_set:
        metadata = {}
    else:
        raise IOError("File '%s' not found." % filename_set)

    # Estract the creation time from the .SET file metadata as it will be
    # more reliable than the creation time from the file system
    provenance = {}
    if 'identification' in metadata:
        identification = metadata['identification']
        date_str = identification['Date'].decode()
        time_str = identification['Time'].decode()
        creation_time = date_str + ' ' + time_str
        provenance = {'creation_time': creation_time}

    tcspc_num_bins = 4096
    if metadata is not None:
        print('Ignoring arguments `timestamps_units` and `tcspc_range`.')
        print('These values were retrived from .SET file.')
        sys_params = metadata['sys_params']
        tcspc_unit = sys_params['SP_TAC_TC']
        #tcspc_range = sys_params['SP_TAC_R']
        tcspc_range = tcspc_num_bins*tcspc_unit
        timestamps_unit = tcspc_range
    else:
        print('Using timestamps_units and tcspc_range from function arguments.')
        tcspc_unit = tcspc_range/tcspc_num_bins

    acquisition_time = (timestamps.max() - timestamps.min())*timestamps_unit

    dict_bh = dict(filename=filename_spc,
              alex=True,
              lifetime=True,
              timestamps_unit=timestamps_unit,
              acquisition_time=acquisition_time,
              provenance=provenance,

              num_spots=1,
              num_detectors=2,
              num_polariz_ch=1,
              num_spectral_ch=2,
              laser_pulse_rate=laser_pulse_rate,
              alex_period_donor=alex_period_donor,
              alex_period_acceptor=alex_period_acceptor,

              timestamps=timestamps,
              detectors=detectors,
              nanotimes=nanotimes,
              donor=donor,
              acceptor=acceptor,
              tcspc_num_bins=tcspc_num_bins,
              tcspc_range=tcspc_range,
              tcspc_unit=tcspc_unit,

              excitation_wavelengths=excitation_wavelengths,
              )
    return dict_bh, metadata

def nsalex_ht3(filename, donor=0, acceptor=1, laser_pulse_rate=None):
    """Load a .ht3 file containing ns-ALEX data and return a dict.

    This dictionary can be passed to the :func:`phconvert.hdf5.photon_hdf5`
    function to save the data in Photon-HDF5 format.
    """
    timestamps, detectors, nanotimes, metadata = pqreader.load_ht3(filename)
    timestamps_unit = metadata.pop('timestamps_unit')
    tcspc_unit = metadata.pop('nanotimes_unit')
    tcspc_num_bins = 4096
    tcspc_range = tcspc_num_bins*tcspc_unit
    acquisition_time = metadata['header']['Tacq'][0]*1e-3

    # Estract the creation time from the HT3 file header as it will be
    # more reliable than the creation time from the file system
    ctime_t = time.strptime(metadata['header']['FileTime'][0].decode(),
                            "%d/%m/%y %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    provenance = {'creation_time': creation_time}

    dict_pq = dict(
        filename=filename,
        alex=True,
        lifetime=True,
        timestamps_unit=timestamps_unit,
        acquisition_time=acquisition_time,
        provenance=provenance,

        num_spots=1,
        num_detectors=2,
        num_polariz_ch=1,
        num_spectral_ch=2,
        laser_pulse_rate=1/timestamps_unit,
        #alex_period_donor=alex_period_donor,
        #alex_period_acceptor=alex_period_acceptor,

        timestamps=timestamps,
        detectors=detectors,
        nanotimes=nanotimes,
        donor=donor,
        acceptor=acceptor,

        tcspc_num_bins=tcspc_num_bins,
        tcspc_range=tcspc_range,
        tcspc_unit=tcspc_unit,

        #excitation_wavelengths=excitation_wavelengths,
    )
    return dict_pq, metadata

def nsalex_pt3(filename, donor=1, acceptor=2, laser_pulse_rate=None):
    """Load a .pt3 file containing ns-ALEX data and return a dict.

    This dictionary can be passed to the :func:`phconvert.hdf5.photon_hdf5`
    function to save the data in Photon-HDF5 format.
    """
    timestamps, detectors, nanotimes, metadata = pqreader.load_pt3(filename)

    tcspc_unit = metadata['nanotimes_unit']
    tcspc_num_bins = max(nanotimes)  # or int(timestamps_unit/nanotimes_unit ?
    tcspc_range = tcspc_num_bins*tcspc_unit

    dict_pq = dict(
        filename=filename,
        alex=True,
        lifetime=True,
        timestamps_unit=metadata['timestamps_unit'],

        num_spots=1,
        num_detectors=2,
        num_polariz_ch=1,
        num_spectral_ch=2,
        #laser_pulse_rate=laser_pulse_rate,
        #alex_period_donor=alex_period_donor,
        #alex_period_acceptor=alex_period_acceptor,

        timestamps=timestamps,
        detectors=detectors,
        nanotimes=nanotimes,
        donor=donor,
        acceptor=acceptor,

        tcspc_num_bins=tcspc_num_bins,
        tcspc_range=tcspc_range,
        tcspc_unit=tcspc_unit,

        #excitation_wavelengths=excitation_wavelengths,
    )
    return dict_pq