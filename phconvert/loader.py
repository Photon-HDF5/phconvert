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
"""

from __future__ import print_function, absolute_import, division
del print_function, absolute_import, division
import os

from . import smreader
from . import bhreader


def usalex_sm(
        filename, donor=0, acceptor=1, alex_period=4000,
        alex_period_donor=(2850, 580), alex_period_acceptor=(930, 2580),
        excitation_wavelengths=(532e-9, 635e-9)):
    """Load a .sm us-ALEX file and returns a dictionary.
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
        excitation_wavelengths=(532e-9, 635e-9)):
    """Load a .spc and (optionally) .set files for ns-ALEX and return 2 dict.
    """
    # Load .SPC file
    assert os.path.isfile(filename_spc), "File '%s' not found." % filename_spc
    print(" - Loading '%s' ... " % filename_spc)
    timestamps, detectors, nanotimes = bhreader.load_spc(filename_spc)
    print(" [DONE]\n")

    # Load .SET file
    filename_set = filename_spc[:-3] + 'set'
    if os.path.isfile(filename_set):
        dict_set = bhreader.load_set(filename_set)
    elif allow_missing_set:
        dict_set = {}
    else:
        raise IOError("File '%s' not found." % filename_set)

    tcspc_num_bins = 4096
    if dict_set is not None:
        print('Ignoring arguments `timestamps_units` and `tcspc_range`.')
        print('These values were retrived from .SET file.')
        sys_params = dict_set['sys_params']
        tcspc_unit = sys_params['SP_TAC_TC']
        #tcspc_range = sys_params['SP_TAC_R']
        tcspc_range = tcspc_num_bins*tcspc_unit
        timestamps_unit = tcspc_range
    else:
        print('Using timestamps_units and tcspc_range from function arguments.')
        tcspc_unit = tcspc_range/tcspc_num_bins

    dict_spc = dict(filename=filename_spc,
              alex=True,
              lifetime=True,
              timestamps_unit=timestamps_unit,

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
    return dict_spc, dict_set

