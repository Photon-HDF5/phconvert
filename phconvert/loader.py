#
# Copyright (C) 2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load each supported data format.
The loader functions are used to load data from a specific format and
return a dictionary with keys names corresponding to the Photon-HDF5
data fields.
"""

from __future__ import print_function, absolute_import, division
del print_function, absolute_import, division

from .smreader import load_sm as _load_sm
from .bhreader import load_spc as _load_spc


def usalex_sm(
        filename, donor=0, acceptor=1, alex_period=4000,
        alex_period_donor=(2850, 580), alex_period_acceptor=(930, 2580),
        excitation_wavelengths=(532e-9, 635e-9)):
    """Load a .sm us-ALEX file and returns a dictionary.
    """
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, labels = _load_sm(filename, return_labels=True)
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

def nsalex_spc(
        filename, donor=4, acceptor=6, laser_pulse_rate=40e6,
        tcspc_range=50e-9, timestamps_unit=50e-9,
        alex_period_donor=(10, 1500), alex_period_acceptor=(2000, 3500),
        excitation_wavelengths=(532e-9, 635e-9)):
    """Load a .spc ns-ALEX file and returns a dictionary.
    """
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, nanotimes = _load_spc(filename)
    print(" [DONE]\n")

    tcspc_num_bins = 4096

    dx = dict(filename=filename,
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
              tcspc_unit=tcspc_range/tcspc_num_bins,

              excitation_wavelengths=excitation_wavelengths,
              )
    return dx
