#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:36:02 2024

@author: paul
"""
# import os
import numpy as np


import pytest
import phconvert as phc


tcspc_unit = 5e-8/4096

timestamps = np.cumsum(np.random.poisson(100, 1000).astype(np.int64))
detectors = np.random.randint(0,2,timestamps.shape, np.uint8)
nanotimes = np.random.exponential(50, timestamps.shape)
nanotimes[nanotimes>4096] = 4095
nanotimes = nanotimes.astype(np.int16)
description="Test file save"

timestamps_unit = 5e-8
timestamps_specs = dict(timestamps_unit=timestamps_unit)


def make_setup(excitations, spectral, polarization, split, lifetime):
    setup = dict(num_spectral_ch=spectral, num_polarization_ch=polarization, 
                 num_split_ch=split, num_spots=1, 
                 num_pixels=spectral*polarization*split,
                 lifetime=lifetime, modulated_excitation=True,
                 excitation_cw = np.array([not lifetime for _ in range(excitations)]),
                 excitation_alternated=np.array([not lifetime for _ in range(excitations)]),
                 detectors=dict(),
                 )
    return setup

@pytest.mark.filterwarnings("ignore")
def test_saveminimal():
    data = dict(_filename='test.spc', description=description, 
                photon_data=dict(timestamps=timestamps, 
                                 timestamps_specs=timestamps_specs))
    phc.hdf5.save_photon_hdf5(data, require_setup=False)

@pytest.mark.filterwarnings("ignore")
def test_saveCW():
    measurement_specs = dict(measurement_type='smFRET-usALEX',
                             alex_period=4000,
                             alex_offset=100,
                             alex_excitation_period1=np.array([10,1990]),
                             alex_excitation_period2=np.array([2010, 3990]),
                             detectors_specs=dict(spectral_ch1=np.array([0,], dtype=np.uint8),
                                                  spectral_ch2=np.array([1,], dtype=np.uint8)))
    setup = make_setup(2,2,1,1,False)
    setup['detectors']['id'] = np.array([0,1], dtype=np.uint8)
    data = dict(_filename='test.ptu', description=description,
                photon_data=dict(timestamps=timestamps, detectors=detectors,
                                 measurement_specs=measurement_specs,
                                 timestamps_specs=timestamps_specs),
                setup=setup)
    phc.hdf5.save_photon_hdf5(data)