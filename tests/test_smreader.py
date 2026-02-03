#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the (old) sm format
"""
from phconvert import smreader

import pytest

DATADIR = '../PhotonHDF5_testdata/'


def test_smreader():
    filename = DATADIR + '0023uLRpitc_NTP_20dT_0.5GndCl.sm'
    timestamps, detectors = smreader.load_sm(filename)
    assert timestamps.size == detectors.size