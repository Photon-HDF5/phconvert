from phconvert import bhreader
import numpy as np
import pandas as pd

import pytest

# Run all tests  with $ nosetests

# class TestBhReader(unittest.TestCase):
# 
#     def test_import_SPC_150_nanotime(self):
#         input_file = os.path.join(os.path.dirname(__file__),
#             'test_files/test_noise.spc')
#         check_file = os.path.join(os.path.dirname(__file__),
#             'test_files/test_noise.asc')
# 
#         data = bhreader.load_spc(input_file, 'SPC-150')
#         check = pd.read_csv(check_file, delimiter=' ', dtype='int64',
#             usecols=[0, 1], header=None).values.T  # Way faster than numpy
# 
#         # Same number of photons in both files
#         self.assertTrue(data[2].size == check[1].size)
#         # Same length for macrotime, microtime and detector
#         self.assertTrue(data[0].size == data[1].size == data[2].size)
#         self.assertTrue(np.equal(data[2], check[1]).all())  # Equal macrotimes
#         self.assertTrue(np.equal(data[0], check[0]).all())  # Equal microtimes
#         self.assertAlmostEqual(data[3]*1e9, 9.5, 1)         # Right timestamp

def test_import_SPC_150_nanotime():
    input_file = '../PhotonHDF5_testdata/test_noise.spc'
    check_file = '../PhotonHDF5_testdata/test_noise.asc'

    data = bhreader.load_spc(input_file)
    timestamps = data['photon_data']['timestamps']
    nanotimes = data['photon_data']['nanotimes']
    detectors = data['photon_data']['detectors']
    macroclock = data['photon_data']['timestamps_unit']
    assert timestamps.size == nanotimes.size == detectors.size
    timecheck, nanocheck = pd.read_csv(check_file, delimiter=' ', dtype='int64', usecols=[0,1], header=None).values.T
    assert nanotimes.size == nanocheck.size
    assert np.all(nanotimes == nanocheck)
    assert np.all(timestamps == timecheck)
    assert round(macroclock*1e9, 1) == 9.5

# def test_import_SPC_600_nanotime():
#     input_file = cd .