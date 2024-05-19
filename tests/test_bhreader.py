import unittest
from phconvert import bhreader
import numpy as np
import pandas as pd
import os.path

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
    input_file = 'tests/test_files/test_noise.spc'
    check_file = 'tests/test_files/test_noise.asc'

    data = bhreader.load_spc(input_file, 'SPC-150')
    check = pd.read_csv(check_file, delimiter=' ', dtype='int64', usecols=[0,1], header=None).values.T
    
    assert data[2].size == check[1].size
    assert np.equal(data[2], check[1]).all()
    assert np.equal(data[0], check[0]).all()
    assert round(data[3]*1e9, 1) == 9.5

