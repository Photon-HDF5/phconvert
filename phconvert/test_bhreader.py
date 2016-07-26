import unittest
import bhreader
import numpy as np
import pandas as pd

# TODO: use a smaller file for test

class TestBhReader(unittest.TestCase):

    def test_import_SPC_150_nanotime(self):
        input_file = 'test_files/7od.spc'
        check_file = 'test_files/7od.asc'

        data = bhreader.load_spc(input_file, 'SPC-150')
        check = pd.read_table(check_file, delimiter=' ', dtype='int64',
            usecols=[0, 1], header=None).as_matrix().T  # Way faster than numpy

        self.assertTrue(data[2].size == check[1].size)  # Same number of photons
        self.assertTrue(data[0].size == data[1].size == data[2].size)
        self.assertTrue(np.equal(data[2], check[1]).all())
        self.assertTrue(np.equal(data[0], check[0]).all())

if __name__ == '__main__':
    unittest.main()