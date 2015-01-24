#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
SPC Format (Beker & Hickl)
--------------------------

48-bit element in little endian (<) format

Drawing (note: each char represents 2 bits)::

    bit: 64        48                          0
         0000 0000 XXXX XXXX XXXX XXXX XXXX XXXX
                   '-------' '--' '--'   '-----'
    field names:       a      c    b        d

         0000 0000 XXXX XXXX XXXX XXXX XXXX XXXX
                   '-------' '--' '--' '-------'
    numpy dtype:       a      c    b    field0

    macrotime = [ b  ] [     a     ]  (24 bit)
    detector  = [ c  ]                (8 bit)
    nanotime  = [  d  ]               (12 bit)

    overflow bit: 13, bit_mask = 2^(13-1) = 4096
"""

import numpy as np


def load_spc(fname):
    """Load data from Becker&Hickl SPC files.

    Returns:
        3 numpy arrays: timestamps, detector, nanotime
    """
    spc_dtype = np.dtype([('field0', '<u2'), ('b', '<u1'), ('c', '<u1'),
                          ('a', '<u2')])
    data = np.fromfile(fname, dtype=spc_dtype)

    nanotime =  4095 - np.bitwise_and(data['field0'], 0x0FFF)
    detector = data['c']

    # Build the macrotime (timestamps) using in-place operation for efficiency
    timestamps = data['b'].astype('int64')
    np.left_shift(timestamps, 16, out=timestamps)
    timestamps += data['a']

    # extract the 13-th bit from data['field0']
    overflow = np.bitwise_and(np.right_shift(data['field0'], 13), 1)
    overflow = np.cumsum(overflow, dtype='int64')

    # Add the overflow bits
    timestamps += np.left_shift(overflow, 24)

    return timestamps, detector, nanotime
