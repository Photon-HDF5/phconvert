#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load and decode files from Becker & Hickl
hardware.

The high-level function in this module are:

- :func:`load_spc` which loads and decoded the photon data from SPC files.
- :func:`load_set` which returns a dictionary of metadata from SET files.


Becker & Hickl SPC Format
-------------------------

The structure of the SPC format is here described.
Each record is a 6-bytes element in little endian (<) format.

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

The first 6 bytes of a SPC file are an header containing the timestamps_unit
(in 0.1ns units) in the two central bytes (i.e. bytes 2 and 3).
"""

from __future__ import print_function, division
import numpy as np


def load_spc(fname):
    """Load data from Becker & Hickl SPC files.

    Returns:
        3 numpy arrays (timestamps, detector, nanotime) and a float
        (timestamps_unit).
    """

    f = open(fname, 'rb')
    # We first decode the first 6 bytes which is a header...
    header = np.fromfile(f, dtype='u2', count=3)
    timestamps_unit = header[1] * 0.1e-9
    num_routing_bits = np.bitwise_and(header[0], 0x000F)  # unused

    # ...and then the remaining records containing the photon data
    spc_dtype = np.dtype([('field0', '<u2'), ('b', '<u1'), ('c', '<u1'),
                          ('a', '<u2')])
    data = np.fromfile(f, dtype=spc_dtype)

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

    return timestamps, detector, nanotime, timestamps_unit

def load_set(fname_set):
    """Return a dict with data from the Becker & Hickl .SET file.
    """
    identification = bh_set_identification(fname_set)
    sys_params = bh_set_sys_params(fname_set)
    return dict(identification=identification, sys_params=sys_params)


def bh_set_identification(fname_set):
    """Return a dict containing the IDENTIFICATION section of .SET files.

    The both keys and values are native strings (binary strings on py2
    and unicode strings on py3).
    """
    with open(fname_set, 'rb') as f:
        line = f.readline()
        assert line.strip().endswith(b'IDENTIFICATION')
        identification = {}
        # .decode() returns a unicode string and str() a native string
        # on both py2 and py3
        line = str(f.readline().strip().decode('utf8'))
        while not line.startswith('*END'):
            item = [s.strip() for s in line.split(':')]
            if len(item) > 1:
                # found ':'  ->  retrive key and value
                key = item[0]
                value = ':'.join(item[1:])
            else:
                # no ':' found ->  it's a new line continuing the previous key
                value = ' '.join([identification[key], item[0]])
            identification[key] = value
            line = str(f.readline().strip().decode('utf8'))
    return identification

def bh_set_sys_params(fname_set):
    """Return a dict containing the SYS_PARAMS section of .SET files.

    The keys are native strings (traditional strings on py2
    and unicode strings on py3) while values are numerical type or
    byte strings.
    """
    with open(fname_set, 'rb') as f:
        ## Make a dictionary of system parameters
        start = False
        sys_params  = {}
        for line in f.readlines():
            # line can contain byte garbage, so don't convert to str
            line = line.strip()
            if line == b'SYS_PARA_BEGIN:':
                start = True
                continue
            if line == b'SYS_PARA_END:':
                break
            if start and line.startswith(b'#'):
                # Still there can be unknown fields, keep it binary
                fields = line[5:-1].split(b',')

                if fields[1] == b'B':
                    value = bool(fields[2])
                elif fields[1] in [b'I', b'U', b'L']:
                    value = int(fields[2])
                elif fields[1] == b'F':
                    value = float(fields[2])
                elif fields[1] == b'S':
                    value = fields[2]  # binary string
                else:
                    value = b','.join(fields[1:])  # unknown, recomposing it

                sys_params[str(fields[0].decode())] = value
    return sys_params

def bh_decode(s):
    """Replace code strings from .SET files with human readble label strings.
    """
    s = s.replace('SP_', '')
    s = s.replace('_ZC', ' ZC Thresh.')
    s = s.replace('_LL', ' Limit Low')
    s = s.replace('_LH', ' Limit High')
    s = s.replace('_FD', ' Freq. Div.')
    s = s.replace('_OF', ' Offset')
    s = s.replace('_HF', ' Holdoff')
    s = s.replace('TAC_G', 'TAC Gain')
    s = s.replace('TAC_R', 'TAC Range')
    s = s.replace('_TC', ' Time/Chan')
    s = s.replace('_TD', ' Time/Div')
    s = s.replace('_FQ', ' Threshold')
    return s

def bh_print_sys_params(sys_params):
    """Print a summary of the Becker & Hickl system parameters (.SET file).
    """
    for k, v in sys_params.iteritems():
        if 'TAC' in k: print('%s\t %f' % (bh_decode(k), v))
    print()
    for k, v in sys_params.iteritems():
        if 'CFD' in k: print('%s\t %f' % (bh_decode(k), v))
    print()
    for k, v in sys_params.iteritems():
        if 'SYN' in k: print('%s\t %f' % (bh_decode(k), v))
