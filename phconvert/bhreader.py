#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
SPC Format (Beker & Hickl)
--------------------------

SPC-600/630:
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

SPC-134/144/154/830:

    bit:                     32                0
                             XXXX XXXX XXXX XXXX
                             '''-----' '''-----'
    field names:             a    b    c    d

                             XXXX XXXX XXXX XXXX
                             '-------' '-------'
    numpy dtype:              field1    field0

    macrotime = [ d ]       (12 bit)
    detector  = [ c ]       (4 bit)
    nanotime  = [ b ]       (12 bit)
    aux       = [ a ]       (4 bit)

    aux = [invalid, overflow, gap, mark]

    If overflow == 1 and invalid == 1 --> number of overflows = [ b ][ c ][ d ]
"""

# TODO: automatic board model identification (in a new function?)

from __future__ import print_function, division
import numpy as np


def load_spc(fname, spc_model='SPC-630'):
    """Load data from Becker & Hickl SPC files.

    spc_model: name of the board model (ex. 'SPC-630')

    Returns:
        3 numpy arrays: timestamps, detector, nanotime
    """

    if ('630' in spc_model) or ('600' in spc_model):
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

    elif ('SPC-1' in spc_model) or ('SPC-830' in spc_model):
        spc_dtype = np.dtype([('field0', '<u2'),('field1', '<u2')])
        data = np.fromfile(fname, dtype=spc_dtype)

        nanotime =  4095 - np.bitwise_and(data['field1'], 0x0FFF)
        detector = np.bitwise_and(np.right_shift(data['field0'], 12), 0x0F)

        # Build the macrotime
        timestamps = np.bitwise_and(data['field0'], 0x0FFF).astype(dtype='int64')

        # Extract the various status bits
        mark = np.bitwise_and(np.right_shift(data['field1'], 12), 0x01)
        gap = np.bitwise_and(np.right_shift(data['field1'], 13), 0x01)
        overflow = np.bitwise_and(np.right_shift(data['field1'], 14), 0x01).astype(dtype='int64')
        invalid = np.bitwise_and(np.right_shift(data['field1'], 15), 0x01)

        # Invalid bytes: number of overflows from the last detected photon
        for i_ovf in np.nonzero(overflow)[0].tolist():
            if invalid[i_ovf]:
                overflow[i_ovf] = np.left_shift(np.bitwise_and(data['field1'][i_ovf], 0x0FFF), 16)\
                    + data['field0'][i_ovf]

        overflow = np.left_shift(np.cumsum(overflow), 12)  # Each overflow occurs every 2^12 macrotimes

        # Add the overflow bits
        timestamps += overflow

        # Delete invalid entries
        nanotime = np.delete(nanotime, invalid.nonzero())
        timestamps = np.delete(timestamps, invalid.nonzero())
        detector = np.delete(detector, invalid.nonzero())

    return timestamps, detector, nanotime


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
            if len(item) == 1:
                # no ':'  ->  it's a new line continuing the previous key
                value = ' '.join([identification[key], item[0]])
            else:
                # found ':'  ->  retrive key and value
                key = item[0]
                value = ':'.join(item[1:])
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
    """Decode strings from Becker & Hickl system parameters (.SET file)."""
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
    """Print a summary of the Becker & Hickl system parameters (.SET file)."""
    for k, v in sys_params.iteritems():
        if 'TAC' in k: print('%s\t %f' % (bh_decode(k), v))
    print()
    for k, v in sys_params.iteritems():
        if 'CFD' in k: print('%s\t %f' % (bh_decode(k), v))
    print()
    for k, v in sys_params.iteritems():
        if 'SYN' in k: print('%s\t %f' % (bh_decode(k), v))

