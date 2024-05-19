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


SPC-600/630
~~~~~~~~~~~

SPC-600/630 files have a record of 48-bit (6 bytes)
in little endian (<) format.
The first 6 bytes of the file are an header containing
the `timestamps_unit` (in 0.1ns units) in the two central bytes
(i.e. bytes 2 and 3).
In the following drawing each char represents 2 bits::

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

SPC-134/144/154/830
~~~~~~~~~~~~~~~~~~~

SPC-134/144/154/830 files have a record of 32-bits (4 bytes) in
little endian (<) format.
The first 4 bytes of the file are an header containing the
`timestamps_unit` (in 0.1ns units) in first two bytes.
In the following drawing each char represents 2 bits::


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

import numpy as np


def load_spc(fname, spc_model='SPC-630'):
    """Load data from Becker & Hickl SPC files.

    Arguments:
        spc_model (string): name of the board model. Valid values are
            'SPC-630', 'SPC-134', 'SPC-144', 'SPC-154' and 'SPC-830'.

    Returns:
        3 numpy arrays (timestamps, detector, nanotime) and a float
        (timestamps_unit).
    """

    with open(fname, 'rb') as f:

        if ('630' in spc_model) or ('600' in spc_model):
    
            # We first decode the first 6 bytes which is a header...
            header = np.fromfile(f, dtype='u2', count=3)
            timestamps_unit = header[1] * 0.1e-9
            num_routing_bits = np.bitwise_and(header[0], 0x000F)  # unused
    
            # ...and then the remaining records containing the photon data
            spc_dtype = np.dtype([('field0', '<u2'), ('b', '<u1'), ('c', '<u1'),
                                  ('a', '<u2')])
            data = np.fromfile(f, dtype=spc_dtype)
    
            nanotime = 4095 - np.bitwise_and(data['field0'], 0x0FFF)
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
            # We first decode the first 4 bytes which is a header...
            header = np.fromfile(f, dtype='u4', count=1)[0]
            timestamps_unit = np.bitwise_and(header, 0x00FFFFFF) * 0.1e-9
            num_routing_bits = np.bitwise_and(np.right_shift(header, 32), 0x78)  # unused
    
            # ...and then the remaining records containing the photon data
            spc_dtype = np.dtype([('field0', '<u2'), ('field1', '<u2')])
            data = np.fromfile(f, dtype=spc_dtype)
    
            nanotime = 4095 - np.bitwise_and(data['field1'], 0x0FFF)
            detector = np.bitwise_and(np.right_shift(data['field0'], 12), 0x0F)
    
            # Build the macrotime
            timestamps = np.bitwise_and(data['field0'], 0x0FFF).astype(dtype='int64')
    
            # Extract the various status bits
            mark = np.bitwise_and(np.right_shift(data['field1'], 12), 0x01)
            gap = np.bitwise_and(np.right_shift(data['field1'], 13), 0x01)
            overflow = np.bitwise_and(np.right_shift(data['field1'], 14), 0x01).\
                astype(dtype='int64')
            invalid = np.bitwise_and(np.right_shift(data['field1'], 15), 0x01)
    
            # Invalid bytes: number of overflows from the last detected photon
            for i_ovf in np.nonzero(overflow)[0].tolist():
                if invalid[i_ovf]:
                    overflow[i_ovf] = np.left_shift(np.bitwise_and(
                        data['field1'][i_ovf], 0x0FFF), 16)\
                        + data['field0'][i_ovf]
    
            # Each overflow occurs every 2^12 macrotimes
            overflow = np.left_shift(np.cumsum(overflow), 12)
    
            # Add the overflow bits
            timestamps += overflow
    
            # Delete invalid entries
            nanotime = np.delete(nanotime, invalid.nonzero())
            timestamps = np.delete(timestamps, invalid.nonzero())
            detector = np.delete(detector, invalid.nonzero())

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
    """Replace code strings from .SET files with human readable label strings.
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
    if 'sys_params' in sys_params:
        # Passed output of load_set().  Extract sys_params and retry
        bh_print_sys_params(sys_params['sys_params'])
    else:
        for k, v in sys_params.items():
            if 'TAC' in k: print('%s\t %g' % (bh_decode(k), v))
        print()
        for k, v in sys_params.items():
            if 'CFD' in k: print('%s\t %g' % (bh_decode(k), v))
        print()
        for k, v in sys_params.items():
            if 'SYN' in k: print('%s\t %g' % (bh_decode(k), v))
