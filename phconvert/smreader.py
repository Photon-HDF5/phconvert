#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
SM Format, written by a LV program in WeissLab us-ALEX setup
------------------------------------------------------------

A SM file is composed by two parts:

  - XXX bytes: an header (usually 166 bytes)
  - the remainig bytes: the data
  - 26 bytes of trailing cruft

The data is a list of 96-bit records that contain for each ph (count)
the detection time and the detector.

The first 64-bit of each record are the ph time, and the remainig 16-bit
are the detector.

::
              64-bit ph time        detector
            -------------------     -------
           |                   |   |       |
           XXXX XXXX - XXXX XXXX - XXXX XXXX
     bit:  0      32   0      32   0      32
           '-------'   '-------'   '-------'
            data[0]     data[1]     data[2]

The data is in unsigned big endian (>) format.

The proper way to read the data is to read the byte-stream and interpret
it as a record array in which each element is 12 bytes.
"""

import numpy as np

class Decoder:
    def __init__(self, buffer):
        self.buff = buffer
        self.cursor = 0

    def readscalar(self, size=4, basetype='u', endiness='>', inplace=False):
        dtype = endiness + basetype + str(size)
        buffer = self.buff[self.cursor:self.cursor + size]
        if not inplace:
            self.cursor += size
        return np.frombuffer(buffer=buffer, dtype=dtype)

    def readstring(self, size_max=256, inplace=False, **kwargs):
        orig_cursor = self.cursor
        size = int(self.readscalar(**kwargs))
        if size > size_max:
            print('Big string: size limited to %d.' % size_max)
            size = size_max

        string = self.buff[self.cursor:self.cursor + size]
        self.cursor = orig_cursor if inplace else (self.cursor + size)
        return string

def decode_header(data):
    """
    Decode the cervellotic header of the SM file.

    Returns
        A 2-element tuple with header-size and a list of channel labels
    """
    decoder = Decoder(data)
    version = decoder.readscalar()
    string1 = decoder.readstring()   # -> Comment
    string2 = decoder.readstring()   # -> 'Simple'
    pointer1 = decoder.readscalar()  # -> Pointer to 8 bytes before the end
    string3 = decoder.readstring()   # -> File section type

    magic1 = decoder.readscalar()
    magic2 = decoder.readscalar()

    col1_name = decoder.readstring()
    col1_resolution = decoder.readscalar(size=8, basetype='f')
    col1_offset = decoder.readscalar(size=8, basetype='f')
    col1_bho = decoder.readscalar()

    col2_name = decoder.readstring()
    col2_resolution = decoder.readscalar(size=8, basetype='f')
    col2_offset = decoder.readscalar(size=8, basetype='f')
    col2_bho = decoder.readscalar()

    col3_name = decoder.readstring()
    col3_resolution = decoder.readscalar(size=8, basetype='f')
    col3_offset = decoder.readscalar(size=8, basetype='f')
    num_channels = decoder.readscalar()

    ch_labels = [decoder.readstring() for _ in range(num_channels)]

    return decoder.cursor, ch_labels


def load_sm(fname, return_labels=False):
    """Read an SM data file.

    Return
        timestamps, detectors and optionally a list of detectors labels
    """
    with open(fname, 'rb') as f:
        fulldata = f.read()

    header_size, labels = decode_header(fulldata)
    rawdata = fulldata[header_size:]

    # Remove the end of the file
    end_field1 = 4
    end_str = 'End Of Run'
    end_field2 = 12
    valid_size = len(rawdata) - end_field1 - len(end_str) - end_field2

    # Description of the record element in the file
    sm_dtype = np.dtype([('timestamp', '>i8'), ('detector', '>u4')])

    # View of the binary data as an array (no copy performed)
    data = np.frombuffer(rawdata[:valid_size], dtype=sm_dtype)

    # Swap byte order inplace to little endian
    data.setflags(write=True)
    data = data.byteswap(True).newbyteorder()

    if return_labels:
        return data['timestamp'], data['detector'], labels
    else:
        return data['timestamp'], data['detector']
