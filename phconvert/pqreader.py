#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
# The original function to load .pt3 files is from Dominic Waithe software
# (https://github.com/dwaithe/FCS_point_correlator) and released under GPLv2.
#

import os
import numpy as np
import struct


def load_pt3(filename):
    """Load data from a PicoQuant .pt3 file.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with meatadata conaining at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found."

    t3records, timestamps_unit, nanotimes_unit = pt3record_reader(filename)
    detectors, timestamps, nanotimes = process_t3records(t3records)
    metadata = {'timestamps_unit': timestamps_unit,
                'nanotimes_unit': nanotimes_unit}

    return timestamps, detectors, nanotimes, metadata



def pt3record_reader(filename):
    """Return the raw uint32 T3 records from a PicoQuant .pt3 file.
    """
    with open(filename, 'rb') as f:
        Ident = f.read(16)
        FormatVersion = f.read(6)
        CreatorName = f.read(18)
        CreatorVersion = f.read(12)
        FileTime = f.read(18)
        CRLF = f.read(2)
        CommentField = f.read(256)

        Curves = struct.unpack('i', f.read(4))[0]
        BitsPerRecord = struct.unpack('i', f.read(4))[0]
        RoutingChannels = struct.unpack('i', f.read(4))[0]
        NumberOfBoards = struct.unpack('i', f.read(4))[0]
        ActiveCurve = struct.unpack('i', f.read(4))[0]
        MeasurementMode = struct.unpack('i', f.read(4))[0]
        SubMode = struct.unpack('i', f.read(4))[0]
        RangeNo = struct.unpack('i', f.read(4))[0]
        Offset = struct.unpack('i', f.read(4))[0]
        AcquisitionTime = struct.unpack('i', f.read(4))[0]
        StopAt = struct.unpack('i', f.read(4))[0]
        StopOnOvfl = struct.unpack('i', f.read(4))[0]
        Restart = struct.unpack('i', f.read(4))[0]
        DispLinLog = struct.unpack('i', f.read(4))[0]
        DispTimeFrom = struct.unpack('i', f.read(4))[0]
        DispTimeTo = struct.unpack('i', f.read(4))[0]
        DispCountFrom = struct.unpack('i', f.read(4))[0]
        DispCountTo = struct.unpack('i', f.read(4))[0]

        DispCurveMapTo = []
        DispCurveShow = []
        for i in range(8):
            DispCurveMapTo.append(struct.unpack('i', f.read(4))[0]);
            DispCurveShow.append(struct.unpack('i', f.read(4))[0]);

        ParamStart = []
        ParamStep = []
        ParamEnd = []
        for i in range(3):
            ParamStart.append(struct.unpack('i', f.read(4))[0]);
            ParamStep.append(struct.unpack('i', f.read(4))[0]);
            ParamEnd.append(struct.unpack('i', f.read(4))[0]);

        RepeatMode = struct.unpack('i', f.read(4))[0]
        RepeatsPerCurve = struct.unpack('i', f.read(4))[0]
        RepeatTime = struct.unpack('i', f.read(4))[0]
        RepeatWait = struct.unpack('i', f.read(4))[0]
        ScriptName = f.read(20)

        # The next is a board specific header
        HardwareIdent = f.read(16)
        HardwareVersion = f.read(8)
        HardwareSerial = struct.unpack('i', f.read(4))[0]
        SyncDivider = struct.unpack('i', f.read(4))[0]

        CFDZeroCross0 = struct.unpack('i', f.read(4))[0]
        CFDLevel0 = struct.unpack('i', f.read(4))[0]
        CFDZeroCross1 = struct.unpack('i', f.read(4))[0]
        CFDLevel1 = struct.unpack('i', f.read(4))[0]

        Resolution = struct.unpack('f', f.read(4))[0]

        # Below is new in format version 2.0
        RouterModelCode = struct.unpack('i', f.read(4))[0]
        RouterEnabled = struct.unpack('i', f.read(4))[0]

        # Router Ch1-4 (ich = 0..3)
        RouterParameters = ['InputType', 'InputLevel', 'InputEdge',
                            'CFDPresent', 'CFDLevel', 'CFDZeroCross']
        RouterCh = []
        for ich in range(4):
            RouterCh.append({})
            for param in RouterParameters:
                RouterCh[ich][param] = struct.unpack('i', f.read(4))[0]

        #The next is a T3 mode specific header.
        ExtDevices = struct.unpack('i', f.read(4))[0]
        Reserved1 = struct.unpack('i', f.read(4))[0]
        Reserved2 = struct.unpack('i', f.read(4))[0]
        CntRate0 = struct.unpack('i', f.read(4))[0]
        CntRate1 = struct.unpack('i', f.read(4))[0]
        StopAfter = struct.unpack('i', f.read(4))[0]
        StopReason = struct.unpack('i', f.read(4))[0]
        NumRecords = struct.unpack('i', f.read(4))[0]
        ImgHdrSize =struct.unpack('i', f.read(4))[0]

        # Special Header for imaging.
        if ImgHdrSize > 0:
            ImgHdr = struct.unpack('i', f.read(ImgHdrSize))[0]

        # Read all the T3 records in a byte string
        t3records_buffer = f.read(4*NumRecords)

    # View the T3 byte string as a uint32 array
    t3records = np.frombuffer(t3records_buffer, dtype='uint32')

    timestamps_unit = 1./CntRate0
    nanotimes_unit = 1e-9*Resolution
    return t3records, timestamps_unit, nanotimes_unit

def process_t3records(t3records):
    """Extract the different fields from the raw t3records array.

    Note that this function MODIFIES THE INPUT ARRAY in order to lower the
    memory usage.

    Returns:
        3 arrays representing detectors, timestamps and nanotimes.
    """
    # View t3records with a custom dtype to access differents bitfields
    # Note that in little-endian format the low bytes comes first
    t3dtype = np.dtype([('low16', np.uint16), ('high16', np.uint16)])

    # Allocate a new uint8 array and copy the detetctor/channel info
    detectors = np.zeros_like(t3records, dtype='uint8')
    np.right_shift(t3records, 28, out=detectors)

    # Get nanotimes "in-place" from the t3records by zeroing the highest 4 bits
    nanotimes = np.frombuffer(t3records, dtype=t3dtype)['high16']
    nanotimes.setflags(write=True)
    np.bitwise_and(nanotimes, 0x0FFF, out=nanotimes)

    # Get the raw timestamp (aka nsync) data without memory allocation
    raw_timestamps = np.frombuffer(t3records, dtype=t3dtype)['low16']

    # Find the overflow position index
    index_overflows = np.where((detectors == 15)*(nanotimes == 0))[0]

    # Allocate a int64 array for the corrected timestamps
    # and compute the overflow correction
    timestamps = raw_timestamps.astype(np.int64)
    max_time = 2**16
    for n, (idx1, idx2) in enumerate(zip(index_overflows[:-1],
                                     index_overflows[1:])):
        timestamps[idx1:idx2] += (n + 1)*max_time
    timestamps[idx2:] += (n + 2)*max_time

    return detectors, timestamps, nanotimes
