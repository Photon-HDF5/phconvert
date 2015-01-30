#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
# The original function to load .pt3 files is from Dominic Waithe software
# (https://github.com/dwaithe/FCS_point_correlator) and released under GPLv2.
#

from past.builtins import xrange
from builtins import zip

import os
import numpy as np
import struct

has_numba = True
try:
    import numba
except ImportError:
    has_numba = False


def load_ht3(filename, ovcfunc=None):
    """Load data from a PicoQuant .ht3 file.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with meatadata conaining at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found."

    t3records, timestamps_unit, nanotimes_unit, meta = ht3_reader(filename)
    detectors, timestamps, nanotimes = process_t3records_ht3(
        t3records, time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
        ovcfunc=ovcfunc)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit})

    return timestamps, detectors, nanotimes, meta

def ht3_reader(filename):
    with open(filename, 'rb') as f:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Binary file header
        header_dtype = np.dtype([
            ('Ident',             'S16'   ),
            ('FormatVersion',     'S6'    ),
            ('CreatorName',       'S18'   ),
            ('CreatorVersion',    'S12'   ),
            ('FileTime',          'S18'   ),
            ('CRLF',              'S2'    ),
            ('Comment',           'S256'  ),
            ('NumberOfCurves',    'int32' ),
            ('BitsPerRecord',     'int32' ),   # bits in each T3 record
            ('ActiveCurve',       'int32' ),
            ('MeasurementMode',   'int32' ),
            ('SubMode',           'int32' ),
            ('Binning',           'int32' ),
            ('Resolution',        'double'),  # in ps
            ('Offset',            'int32' ),
            ('Tacq',              'int32' ),   # in ms
            ('StopAt',            'uint32'),
            ('StopOnOvfl',        'int32' ),
            ('Restart',           'int32' ),
            ('DispLinLog',        'int32' ),
            ('DispTimeAxisFrom',  'int32' ),
            ('DispTimeAxisTo',    'int32' ),
            ('DispCountAxisFrom', 'int32' ),
            ('DispCountAxisTo',   'int32' ),
        ])
        header = np.fromfile(f, dtype=header_dtype, count=1)

        if header['FormatVersion'][0] != b'1.0':
            raise IOError(("Format '%s' not supported. "
                           "Only valid format is '1.0'.") % \
                           header['FormatVersion'][0])

        dispcurve_dtype = np.dtype([
            ('DispCurveMapTo', 'int32'),
            ('DispCurveShow',  'int32')])
        dispcurve = np.fromfile(f, dispcurve_dtype, count=8)

        params_dtype = np.dtype([
            ('ParamStart', 'f4'),
            ('ParamStep',  'f4'),
            ('ParamEnd',   'f4')])
        params = np.fromfile(f, params_dtype, count=3)

        repeat_dtype = np.dtype([
            ('RepeatMode',      'int32'),
            ('RepeatsPerCurve', 'int32'),
            ('RepatTime',       'int32'),
            ('RepeatWaitTime',  'int32'),
            ('ScriptName',      'S20'  )])
        repeatgroup = np.fromfile(f, repeat_dtype, count=1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Hardware information header
        hw_dtype = np.dtype([
            ('HardwareIdent',   'S16'  ),
            ('HardwarePartNo',  'S8'   ),
            ('HardwareSerial',  'int32'),
            ('nModulesPresent', 'int32')])   # 10
        hardware = np.fromfile(f, hw_dtype, count=1)

        hw2_dtype = np.dtype([
            ('ModelCode', 'int32'),
            ('VersionCode', 'int32')])
        hardware2 = np.fromfile(f, hw2_dtype, count=10)

        hw3_dtype = np.dtype([
            ('BaseResolution',   'double'),
            ('InputsEnabled',    'uint64'),
            ('InpChansPresent',  'int32' ),
            ('RefClockSource',   'int32' ),
            ('ExtDevices',       'int32' ),
            ('MarkerSettings',   'int32' ),
            ('SyncDivider',      'int32' ),
            ('SyncCFDLevel',     'int32' ),
            ('SyncCFDZeroCross', 'int32' ),
            ('SyncOffset', 'int32')])
        hardware3 = np.fromfile(f, hw3_dtype, count=1)

        # Channels' information header
        input_dtype = np.dtype([
            ('InputModuleIndex',  'int32'),
            ('InputCFDLevel',     'int32'),
            ('InputCFDZeroCross', 'int32'),
            ('InputOffset',       'int32'),
            ('InputRate',         'int32')])
        inputs = np.fromfile(f, input_dtype,
                             count=hardware3['InpChansPresent'])

        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
            ('SyncRate',   'int32' ),
            ('StopAfter',  'int32' ),
            ('StopReason', 'int32' ),
            ('ImgHdrSize', 'int32' ),
            ('nRecords',   'uint64')])
        ttmode = np.fromfile(f, ttmode_dtype, count=1)

        # Special header for imaging. How many of the following ImgHdr
        # array elements are actually present in the file is indicated by
        # ImgHdrSize above.
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'])

        timestamps_unit = 1./ttmode['SyncRate']
        nanotimes_unit = 1e-12*header['Resolution']

        metadata = dict(header=header, params=params, repeatgroup=repeatgroup,
                        hardware=hardware, hardware2=hardware2,
                        hardware3=hardware3, inputs=inputs, ttmode=ttmode,
                        imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata

def process_t3records_ht3(t3records, time_bit=10, dtime_bit=15,
                          ch_bit=6, special_bit=True, ovcfunc=None):
    """Extract the different fields from the raw t3records array (.ht3).

    Returns:
        3 arrays representing detectors, timestamps and nanotimes.
    """
    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert dtime_bit <= 16

    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit + dtime_bit), 2**ch_bit - 1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, time_bit), 2**dtime_bit - 1).astype('uint16')

    assert time_bit <= 16
    dt = np.dtype([('low16', 'uint16'), ('high16', 'uint16')])

    t3records_low16 = np.frombuffer(t3records, dt)['low16']     # View
    timestamps = t3records_low16.astype(np.int64)               # Copy
    np.bitwise_and(timestamps, 2**time_bit - 1, out=timestamps)

    overflow_ch = 2**ch_bit - 1
    overflow = 2**time_bit
    if ovcfunc is None:
        ovcfunc = _correct_overflow
    ovcfunc(timestamps, detectors, overflow_ch, overflow)
    return detectors, timestamps, nanotimes

def _correct_overflow1(timestamps, detectors, overflow_ch, overflow):
    overflow_correction = 0
    for i in xrange(detectors.size):
        if detectors[i] == overflow_ch:
            overflow_correction += overflow
        timestamps[i] += overflow_correction

def _correct_overflow2(timestamps, detectors, overflow_ch, overflow):
    index_overflows = np.where((detectors == overflow_ch))[0]
    for n, (idx1, idx2) in enumerate(zip(index_overflows[:-1],
                                         index_overflows[1:])):
        timestamps[idx1:idx2] += (n + 1)*overflow
    timestamps[idx2:] += (n + 2)*overflow

if has_numba:
    _correct_overflow = numba.jit('void(i8[:], u1[:], u4, u8)')(
        _correct_overflow1)
else:
    _correct_overflow = _correct_overflow2


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
