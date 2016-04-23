#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load and decode files from PicoQuant
hardware.

The primary exported functions are:

- :func:`load_ht3` which returns decoded
  timestamps, detectors, nanotimes and metadata from an HT3 file.
- :func:`load_pt3` which returns decoded
  timestamps, detectors, nanotimes and metadata from a PT3 file.

Other lower level functions are:

- :func:`ht3_reader` which loads metadata and raw t3 records from HT3 files
- :func:`pt3_reader` which loads metadata and raw t3 records from PT3 files
- :func:`process_t3records` which decodes the t3 records returning
  timestamps (after overflow correction), detectors and TCSPC nanotimes.

Note that the functions performing overflow/rollover correction
can take advantage of numba, if installed, to significanly speed-up
the processing.
"""

from past.builtins import xrange
from builtins import zip

import os
import numpy as np

has_numba = True
try:
    import numba
except ImportError:
    has_numba = False


def load_ht3(filename, ovcfunc=None):
    """Load data from a PicoQuant .ht3 file.

    Arguments:
        filename (string): the path of the HT3 file to be loaded.
        ovcfunc (function or None): function to use for overflow/rollover
            correction of timestamps. If None, it defaults to the
            fastest available implementation for the current machine.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with metadata containing at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, timestamps_unit, nanotimes_unit, meta = ht3_reader(filename)
    detectors, timestamps, nanotimes = process_t3records(
        t3records, time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
        ovcfunc=ovcfunc)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit})

    return timestamps, detectors, nanotimes, meta

def load_pt3(filename, ovcfunc=None):
    """Load data from a PicoQuant .pt3 file.

    Arguments:
        filename (string): the path of the PT3 file to be loaded.
        ovcfunc (function or None): function to use for overflow/rollover
            correction of timestamps. If None, it defaults to the
            fastest available implementation for the current machine.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with metadata containing at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, timestamps_unit, nanotimes_unit, meta = pt3_reader(filename)
    detectors, timestamps, nanotimes = process_t3records(
        t3records, time_bit=16, dtime_bit=12, ch_bit=4, special_bit=False,
        ovcfunc=ovcfunc)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit})

    return timestamps, detectors, nanotimes, meta

def ht3_reader(filename):
    """Load raw t3 records and metadata from an HT3 file.
    """
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
            ('RepeatTime',      'int32'),
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

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        hardware2=hardware2, hardware3=hardware3,
                        inputs=inputs, ttmode=ttmode, imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata


def pt3_reader(filename):
    """Load raw t3 records and metadata from a PT3 file.
    """
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
            ('RoutingChannels',   'int32' ),
            ('NumberOfBoards',    'int32' ),
            ('ActiveCurve',       'int32' ),
            ('MeasurementMode',   'int32' ),
            ('SubMode',           'int32' ),
            ('RangeNo',           'int32' ),
            ('Offset',            'int32' ),
            ('AcquisitionTime',   'int32' ),   # in ms
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

        if header['FormatVersion'][0] != b'2.0':
            raise IOError(("Format '%s' not supported. "
                           "Only valid format is '2.0'.") % \
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
            ('RepeatTime',       'int32'),
            ('RepeatWaitTime',  'int32'),
            ('ScriptName',      'S20'  )])
        repeatgroup = np.fromfile(f, repeat_dtype, count=1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Hardware information header
        hw_dtype = np.dtype([
            ('HardwareIdent',   'S16'  ),
            ('HardwarePartNo',  'S8'   ),
            ('HardwareSerial',  'int32'),
            ('SyncDivider',     'int32'),
            ('CFDZeroCross0',   'int32'),
            ('CFDLevel0',       'int32'),
            ('CFDZeroCross1',   'int32'),
            ('CFDLevel1',       'int32'),
            ('Resolution',      'f4'),
            ('RouterModelCode', 'int32'),
            ('RouterEnabled',   'int32')])
        hardware = np.fromfile(f, hw_dtype, count=1)

        rtr_dtype = np.dtype([
            ('InputType',       'int32'),
            ('InputLevel',      'int32'),
            ('InputEdge',       'int32'),
            ('CFDPresent',      'int32'),
            ('CFDLevel',        'int32'),
            ('CFDZCross',       'int32')])
        router = np.fromfile(f, rtr_dtype, count=4)

        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
            ('ExtDevices',      'int32' ),
            ('Reserved1',       'int32' ),
            ('Reserved2',       'int32' ),
            ('InpRate0',        'int32' ),
            ('InpRate1',        'int32' ),
            ('StopAfter',       'int32' ),
            ('StopReason',      'int32' ),
            ('nRecords',        'int32' ),
            ('ImgHdrSize',      'int32')])
        ttmode = np.fromfile(f, ttmode_dtype, count=1)

        # Special header for imaging. How many of the following ImgHdr
        # array elements are actually present in the file is indicated by
        # ImgHdrSize above.
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'])

        timestamps_unit = 1./ttmode['InpRate0']
        nanotimes_unit = 1e-9*hardware['Resolution']

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        router=router, ttmode=ttmode, imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata


def process_t3records(t3records, time_bit=10, dtime_bit=15,
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
    print('NOTE: You can speed-up the loading time by installing numba.')
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
