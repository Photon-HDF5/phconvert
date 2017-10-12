#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load and decode files from PicoQuant
hardware.

The three main functions to decode PTU, HT3 adn PT3 files are respectively:

- :func:`load_ptu`
- :func:`load_ht3`
- :func:`load_pt3`

These functions return the arrays timestamps, detectors, nanotimes and an
additional metadata dict.

Other lower level functions are:

- :func:`ptu_reader` which loads metadata and raw t3 records from PTU files
- :func:`ht3_reader` which loads metadata and raw t3 records from HT3 files
- :func:`pt3_reader` which loads metadata and raw t3 records from PT3 files
- :func:`process_t3records` which decodes the t3 records returning
  timestamps (after overflow correction), detectors and TCSPC nanotimes.

Note that the functions performing overflow/rollover correction
can take advantage of numba, if installed, to significanly speed-up
the processing.
"""

from __future__ import print_function, division
from past.builtins import xrange
from builtins import zip

import os
import struct
import time
from collections import OrderedDict
import numpy as np

has_numba = True
try:
    import numba
except ImportError:
    has_numba = False


def load_ptu(filename, ovcfunc=None):
    """Load data from a PicoQuant .ptu file.

    Arguments:
        filename (string): the path of the PTU file to be loaded.
        ovcfunc (function or None): function to use for overflow/rollover
            correction of timestamps. If None, it defaults to the
            fastest available implementation for the current machine.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with metadata containing the keys
        'timestamps_unit', 'nanotimes_unit', 'acquisition_duration' and
        'tags'. The value of 'tags' is an OrderedDict of tags contained
        in the PTU file header. Each item in the OrderedDict has 'idx', 'type'
        and 'value' keys. Some tags also have a 'data' key.

    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, timestamps_unit, nanotimes_unit, record_type, tags = \
        ptu_reader(filename)

    if record_type == 'rtPicoHarpT3':
        detectors, timestamps, nanotimes = process_t3records(t3records,
                time_bit=16, dtime_bit=12, ch_bit=4, special_bit=False,
                ovcfunc=ovcfunc)
    elif record_type in ('rtHydraHarp2T3', 'rtTimeHarp260NT3',
                         'rtTimeHarp260PT3'):
        detectors, timestamps, nanotimes = process_t3records(t3records,
                time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
                ovcfunc=_correct_overflow_nsync)
    else:
        msg = ('Sorry, decoding "%s" record type is not implemented!' %
               record_type)
        raise NotImplementedError(msg)

    acquisition_duration = tags['MeasDesc_AcquisitionTime']['value'] * 1e-3
    meta = {'timestamps_unit': timestamps_unit,
            'nanotimes_unit': nanotimes_unit,
            'acquisition_duration': acquisition_duration,
            'tags': tags}
    return timestamps, detectors, nanotimes, meta

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

def load_t3r(filename, ovcfunc=None):
    """Load data from a PicoQuant .pt3 file.

    Arguments:
        filename (string): the path of the t3r file to be loaded.
        ovcfunc (function or None): function to use for overflow/rollover
            correction of timestamps. If None, it defaults to the
            fastest available implementation for the current machine.

    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with metadata containing at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, timestamps_unit, nanotimes_unit, meta = t3r_reader(filename)
    detectors, timestamps, nanotimes = process_t3records_t3rfile(
        t3records, reserved=1, valid=1, time_bit=12, dtime_bit=16,
        ch_bit=2, special_bit=False)
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
                             count=hardware3['InpChansPresent'][0])

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
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'][0])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'][0])

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
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'][0])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'][0])

        timestamps_unit = 1./ttmode['InpRate0']
        nanotimes_unit = 1e-9*hardware['Resolution']

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        router=router, ttmode=ttmode, imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata


def ptu_reader(filename):
    """Load raw t3 records and metadata from a PTU file.
    """
    # All the info about the PTU format has been inferred from PicoQuant demo:
    # https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/cc/ptudemo.cc

    # Constants used to decode the header
    FileTagEnd = "Header_End"  # Last tag of the header (BLOCKEND)
    # Tag Types
    _ptu_tag_type = dict(
        tyEmpty8      = 0xFFFF0008,
        tyBool8       = 0x00000008,
        tyInt8        = 0x10000008,
        tyBitSet64    = 0x11000008,
        tyColor8      = 0x12000008,
        tyFloat8      = 0x20000008,
        tyTDateTime   = 0x21000008,
        tyFloat8Array = 0x2001FFFF,
        tyAnsiString  = 0x4001FFFF,
        tyWideString  = 0x4002FFFF,
        tyBinaryBlob  = 0xFFFFFFFF,
        )

    # Record Types
    _ptu_rec_type = dict(
        rtPicoHarpT3     = 0x00010303,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $03 (PicoHarp)
        rtPicoHarpT2     = 0x00010203,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $03 (PicoHarp)
        rtHydraHarpT3    = 0x00010304,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarpT2    = 0x00010204,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtHydraHarp2T3   = 0x01010304,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarp2T2   = 0x01010204,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtTimeHarp260NT3 = 0x00010305,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $05 (TimeHarp260N)
        rtTimeHarp260NT2 = 0x00010205,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $05 (TimeHarp260N)
        rtTimeHarp260PT3 = 0x00010306,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $06 (TimeHarp260P)
        rtTimeHarp260PT2 = 0x00010206,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)
        )

    # Reverse mappings
    _ptu_tag_type_r = {v: k for k, v in _ptu_tag_type.items()}
    _ptu_rec_type_r = {v: k for k, v in _ptu_rec_type.items()}

    # Load only the first few bytes to see is file is valid
    with open(filename, 'rb') as f:
        magic = f.read(8).rstrip(b'\0')
        version = f.read(8).rstrip(b'\0')
    if magic != b'PQTTTR':
        raise IOError("This file is not a valid PTU file. "
                      "Magic: '%s'." % magic)

    # Now load the entire file
    with open(filename, 'rb') as f:
        s = f.read()

    # Decode the header and save data in the OrderedDict `tags`
    # Each item in `tags` is a dict as returned by _ptu_read_tag()
    offset = 16
    tag_end_offset = s.find(FileTagEnd.encode()) + len(FileTagEnd)

    tags = OrderedDict()
    tagname, tag, offset = _ptu_read_tag(s, offset, _ptu_tag_type_r)
    tags[tagname] = tag
    while offset < tag_end_offset:
        tagname, tag, offset = _ptu_read_tag(s, offset, _ptu_tag_type_r)
        tags[tagname] = tag

    # Make sure we have read the last tag
    assert list(tags.keys())[-1] == FileTagEnd

    # A view of the t3recods as a numpy array (no new memory is allocated)
    num_records = tags['TTResult_NumberOfRecords']['value']
    t3records = np.frombuffer(s, dtype='uint32', count=num_records,
                              offset=offset)

    # Get some metadata
    timestamps_unit = 1 / tags['TTResult_SyncRate']['value']
    nanotimes_unit = tags['MeasDesc_Resolution']['value']
    record_type = _ptu_rec_type_r[tags['TTResultFormat_TTTRRecType']['value']]
    return t3records, timestamps_unit, nanotimes_unit, record_type, tags

def t3r_reader(filename):
    """Load raw t3 records and metadata from a PT3 file.
    """
    with open(filename, 'rb') as f:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Binary file header
        header_dtype = np.dtype([
                ('Ident',             'S16'   ),
                ('SoftwareVersion',     'S6'    ),
                ('HardwareVersion',     'S6'    ),
                ('FileTime',          'S18'   ),
                ('CRLF',              'S2'    ),
                ('Comment',           'S256'  ),
                ('NumberOfChannels',   'int32'),
                ('NumberOfCurves',    'int32' ),
                ('BitsPerChannel',     'int32' ),   # bits in each T3 record
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

        if header['SoftwareVersion'][0] != b'5.0':
            raise IOError(("Format '%s' not supported. "
                           "Only valid format is '5.0'.") % \
                           header['SoftwareVersion'][0])

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

                ('BoardSerial',     'int32'),
                ('CFDZeroCross',   'int32'),
                ('CFDDiscriminatorMin',   'int32'),
                ('SYNCLevel',       'int32'),
                ('CurveOffset',       'int32'),
                ('Resolution',      'f4')])
        hardware = np.fromfile(f, hw_dtype, count=1)
        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
                ('TTTRGlobclock',      'int32' ),
                ('ExtDevices',      'int32' ),
                ('Reserved1',       'int32' ),
                ('Reserved2',       'int32' ),
                ('Reserved3',       'int32' ),
                ('Reserved4',       'int32' ),
                ('Reserved5',       'int32' ),
                ('SyncRate',        'int32' ),
                ('AverageCFDRate',        'int32' ),
                ('StopAfter',       'int32' ),
                ('StopReason',      'int32' ),
                ('nRecords',        'int32' ),
                ('ImgHdrSize',      'int32')])
        ttmode = np.fromfile(f, ttmode_dtype, count=1)

        # Special header for imaging. How many of the following ImgHdr
        # array elements are actually present in the file is indicated by
        # ImgHdrSize above.
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'][0])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'][0])

        timestamps_unit = 100e-9 #1./ttmode['SyncRate']
        nanotimes_unit = 1e-9*hardware['Resolution']

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                         ttmode=ttmode, imghdr=ImgHdr)# router=router,
        return t3records, timestamps_unit, nanotimes_unit, metadata

def _ptu_print_tags(tags):
    """Print a table of tags from a PTU file header."""
    line = '{:30s} %s {:8}  {:12} '
    for n in tags:
        value_fmt = '{:>20}'
        if tags[n]['type'] == 'tyFloat8':
            value_fmt = '{:20.4g}'
        endline = '\n'
        if tags[n]['type'] == 'tyAnsiString':
            endline = tags[n]['data'] + '\n'  # hic sunt leones
        print((line % value_fmt).format(n, tags[n]['value'], tags[n]['idx'], tags[n]['type']),
              end=endline)

def _ptu_read_tag(s, offset, tag_type_r):
    """Decode a single tag from the PTU header struct.

    Returns:
        A dict with tag data. The keys 'idx', 'type' and 'value' are present
        in all tags. The key 'data' is present only for a few types of tags.
    """
    # Struct fields: 32-char string, int32, uint32, int64
    tag_struct = struct.unpack('32s i I q', s[offset:offset + 48])
    offset += 48
    # and save it into a dict
    tagname = tag_struct[0].rstrip(b'\0').decode()
    keys = ('idx', 'type', 'value')
    tag = {k: v for k, v in zip(keys, tag_struct[1:])}
    # Recover the name of the type (a string)
    tag['type'] = tag_type_r[tag['type']]

    # Some tag types need conversion
    if tag['type'] == 'tyFloat8':
        tag['value'] = np.int64(tag['value']).view('float64')
    elif tag['type'] == 'tyBool8':
        tag['value'] = bool(tag['value'])
    elif tag['type'] == 'tyTDateTime':
        TDateTime = np.uint64(tag['value']).view('float64')
        t = time.gmtime(_ptu_TDateTime_to_time_t(TDateTime))
        tag['value'] = time.strftime("%Y-%m-%d %H:%M:%S", t)

    # Some tag types have additional data
    if tag['type'] == 'tyAnsiString':
        tag['data'] = s[offset: offset + tag['value']].rstrip(b'\0').decode()
        offset += tag['value']
    elif tag['type'] == 'tyFloat8Array':
        tag['data'] = np.frombuffer(s, dtype='float', count=tag['value'] / 8)
        offset += tag['value']
    elif tag['type'] == 'tyWideString':
        # WideString use type WCHAR in the original C++ demo code.
        # WCHAR size is not fixed by C++ standard, but on windows
        # is 2 bytes and the default encoding is UTF-16.
        # I'm assuming this is what the PTU requires.
        tag['data'] = s[offset: offset + tag['value'] * 2].decode('utf16')
        offset += tag['value']
    elif tag['type'] == 'tyBinaryBlob':
        tag['data'] = s[offset: offset + tag['value']]
        offset += tag['value']

    return tagname, tag, offset

def _ptu_TDateTime_to_time_t(TDateTime):
    """Convert the weird time encoding used in PTU files to standard time_t."""
    EpochDiff = 25569  # days between 30/12/1899 and 01/01/1970
    SecsInDay = 86400  # number of seconds in a day
    return (TDateTime - EpochDiff) * SecsInDay

def process_t3records(t3records, time_bit=10, dtime_bit=15,
                      ch_bit=6, special_bit=True, ovcfunc=None):
    """Extract the different fields from the raw t3records array.

    The input array of t3records is an array of "records" (a C struct).
    It packs all the information of each detected photons. This function
    decodes the different fields and returns 3 arrays
    containing the timestamps (i.e. macro-time or number of sync,
    ns resolution), the nanotimes (i.e. the micro-time or TCSPC time,
    ps resolution) and the detectors.

    Assuming the t3records are in little-endian order, the fields are assumed
    in the following order::

        | Optional special bit | detectors | nanotimes | timestamps |
          MSB                                                   LSB

    - the lowest `time_bit` bits contain the timestamps
    - the next `dtime_bit` bits contain the nanotimes
    - the next `ch_bit` contain the detector number
    - if `special_bit = True`, the highest bit is the special bit.

    The returned timestamps are overflow-corrected, and therefore
    should be monotonically increasing. Each overflow event is marked by
    a special detector (or a special bit) and this information is used for
    the correction. These overflow "events" **are not removed** in the returned
    arrays resulting in spurious detectors. This choice has been made for
    safety (you can always go and check where there was an overflow) and for
    efficiency (removing a few elements requires allocating a new array that
    is potentially expensive for big data files). Under normal usage the
    additional detectors take negligible space and can be safely ignored.

    Arguments:
        t3records (array): raw array of t3records as saved in the
            PicoQuant file.
        time_bit (int): number of bits in the t3record used for timestamps
            (or macro-time).
        dtime_bit (int): number of bits in the t3record used for the nanotime
            (TCSPC time or micro-time)
        ch_bit (int): number of bits in the t3record used for the detector
            number.
        special_bit (bool): if True the t3record contains a special bit
            for overflow correction.
            This special bit will become the MSB in the returned detectors
            array. If False, it assumes no special bit in the t3record.
        ovcfunc (function or None): function to perform overflow correction
            of timestamps. If None use the default function. The default
            function is the numba-accelerated version is numba is installed
            otherwise it is function using plain numpy.

    Returns:
        A 3-element tuple containing the following 1D arrays (all of the same
        length):

        - **timestamps** (*array of int64*): the macro-time (or number of sync)
          of each photons after overflow correction. Units are specified in
          the file header.
        - **nanotimes** (*array of uint16*): the micro-time (TCSPC time), i.e.
          the time lag between the photon detection and the previous laser
          sync. Units (i.e. the bin width) are specified in the file header.
        - **detectors** (*arrays of uint8*): detector number. When
          `special_bit = True` the highest bit in `detectors` will be
          the special bit.
    """
    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 16
    assert time_bit + dtime_bit + ch_bit == 32

    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit + dtime_bit), 2**ch_bit - 1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, time_bit), 2**dtime_bit - 1).astype('uint16')

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

def process_t3records_t3rfile(t3records, reserved=1, valid=1, time_bit=12, dtime_bit=16,
                      ch_bit=2, special_bit=False):
    """ For processing file.t3r format
    time_bit: nanotimes
    dtime_bit: TimeTag
    if valid==1 the Data == Channel
    else Data = Overflow[1], Reserved[8], Marker[3]
    """
    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 16
    assert time_bit+reserved+valid+dtime_bit+ch_bit == 32
    
    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit+dtime_bit+reserved+valid), 2**ch_bit-1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, dtime_bit), 2**time_bit-1).astype('uint16')
    
    valid = np.bitwise_and(
        np.right_shift(t3records, time_bit+dtime_bit+reserved+valid), 2**valid-1).astype('uint8')

    dt = np.dtype([('low16', 'uint16'), ('high16', 'uint16')])
    t3records_low16 = np.frombuffer(t3records, dt)['low16']     # View
    timestamps = t3records_low16.astype(np.int64)               # Copy
    np.bitwise_and(timestamps, 2**dtime_bit - 1, out=timestamps)

    overflow_ch = 2**ch_bit - 1
    overflow = 2**dtime_bit
    _correct_overflow1(timestamps, valid, 0, overflow)        
    return detectors, timestamps, nanotimes

def _correct_overflow1(timestamps, detectors, overflow_ch, overflow):
    """Apply overflow correction when each overflow has a special timestamp.
    """
    overflow_correction = 0
    for i in xrange(detectors.size):
        if detectors[i] == overflow_ch:
            overflow_correction += overflow
        timestamps[i] += overflow_correction

def _correct_overflow2(timestamps, detectors, overflow_ch, overflow):
    """Apply overflow correction when each overflow has a special timestamp.
    """
    print('NOTE: You can speed-up the loading time by installing numba.')
    index_overflows = np.where((detectors == overflow_ch))[0]
    for n, (idx1, idx2) in enumerate(zip(index_overflows[:-1],
                                         index_overflows[1:])):
        timestamps[idx1:idx2] += (n + 1)*overflow
    timestamps[idx2:] += (n + 2)*overflow


def _correct_overflow_nsync(timestamps, detectors, overflow_ch, overflow):
    """Apply overflow correction when ov. timestamps contain # of overflows
    """
    index_overflows = np.where((detectors == overflow_ch))
    num_overflows = timestamps[index_overflows]
    cum_overflows = np.zeros(timestamps.size, dtype='int64')
    cum_overflows[index_overflows] = num_overflows
    np.cumsum(cum_overflows, out=cum_overflows)
    timestamps += (cum_overflows * overflow)
    # put nsync back in the overflow timestamps
    #timestamps[index_overflows] = cum_overflows


def _correct_overflow_nsync_naive(timestamps, detectors, overflow_ch, overflow):
    """Slow implementation of `_correct_overflow_nsync` used for testing.
    """ 
    overflow_correction = 0
    for i in range(detectors.size):
        if detectors[i] == overflow_ch:
            overflow_correction += (overflow * timestamps[i])
        timestamps[i] += overflow_correction

if has_numba:
    _correct_overflow = numba.jit('void(i8[:], u1[:], u4, u8)')(
        _correct_overflow1)
else:
    _correct_overflow = _correct_overflow2
