#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load and decode files from PicoQuant
hardware.

The main functions to decode PicoQuant files (PTU, HT3, PT3, T3R) are respectively:

- :func:`load_ptu`
- :func:`load_ht3`
- :func:`load_pt3`
- :func:`load_t3r`

These functions return the arrays timestamps (also called macro-time or timetag), 
detectors (or channel), nanotimes (also called micro-time or TCSPC time) and an
additional metadata dict.

Other lower level functions are:

- :func:`ptu_reader` to load metadata and raw t3 records from PTU files
- :func:`ht3_reader` to load metadata and raw t3 records from HT3 files
- :func:`pt3_reader` to load metadata and raw t3 records from PT3 files
- :func:`process_t3records` to decode the t3 records and return
  timestamps (after overflow correction), detectors and TCSPC nanotimes.
- :func:`process_t3records_t3rfile` to decode the t3 records for t3r files.
- :func:`process_t2records` to decode the t2 records and return
  timestamps (after overflow correction) and detectors.

The functions performing overflow/rollover correction
can take advantage of numba, if installed, to significanly speed-up
the processing.
"""

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

 
# Constants used to decode the PQ file headers
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
    rtMultiHarpNT3   = 0x00010307,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T3), HW: $07 (MultiHarp150N)
    rtMultiHarpNT2   = 0x00010207,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $07 (MultiHarp150N))
    )

# Reverse mappings
_ptu_tag_type_r = {v: k for k, v in _ptu_tag_type.items()}
_ptu_rec_type_r = {v: k for k, v in _ptu_rec_type.items()}



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
        'tags'. The data in the PTU file header is returned as a
        dictionary of "tags". Each item in the dictionary has 'idx', 'type', 
        'value' and 'offset' keys. Some tags also have a 'data' key.
        Use :func:`_ptu_print_tags` to print the tags as an easy-to-read 
        table.

    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, record_type, tags = ptu_reader(filename)

    if record_type == 'rtPicoHarpT3':
        detectors, timestamps, nanotimes = process_t3records(
            t3records, time_bit=16, dtime_bit=12, ch_bit=4, special_bit=False,
            ovcfunc=ovcfunc)
    elif record_type == 'rtHydraHarpT3':
        detectors, timestamps, nanotimes = process_t3records(
            t3records, time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
            ovcfunc=ovcfunc)
    elif record_type in ('rtHydraHarp2T3', 'rtTimeHarp260NT3',
                         'rtTimeHarp260PT3'):
        detectors, timestamps, nanotimes = process_t3records(
            t3records, time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
            ovcfunc=_correct_overflow_nsync)
    elif record_type in ('rtHydraHarp2T2', 'rtTimeHarp260NT2','rtTimeHarp260PT2', 'rtMultiHarpNT3'):
        detectors, timestamps = process_t2records(t3records,
                time_bit=25, ch_bit=6, special_bit=True,
                ovcfunc=_correct_overflow_nsync)
        nanotimes = None
    elif record_type in ('rtPicoHarpT2'):
        detectors, timestamps = process_t2records(t3records,
                time_bit=28, ch_bit=4, special_bit=False,
                ovcfunc=ovcfunc)
        nanotimes = None
    else:
        msg = ('Sorry, decoding "%s" record type is not implemented!' %
               record_type)
        raise NotImplementedError(msg)

    # Get the metadata
    acquisition_duration = tags['MeasDesc_AcquisitionTime']['value'] * 1e-3
    ctime_t = time.strptime(tags['File_CreatingTime']['value'],
                            "%Y-%m-%d %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    hw_type = tags['HW_Type']
    if isinstance(hw_type, list):
        hw_type = hw_type[0]
    meta = {
        'timestamps_unit': tags['MeasDesc_GlobalResolution']['value'], # both T3 and T2
        'acquisition_duration': acquisition_duration,
        'software': tags['CreatorSW_Name']['data'],
        'software_version': tags['CreatorSW_Version']['data'],
        'creation_time': creation_time,
        'hardware_name': hw_type['data'],
        'record_type': record_type,
        'tags': _convert_multi_tags(tags)}
    if record_type.endswith('T3'):
        meta['nanotimes_unit'] = tags['MeasDesc_Resolution']['value']
        meta['laser_repetition_rate'] = tags['TTResult_SyncRate']['value']
    return timestamps, detectors, nanotimes, meta


def load_phu(filename):
    """Load data from a PicoQuant .phu file.

    Arguments:
        filename (string): the path of the PHU file to be loaded.
 
    Returns:
        A tuple of histograms, histogram resolution, and tags.
        The latter is an dictionary of tags contained
        in the file header. Each item in the dictionary has 'idx', 'type',
        'value' and 'offset' keys. Some tags also have a 'data' key.
        Use :func:`_ptu_print_tags` to print the tags as an easy-to-read 
        table.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename
    histograms, histo_resolution, tags = phu_reader(filename)
    acquisition_duration = tags['MeasDesc_AcquisitionTime']['value'] 
    acquisition_duration *= 1e-3  # in seconds
    meta = {'acquisition_duration': acquisition_duration, 'tags': tags}
    return histograms, histo_resolution, meta


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
    ctime_t = time.strptime(meta['header']['FileTime'][0].decode(),
                            "%d/%m/%y %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit,
                 'acquisition_duration': meta['header']['Tacq'][0] * 1e-3,
                 'laser_repetition_rate': meta['ttmode']['SyncRate'],
                 'software': meta['header']['CreatorName'][0].decode(),
                 'software_version': meta['header']['CreatorVersion'][0].decode(),
                 'creation_time': creation_time,
                 'hardware_name': meta['header']['Ident'][0].decode(),
                 })
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
    acquisition_duration = meta['header']['AcquisitionTime'][0] * 1e-3
    ctime_t = time.strptime(meta['header']['FileTime'][0].decode(),
                            "%d/%m/%y %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit,
                 'acquisition_duration': acquisition_duration,
                 'laser_repetition_rate': meta['ttmode']['InpRate0'],
                 'software': meta['header']['CreatorName'][0].decode(),
                 'software_version': meta['header']['CreatorVersion'][0].decode(),
                 'creation_time': creation_time,
                 'hardware_name': meta['header']['Ident'][0].decode(),
                 })
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
    """Read the header and the raw t3 or t2 records from a PTU file.
    """
    # All the info about the PTU format has been inferred from PicoQuant demo:
    # https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PTU/C/ptudemo.cc
    
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
    tags, offset = _read_header_tags(s)

    # A view of the t3records as a numpy array (no new memory is allocated)
    num_records = tags['TTResult_NumberOfRecords']['value']
    t3records = np.frombuffer(s, dtype='uint32', count=num_records,
                              offset=offset)
    record_type = _ptu_rec_type_r[tags['TTResultFormat_TTTRRecType']['value']]
    return t3records, record_type, tags


def _read_header_tags(s):
    """Read the header tags and return an OrderedDict.

    Each item in `tags` is a dict as returned by _ptu_read_tag().
    The input `s` is a binary-string containing the raw binary data file.
    """
    offset = 16                # initial bytes to skip
    FileTagEnd = "Header_End"  # Last tag of the header (BLOCKEND)
    tag_end_offset = s.find(FileTagEnd.encode()) + len(FileTagEnd)

    tags = OrderedDict()
    tagname, tag, offset = _ptu_read_tag(s, offset)
    tags[tagname] = tag
    while offset < tag_end_offset:
        tagname, tag, offset = _ptu_read_tag(s, offset)
        # In case a `tagname` appears multiple times, we make a list
        # to hold all the tags with the same name
        if tagname in tags.keys():
            if not isinstance(tags[tagname], list):
                tags[tagname]=[tags[tagname]]
            tags[tagname].append(tag)
        else:
            tags[tagname] = tag

    # Make sure we have read the last tag
    assert list(tags.keys())[-1] == FileTagEnd
    return tags, offset


def phu_reader(filename):
    """Load histogram records and metadata from a PHU file.
    """
    # All the info about the PHU format has been inferred from PicoQuant demo:
    # https://github.com/PicoQuant/PicoQuant-Time-Tagged-File-Format-Demos/blob/master/PHU/Matlab/Read_PHU.m
    # this format header is simalarly encoded as ptu files see ptu_reader

    # Load only the first few bytes to see is file is valid
    with open(filename, 'rb') as f:
        magic = f.read(8).rstrip(b'\0')
        version = f.read(8).rstrip(b'\0')
    if magic != b'PQHISTO':
        raise IOError("This file is not a valid PHU file. "
                      "Magic: '%s'." % magic)

    # Now load the entire file
    with open(filename, 'rb') as f:
        s = f.read()
    tags, _ =  _read_header_tags(s)

    # one has to loop over the different curves (histogram) stored in the phu file
    Ncurves = tags['HistoResult_NumberOfCurves']['value']
    # all Nbins should be equal between the Ncurves but there are as many tags as curves
    Nbins = tags['HistResDscr_HistogramBins'][0]['value'] 
    histograms = np.zeros((Ncurves, Nbins), dtype='uint32')

    # populate histograms and get some metadata
    histo_resolution=[]
    for ind_curve in range(Ncurves):
        histograms[ind_curve] = np.frombuffer(s, dtype='uint32',
            count=tags['HistResDscr_HistogramBins'][ind_curve]['value'],
            offset=tags['HistResDscr_DataOffset'][ind_curve]['value'])
        histo_resolution.append(
            tags['HistResDscr_MDescResolution'][ind_curve]['value'])
    return histograms, histo_resolution, tags


def t3r_reader(filename):
    """Load raw t3 records and metadata from a PT3 file.
    """
    with open(filename, 'rb') as f:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Binary file header
        header_dtype = np.dtype([
                ('Ident',             'S16'   ),
                ('SoftwareVersion',   'S6'    ),
                ('HardwareVersion',   'S6'    ),
                ('FileTime',          'S18'   ),
                ('CRLF',              'S2'    ),
                ('Comment',           'S256'  ),
                ('NumberOfChannels',  'int32'),
                ('NumberOfCurves',    'int32' ),
                ('BitsPerChannel',    'int32' ),   # bits in each T3 record
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
                ('RepeatTime',      'int32'),
                ('RepeatWaitTime',  'int32'),
                ('ScriptName',      'S20'  )])
        repeatgroup = np.fromfile(f, repeat_dtype, count=1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Hardware information header
        hw_dtype = np.dtype([
                ('BoardSerial',         'int32'),
                ('CFDZeroCross',        'int32'),
                ('CFDDiscriminatorMin', 'int32'),
                ('SYNCLevel',           'int32'),
                ('CurveOffset',         'int32'),
                ('Resolution',          'f4')])
        hardware = np.fromfile(f, hw_dtype, count=1)
        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
                ('TTTRGlobclock',   'int32' ),
                ('ExtDevices',      'int32' ),
                ('Reserved1',       'int32' ),
                ('Reserved2',       'int32' ),
                ('Reserved3',       'int32' ),
                ('Reserved4',       'int32' ),
                ('Reserved5',       'int32' ),
                ('SyncRate',        'int32' ),
                ('AverageCFDRate',  'int32' ),
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

        timestamps_unit = 100e-9  # 1./ttmode['SyncRate']
        nanotimes_unit = 1e-9*hardware['Resolution']

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        ttmode=ttmode, imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata


def _ptu_read_tag(s, offset):
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
    tag['offset'] = offset
    # Recover the name of the type (a string)
    tag['type'] = _ptu_tag_type_r[tag['type']]

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
        byte_string = s[offset: offset + tag['value']].rstrip(b'\0')
        try:
            tag['data'] = byte_string.decode()  # try decoding from UTF-8
        except UnicodeDecodeError:
            # Not UTF-8, trying 'latin1'
            # See https://github.com/Photon-HDF5/phconvert/issues/35
            tag['data'] = byte_string.decode('latin1')
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


def _convert_multi_tags(tags_dict):
    """Convert format of `tags_dict` from list of dict to dict of lists.
    
    When a tag in the file header is present multiple times, the values are
    accumulated in a list of dicts. This function replace the list-of-dicts
    with a dict-of-lists to facilitate saving to Photon-HDF5.
    """
    new_tags = tags_dict.copy()
    for tagname, tag in tags_dict.items():
        if isinstance(tag, list):
            new_tags[tagname] = _lod_to_dol(tag)
    return new_tags


def _lod_to_dol(lod):
    """Convert a list-of-dicts into a dict-of-lists.
    
    All the dicts in the input list must have the same keys.
    """
    assert isinstance(lod, list)
    assert len(lod) > 0
    keys = lod[0].keys()
    dol = {k: [] for k in keys}
    for d in lod:
        for k in keys:
            dol[k].append(d[k])
    return dol


def _dol_to_lod(dol):
    """Convert a dict-of-lists into a list-of-dicts.
    
    Reverse transformation of :func:`_lod_to_dol()`.
    """
    keys = list(dol.keys())
    lod = []
    for i in range(len(dol[keys[0]])):
        lod.append({k: v[i] for k, v in dol.items()})
    return lod


def _unconvert_multi_tags(tags_dict):
    new_tags = tags_dict.copy()
    for tagname, tag in tags_dict.items():
        keys = list(tag.keys())
        if isinstance(tag[keys[0]], list):
            new_tags[tagname] = _dol_to_lod(tag)
    return new_tags


def _ptu_print_tags(tags):
    """Print a table of tags from a PTU file header."""
    def _byte_to_str(x):
        if isinstance(x, bytes):
            # When loading from HDF5 string are binary
            x = x.decode()
        return x

    is_dol = True
    for tagname, tag in tags.items():
        if isinstance(tag, list):
            is_dol = False
            break
    if is_dol:
        tags = _unconvert_multi_tags(tags)
    for n in tags:
        start = 'D'  # mark for duplicated tags
        tags_n = tags[n]
        if not isinstance(tags[n], list):
            tags_n = [tags_n]
            start = ' '
        for tag in tags_n:
            tag_type = _byte_to_str(tag["type"])
            if tag_type == 'tyFloat8':
                value = f'{tag["value"]:20.4g}'
            else:
                value = f'{tag["value"]:>20}'
            endline = '\n'
            if tag_type == 'tyAnsiString':
                endline = _byte_to_str(tag['data']) + '\n'
            line = f'{start} {tag["offset"]:4} {n:28s} {value} {tag["idx"]:8}  {tag_type:12} '
            print(line, end=endline)


def process_t3records(t3records, time_bit=10, dtime_bit=15,
                      ch_bit=6, special_bit=True, ovcfunc=None):
    """Extract the different fields from the raw t3records array.

    The input array of t3records is an array of "records" (a C struct).
    It packs all the information of each detected photons. This function
    decodes the different fields and returns 3 arrays
    containing the timestamps (i.e. macro-time or number of sync,
    few-ns resolution), the nanotimes (i.e. the micro-time or TCSPC time,
    ps resolution) and the detectors.

    t3records have these fields (in little-endian order)::

        | Optional special bit | detectors | nanotimes | timestamps |
          MSB                                                   LSB

    Bit allocation of these fields, starting from the MSB:

    - **special bit**: 1 bit if `special_bit = True` (default), else no special bit.
    - **channel**: default 6 bit, (argument `ch_bit`), detector or special marker
    - **nanotimes**: default 15 bit (argument `dtime_bit`), nanotimes (TCSPC time)
    - **timestamps**: default 10 bit, (argument `time_bit`), the timestamps (macro-time)
    
    **Timestamps**: The returned timestamps are overflow-corrected, and therefore
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

    """
    Notes on detectors:

        The bit allocation in the record is, starting from the MSB::

            special: 1
            channel: 6
            dtime: 15
            nsync: 10
        
        If the special bit is clear, it's a regular event record.
        If the special bit is set, the following interpretation of 
        the channel code is given:

        - code 63 (all bits ones) identifies a sync count overflow, 
          increment the sync count overflow accumulator. For 
          HydraHarp V1 ($00010304) it means always one overflow. 
          For all other types the number of overflows can be read from nsync value.
        - codes from 1 to 15 identify markers, the individual bits are external markers.

        If detectors is above 64 then it is a special record.
            
            detectors==127 =>overflow
            detectors==65 => Marker 1 event
            detectors==66 => Marker 2 event
            ...
            detectors==79 => Marker 15 event
        else if
            detectors==0 => regular event regular detector 0
            detectors==1 => regular event regular detector 1
            detectors==2 => regular event regular detector 2
            ...

    """

    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 16
    assert time_bit + dtime_bit + ch_bit == 32

    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit + dtime_bit),
        2**ch_bit - 1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, time_bit),
        2**dtime_bit - 1).astype('uint16')

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


def  process_t2records(t2records, time_bit=25,
                      ch_bit=6, special_bit=True, ovcfunc=None):
    """Extract the different fields from the raw t2records array.

    The input array of t2records is an array of "records" (a C struct).
    It packs all the information of each detected photons. This function
    decodes the different fields and returns 2 arrays
    containing the timestamps (also called macro-time or timetag) and 
    the detectors (or channel).

    t2records have these fields (in little-endian order)::

        | Optional special bit | detectors |  timestamps |
          MSB                                        LSB

    - **special bit**: 1 bit if `special_bit = True` (default), else no special bit.
    - **channel**: default 6 bit, (argument `ch_bit`), detector or special marker
    - **timestamps**: default 25 bit, (argument `time_bit`), the timestamps (macro-time)

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
        t2records (array): raw array of t2records as saved in the
            PicoQuant file.
        time_bit (int): number of bits in the t2record used for timestamps
        ch_bit (int): number of bits in the t2record used for the detector
            number.
        special_bit (bool): if True the t2record contains a special bit
            for overflow correction or external markers.
            This special bit will become the MSB in the returned detectors
            array. If False, it assumes no special bit in the t2record.
        ovcfunc (function or None): function to perform overflow correction
            of timestamps. If None use the default function. The default
            function is the numba-accelerated version if numba is installed
            otherwise it is function using plain numpy.

    Returns:
        A 2-element tuple containing the following 1D arrays (all of the same
        length):

        - **timestamps** (*array of int64*): the macro-time (or number of sync)
          of each photons after overflow correction. Units are specified in
          the file header.
        - **detectors** (*arrays of uint8*): detector number. When
          `special_bit = True` the highest bit in `detectors` will be
          the special bit.
    """
    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 25
    assert time_bit + ch_bit== 32

    # called "dtime" in picoquant library
    timestamps = np.bitwise_and(t2records,2**time_bit - 1).astype('int64')

    # called "channel" in picoquant library
    detectors = np.bitwise_and(
        np.right_shift(t2records, time_bit), 2**(ch_bit) - 1).astype('uint8') 
        
    overflow_ch = 2**ch_bit - 1
    overflow = 2**time_bit
    if ovcfunc is None:
        ovcfunc = _correct_overflow
    ovcfunc(timestamps, detectors, overflow_ch, overflow)
    return detectors, timestamps


def process_t3records_t3rfile(t3records, reserved=1, valid=1, time_bit=12,
                              dtime_bit=16, ch_bit=2, special_bit=False):
    """ Decode t3records from .T3R files.

    See also :func:`process_t3records`.

    Arguments:
        reserved (int): reserved bit
        valid (int): valid bit. If valid==1 the Data == Channel
            else Data = Overflow[1], Reserved[8], Marker[3]
        time_bit (int): bits for nanotimes
        dtime_bit (int): bits for TimeTag (timestamps)
        ch_bit (int): number of bits encoding channel
        special_bit (bool): True if the record contatins the special bit.

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
    assert time_bit+reserved+valid+dtime_bit+ch_bit == 32

    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit+dtime_bit+reserved+valid),
        2**ch_bit - 1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, dtime_bit),
        2**time_bit - 1).astype('uint16')

    valid = np.bitwise_and(
        np.right_shift(t3records, time_bit+dtime_bit+reserved+valid),
        2**valid - 1).astype('uint8')

    dt = np.dtype([('low16', 'uint16'), ('high16', 'uint16')])
    t3records_low16 = np.frombuffer(t3records, dt)['low16']     # View
    timestamps = t3records_low16.astype(np.int64)               # Copy
    np.bitwise_and(timestamps, 2**dtime_bit - 1, out=timestamps)

    overflow = 2**dtime_bit
    _correct_overflow1(timestamps, valid, 0, overflow)
    return detectors, timestamps, nanotimes


def _correct_overflow1(timestamps, detectors, overflow_ch, overflow):
    """Apply overflow correction when each overflow has a special timestamp.
    """
    overflow_correction = 0
    for i in range(detectors.size):
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


def _correct_overflow_nsync_naive(timestamps, detectors, overflow_ch,
                                  overflow):
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
