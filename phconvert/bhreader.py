"""
This module contains functions to load and decode files from Becker & Hickl
hardware.

The high-level function in this module are:
    - :func:`load_spc` which loads photon data from .spc file, along with .set file
    - :func:`load_set` which just loads the metadata from a .set file
    
Becker & Hickl SPC format
-------------------------

Becker & Hickl documentation for their file format is scattered across the
`TCSPC handbook <https://www.becker-hickl.com/literature/documents/flim/the-bh-tcspc-handbook/>`_
and in the SPC_data_file_structure.h file which can be located in the SPCM sofware
directory within the program files directory after installation of the B&H SPCM
sofware.

The format of the `.set` file is consistent across all versions and TCSPC cards.
The `.spc` file however has several potential layout depending on the B&H card
used to acquire the data.

The `.set` file starts with binary data according to a custom structure defined
in SPC_data_file_structure.h, most notably the first 2 bytes contain information
on the version of the sofware, and TCSPC card used. Next are 2 ascii formated
sections, IDENTIFICATION and SETUP. IDENTIFICATION contains general user specified
data, while SETUP contains specific information on settings of the TCSPC acquisition.

Depending on the B&H TCSPC card, the data will be stored according to one of the
following formats. In all cases, the 1st 48 or 32 bits are a header that defines
the macrotime clock (timestamps_unit), and number of routing bits. After the header,
all subsequent data are records of single photon arrivals, each record is always
the same number of bytes as the header.

"""

import os
from pathlib import Path
from typing import Union
from collections.abc import Callable
import warnings

import numpy as np

FileName = Union[str, bytes, os.PathLike] # for type hinting

class InvalidSetFileError(Exception):
    """Error indicating set file has been corrupted or otherwise invalid"""
    pass

#: dtype equivalent to the header structure of .set files (see B&H file SPC_data_file_structure.h)
_sethead_dtype:np.dtype = np.dtype([('revision', '<i2'),
                                    ('info_offs', '<i4'),('info_length', '<i2'),
                                    ('setup_offs', '<i4'), ('setup_length', '<i2'),
                                    ('data_block_offs', '<i4'), ('no_of_data_blocks', '<i2'), ('data_block_length', '<u4'),
                                    ('meas_desc_block_offs', '<i4'), ('no_of_meas_desc_blocks', '<i2'), ('meas_desc_block_length', '<i2'), 
                                    ('header_valid', '<u2'), ('reserved1', '<u4'), ('reserved2', '<u2'), ('chksum', '<u2')
                                   ])

#: int -> product no. map of codes in revision of _sethead_dtype (see B&H file SPC_data_file_structure.h)
_spc_codes:dict = {
    0x20:'SPC-130', 0x21:'SPC-600', 0x22:'SPC-630',
    0x23:'SPC-700', 0x24:'SPC-730', 0x25:'SPC-830',
    0x26:'SPC-140', 0x27:'SPC-930', 0x28:'SPC-150',
    0x29:'DPC-230', 0x2a:'SPC-130EM', 0x2b:'SPC-160',
    0x2e:'SPC-150N', 0x80:'SPC-150NX', 0x81:'SPC-160X',
    0x82:'SPC-160PCIE', 0x83:'SPC-130EMN', 0x84:'SPC-180N',
    0x85:'SPC-180NX',   0x86:'SPC-180NXX', 0x87:'SPC-180N-USB',
    0x88:'SPC-130IN',   0x89:'SPC-130INX', 0x8a:'SPC-130INXX',
    0x8b:'SPC-QC-104',  0x8c:'SPC-QC-004', 0x8d:'SPC-QC-106',
    0x8e:'SPC-QC-006'
                  }

#: map of each card name in _spc_codes to a name indicating how .spc file is structured (see B&H file SPC_data_file_structure.h)
_spc_types:dict = {
    'SPC-130':'SPC-1XX', 'SPC-600':'SPC-6XX', 'SPC-630':'SPC-6XX',
    'SPC-700':'NotSup', 'SPC-730':'NotSup', 'SPC-830':'SPC-1XX',
    'SPC-140':'SPC-140', 'SPC-930':'NotSup', 'SPC-150':'SPC-1XX',
    'DPC-230':'DPC-230', 'SPC-130EM':'SPC-1XX', 'SPC-160':'SPC-1XX',
    'SPC-150N':'SPC-1XX', 'SPC-150NX':'SPC-1XX', 'SPC-160X':'SPC-1XX',
    'SPC-160PCIE':'SPC-1XX', 'SPC-130EMN':'SPC-1XX', 'SPC-180N':'SPC-1XX',
    'SPC-180NX':'SPC-1XX',   'SPC-180NXX':'SPC-1XX', 'SPC-180N-USB':'SPC-1XX',
    'SPC-130IN':'SPC-1XX',   'SPC-130INX':'SPC-1XX', 'SPC-130INXX':'SPC-1XX',
    'SPC-QC-104':'QC-X04',  'SPC-QC-004':'QC-X04', 'SPC-QC-106':'QC-X06',
    'SPC-QC-006':'QC-X06'
                  }

def _mask_shift(arr:np.ndarray, mask:int, shift:int, dtype:np.dtype)->np.ndarray:
    """Treat range of bits as number, useful for getting number from non-byte alligned binary fields"""
    return np.right_shift(np.bitwise_and(arr, mask), shift).astype(dtype)


def _advance_to(file, delim:bytes, out:bytes=None)->bytes:
    """
    Advance file buffer to delim. returns bytes after delimiter up to the next newline character
    If out is specified, appends this data to out
    """
    if out is None:
        out = bytes()
    while line := file.readline(): # advance to end of file if never find delim
        if delim in line: # check if delim reached
            out += line.split(delim, 1)[1] # append read data to out
            break
    return out


def _collect_to(file, delim:bytes, out:bytes=None)->bytes:
    """Return all bytes in file up to delim, if out is specified, appends bytes to out"""
    if out is None:
        out = bytes()
    while newline := file.readline():
        if delim in newline:
            out += newline.split(delim)[0]
            break
        out += newline
    return out


def _collect_between(file, start:bytes, stop:bytes, out:bytes=None)->bytes:
    """Return bytes between start and stop delimiters in file, if out is specified, appends this data to out"""
    if out is None:
        out = bytes()
    out = _advance_to(file, start, out)
    out = _collect_to(file, stop, out)
    return out


def _make_bh_ident_dictionary(data:bytes)->dict:
    """take bytes of the IDENTITY section of .set file and return dictionary of strings"""
    out = dict()
    key, val = None, None
    for line in data.decode("utf8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # if line contains a ':' then update dictionary with key, value pair
        if ":" in line:
            out[key] = val # add previous value to dictionary
            key, val = [l.strip() for l in line.split(":", 1)] # get current key value pair
        else: # no : means append line to previous
            val += '\n' + line
    out.pop(None)
    return out


def _make_bh_setup_dictionary(data:bytes)->dict:
    """take bytes of SETUP section of .set file and return dictionary of strings"""
    start = False
    sys_params  = {}
    for line in data.split(b"\n"):
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


def load_set(setfile:FileName)->dict:
    """
    Read set file into dictionary.

    Parameters
    ----------
    setfile : str, bytes, os.PathLike
        Path to set file of data.

    Returns
    -------
    meta : dict
        A dictionary of the metadata with 3 keys:
            - 'header': the header of the set file, represented as single element array with custom numpy.dtype
            - 'identification': a dictionary of the key:value pairs represented in the IDENTIFICATION section of the set file
            - 'setup': a dictionary of the key:value pairs represented in the SETUP section of the set file (formerly sys_params)
    """
    with open(setfile, 'rb') as f:
        header = np.fromfile(f, dtype=_sethead_dtype, count=1)
        ident_bytes = _collect_between(f, b'*IDENTIFICATION', b'*END')
        ident_dict = _make_bh_ident_dictionary(ident_bytes)
        setup_bytes = _collect_between(f, b'*SETUP', b'*END')
        setup_dict = _make_bh_setup_dictionary(setup_bytes)
    return dict(header=header, identification=ident_dict, setup=setup_dict)


def _read_32_nonQC_records(records:np.ndarray[np.uint32], 
                           timestamps_bitmask:int=0x00000FFF, timestamps_shift:int=0,
                           routing_bitmask:int=0x0000F000, routing_shift:int=12,
                           nanotimes_bitmask:int=0x0FFF0000, nanotimes_shift:int=16
                          )->tuple[np.ndarray[np.uint64],np.ndarray[np.uint8],np.ndarray[np.uint16],np.ndarray[np.uint8]]:
    """
    Reads 32 bit records that use the timestamp:routing:nanotime specification.
    NOTE: cannot read QC records, as these contain additional CH fields, and use
    diferent specification for markers and overflows.
    Specify field locations with bitmask and shift arguments
    """
    timestamps = _mask_shift(records, timestamps_bitmask, timestamps_shift, 'u8')
    ovfl       = _mask_shift(records, 0xC0000000, 30, 'u1')
    # recompute times for overflows w/ invalid photons
    timestamps += np.left_shift(np.cumsum((ovfl == 0b11)*np.bitwise_and(records, 0x0FFFFFFF)), 12)
    # remove invalid photons
    valid = ~(np.bitwise_and(ovfl, 0b10).astype(np.bool_))
    timestamps = timestamps[valid]
    records = records[valid]
    ovfl = ovfl[valid]
    del valid # save memory
    # compute single time overflows
    timestamps += np.left_shift(np.cumsum(ovfl, dtype='u8'), 12)
    del ovfl # save memory
    # get routing (marker/detector) and nanotimes arrays
    # do this only after removing invalid from records to save memory
    routing = _mask_shift(records,   routing_bitmask,   routing_shift, np.uint8)
    adc     = _mask_shift(records, nanotimes_bitmask, nanotimes_shift, np.uint16)
    # process markers
    marker_shift = 2**int(np.max(routing)).bit_length()
    marker_mask = np.bitwise_and(records, 0x10000000).astype(np.bool_)
    routing[marker_mask] += marker_shift
    markers = np.unique(routing[marker_mask])
    return timestamps, routing, adc, markers


def _read_spc1xx_8xx(file, meta:dict=None)->tuple[np.ndarray[np.uint64],np.ndarray[np.uint8],np.ndarray[np.uint16],np.ndarray[np.uint8]]:
    """Read .spc file in the most common SPC-1XX/SPC-8XX format, note file must **alread be opened**"""
    header = np.fromfile(file, dtype='<u4', count=1)
    # process header bits for timestamps
    timestamps_unit = np.bitwise_and(header, 0x00FFFFFF)[0] * 0.1e-9
    # nroute          = np.bitwise_and(header, 0x78000000)[0] >> 27
    # get all records
    records    = np.fromfile(file, dtype='<u4')
    # get each of timestamps, routing nanotimes, and addtional metadata bits
    timestamps, routing, adc, marker_ids = _read_32_nonQC_records(records)
    return dict(timestamps=timestamps, detectors=routing, nanotimes=4095-adc, 
                marker_ids=marker_ids, timestamps_unit=timestamps_unit, tcspc_num_bins=4096)


def _read_spc6xx_32bit(file, meta:dict=None)->dict:
    """Read .spc file in the most older SPC-6XX 32 bit format (ie 8 bit ADC), note file must **alread be opened**"""
    header = np.fromfile(file, dtype='<u4', count=1)
    # process header bits for timestamps
    timestamps_unit = np.bitwise_and(header, 0x00FFFFFF)[0] * 0.1e-9
    # nroute          = np.bitwise_and(header, 0x78000000)[0] >> 27
    # get all records
    records    = np.fromfile(file, dtype='<u4')
    # get each of timestamps, routing nanotimes, and addtional metadata bits
    out = _read_32_nonQC_records(records, 
                                 timestamps_bitmask=0x01FFFF00, timestamps_shift=8,
                                 routing_bitmask=0x0E000000, routing_shift=25,
                                 nanotimes_bitmask=0x000000FF, nanotimes_shift=0
                                )
    return dict(timestamps=out[0], detectors=out[1], 
                nanotimes=4095-out[2], marker_ids=out[3], 
                timestamps_unit=timestamps_unit, tcspc_num_bins=256)


_spc48_dtype:np.dtype = np.dtype([('field0', '<u2'), ('b', '<u1'), ('c', '<u1'), ('a', '<u2')])

def _read_spc6xx_48bit(file, meta:dict=None)->dict:
    """Read .spc file in the most older SPC-6XX 32 bit format (ie 8 bit ADC), note file must **alread be opened**"""
    # decode header
    header = np.fromfile(file, dtype='<u2', count=3)
    timestamps_unit = header[1] * 0.1e-9
    # nroute = np.bitwise_and(header[0], 0x000F)  # unused
    # ...and then the remaining records containing the photon data
    data = np.fromfile(file, dtype=_spc48_dtype)

    nanotimes = 4095 - np.bitwise_and(data['field0'], 0x0FFF)
    detectors = data['c']

    # Build the macrotime (timestamps) using in-place operation for efficiency
    timestamps = data['b'].astype('i8')
    np.left_shift(timestamps, 16, out=timestamps)
    timestamps += data['a']

    # extract the 13-th bit from data['field0']
    overflow = np.bitwise_and(np.right_shift(data['field0'], 13), 1)
    overflow = np.cumsum(overflow, dtype='i8')

    # Add the overflow bits
    timestamps += np.left_shift(overflow, 24)
    return dict(timestamps=timestamps, detectors=detectors,
                nanotimes=nanotimes, 
                marker_ids=np.empty(0, dtype=detectors.dtype),
                timestamps_unit=timestamps_unit, tcspc_num_bins=4096)


def _read_spc6xx(file, meta:dict=None)->dict:
    """Read SPC file from SPC-6XX card, automatically determines if is in 32 or 48 bit format"""
    header = np.fromfile(file, dtype='<u2', count=3)[2]
    file.seek(0, 0) # return file to 0 position
    rev = meta.get('identification',dict()).get('Revision', '')
    if header: # must be 32 bit
        if '12' in rev:
            warnings.warn(".spc file must be 8 bit ADC, but .set file indicates 12 bin adc")
        return _read_spc6xx_32bit(file)
    if '12' in rev:
        return _read_spc6xx_48bit(file)
    if '8' in rev:
        return _read_spc6xx_32bit(file)
    warnings.warn("file lacks definition for 8 or 12 bit ADC, assuming 48 bit")
    return _read_spc6xx_48bit(file)
        

def _read_QC_header(file, meta:dict=None)->tuple[float, int]:
    """
    Read timestamps_unit and number of routing channels from QC header of .spc file (header is same for QC-X04/X06
    NOTE: 
        QC header contains additional information, but none is relevant to photon-hdf5, or can be infered elsewhere, 
        may reconsider reading additional fields in the future.
    """
    header = np.fromfile(file, dtype='<u4', count=1)
    # process header bits for timestamps
    timestamps_unit = np.bitwise_and(header, 0x003FFFFF)[0] * 1e-15
    nroute          = np.bitwise_and(header, 0x78000000)[0] >> 27
    return timestamps_unit, nroute
    

def _read_QCX04(file, meta:dict=None)->dict:
    """Read QC-X04 style .spc file"""
    timestamps_unit, nroute = _read_QC_header(file)
    records = np.fromfile(file, dtype='<u4')
    # read final 2 bits, which indicate the record type as photon, marker, macrotime overflow, or GAP
    phtype = _mask_shift(records, 0xC0000000, 30, np.uint8)
    # read timestamps
    timestamps = np.bitwise_and(records, 0x00000FFF).astype('u8')
    # adjust timestamps for overflows
    ovfl_mask = phtype == 0b10
    timestamps += np.left_shift(np.cumsum(ovfl_mask, dtype=np.uint64), 12)
    ovfl_mask = ~ovfl_mask
    records = records[ovfl_mask]
    timestamps = timestamps[ovfl_mask]
    phtype = phtype[ovfl_mask]
    del ovfl_mask # save memory
    # read nanotimes
    nanotimes = _mask_shift(records, 0x0FFF0000, 16, 'u4')
    # combine CH bits and routing bits for detectors channel
    route_ch = _mask_shift(records, 0x30000000, 28, 'u1')
    route_ch += _mask_shift(records, 0x0000F000, 10, 'u1')
    # get markers
    marker_mask = phtype == 0b01
    # set detectors to shift according to marker
    route_ch[marker_mask] += 1<<int(np.max(route_ch)).bit_length()
    marker_ids = np.unique(route_ch[marker_mask])
    return dict(timestamps=timestamps, detectors=route_ch, 
                nanotimes=nanotimes, marker_ids=marker_ids, 
                timestamps_unit=timestamps_unit, tcspc_num_bins=4096)


def _read_QCX06(file, meta:dict=None)->dict:
    """Read QC-X04 style .spc file"""
    timestamps_unit, nroute = _read_QC_header(file)
    records = np.fromfile(file, dtype='<u4')
    # read final 4 bits, indicates record type, with some complexities...
    phtype = _mask_shift(records, 0xF0000000, 28, 'u1')
    # read timestamps
    timestamps = np.bitwise_and(records, 0x00000FFF).astype('u8')
    # adjust for macrotime overflows
    overflow_mask = phtype == 0b1000
    timestamps += np.left_shift(np.cumsum(overflow_mask, dtype='u8'), 12)
    # remove overflow photons from records
    overflow_mask = ~overflow_mask
    records = records[overflow_mask]
    timestamps = timestamps[overflow_mask]
    phtype = phtype[overflow_mask]
    del overflow_mask # save memory
    # start building router array
    route_ch = _mask_shift(records, 0x0000F000, 9, 'u1')
    route_ch[np.bitwise_and(phtype, 0b1000)==0b0000] += np.bitwise_and(phtype, 0b0111) # normal photon
    route_ch[np.bitwise_and(phtype, 0b1100)==0b1100] += np.bitwise_and(phtype, 0b0011) # GAP photon
    # identify and shift marker photons
    marker_mask = phtype == 0b1010
    route_ch[marker_mask] += 2**int(np.max(route_ch)).bit_length()
    marker_ids = np.unique(route_ch[marker_mask])
    # get nanotimes
    nanotimes = _mask_shift(records, 0x0FFF0000, 16, 'u4')
    return dict(timestamps=timestamps, detectors=route_ch, 
                nanotimes=nanotimes, marker_ids=marker_ids, 
                timestamps_unit=timestamps_unit)

#: map of SPC-type to reader function
_spc_reader_funcs:dict = {
    'SPC-1XX':_read_spc1xx_8xx, 
    'QC-X04':_read_QCX04, 'QC-X06':_read_QCX06,
    'SPC-6XX':_read_spc6xx,
    'SPC-6XX-48bit':_read_spc6xx_48bit,
    'SPC-6XX-32bit':_read_spc6xx_32bit
                         }

def _get_reader_func(meta:dict, SPC_type:str)->Callable:
    """Read the metadata to determine which """
    spc_code = (meta['header'][0]['revision']&0x0FF0)>>4
    spc_type = _spc_types.get(_spc_codes.get(spc_code, None), None)
    if SPC_type != 'auto': # user specified SPC_type, indicating new, unsupported card
        if SPC_type not in _spc_reader_funcs:
            raise TypeError(f"invalid SPC_type, must be in {list(_spc_reader_funcs.keys())}")
        if spc_type is not None and SPC_type not in spc_type: # double check if phconvert DOES recognize the card
            warnings.warn("SPC_type indicates different SPC-card than indicated in header")
        return _spc_reader_funcs[SPC_type]
    if spc_type is None:
        raise NotImplementedError(f"unrecognzied revision (SPC card), {hex(spc_code)}")
    return _spc_reader_funcs[spc_type]
        

def load_spc(spcfile:Union[FileName, None]=None, setfile:Union[FileName, None]=None,
             SPC_type:str='auto')->dict:
    """
    Read data from .spc/.set file pair. Must specify either spcfile or setfile, and
    function will infer the name of the other.

    If you are using an SPC-card that is newer than the current publication of phconvert,
    you will need to specify SPC_type, which specifies the format of the data in the .spc file.
    Options are:
    
    - 'SPC-1XX' for SPC-1XX and SPC-8XX cards (this is the most common format)
    - 'SPC-6XX' for SPC-6XX cards, and will infer if saved in 32 or 48 bit formats
    - 'SPC-6XX-48bit' for SPC-6XX cards with 12 bit TCSPC resolution (most common for these cards)
    - 'SPC-6XX-32bit'for SPC-6XX cards with 8 bit TCSPC resolution (most common for these cards)
    - 'QC-X04' for SPC-QC-X04 cards 
    - 'QC-X06'for SPC-QC-X06 cards

    Parameters
    ----------
    spcfile : str, bytes, os.PathLike, conditionally optional
        Path to the .spc file, must specify if setfile is not specified
    setfile : str, bytes, os.PathLike, conditionally optional
        Path to the .set file, must specify if spcfile is not specified
    SPC_type : str
        Used if your card is newer than phconvert, otherwise leave as 'auto'
        See above the options for cards.
    
    Returns
    -------
    data : dict
        Dictionary with 2 keys containing the following data

        - 'meta' the metadata from the .set file, with header, identity and setup 
          sub-keys (see :func:`load_set` for more details)
        - 'photon_data' dictionary containing arrays of photon data for 
          timestamps, detectors, and nanotimes, as well as an array of
          marker_ids indicating the id of detectors that are markers, and
          timestamps_unit, a float indicating the clock rate of timestamps
    """
    # checking file types,
    if spcfile is None and setfile is None:
        raise TypeError("must specify at least spcfile or setfile")
    # cast spcfile and setfile to Path objects
    spcfile = Path(spcfile) if spcfile is not None else spcfile
    setfile = Path(spcfile) if setfile is not None else setfile
    # Auto-fill non-specified file
    if setfile is None:
        setfile = spcfile.with_suffix('.set')
        if not setfile.is_file():
            setfile = setfile.with_suffix('.SET')
    if spcfile is None:
        spcfile = setfile.with_suffix('.spc')
        if not spcfile.is_file():
            spcfile = spcfile.with_suffix('.SET')
    # read metadata from setfile, used to determine which SPC board was used
    meta = load_set(setfile)
    # retrieve function for reading spc file based on SPC card in 
    with open(spcfile, 'rb') as file:
        ph_data = _get_reader_func(meta, SPC_type)(file, meta=meta)
    return dict(meta=meta, photon_data=ph_data)

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