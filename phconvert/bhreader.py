"""
.. currentmodule:: phconvert

Using bhreader module
---------------------

.. note::
    
    It is rarely necessary to call ``bhreader`` directly, instead call
    :func:`loader.loadfile_bh`

This module contains functions to load and decode files from Becker & Hickl (B&H) 
hardware.

.. currentmodule::phconvert.bhreader

The main, highest level function of the module is :func:`load_spc`.
This function will automatically detect the type of card, and read the files
appropriately.

:func:`load_spc` returns a dictionary of with 2 keys: 'meta', and 'photon_data':
    - 'meta': the dictionary loaded by :func:`load_set`
    - 'photon_data': dictionary of the data loaded from the .spc file, all values
      are already converted into form acceptable for photon-hdf5
        - 'timestamps': numpy array of timestamps of all photons/markers
        - 'detectors': numpy array of the detector of each photon/marker 
          (markers already reassigned unique index)
        - 'nanotimes': nanotime of each photon (ADC automatically converted to nanotime)
        - 'timestamps_unit' : the timestamps unit of macrotimes
        - 'marker_ids' : array of unuiqe indexes cooresponding to detector indexes
          that should be read as markers


:func`load_spc` loads just the .set file which contains metadata on the .spc file.
It returns a dictionary with the following keys:
    - 'identification': a dictionary of key value pairs that are both strings
      equivalent to the IDENTIFICATION portiion of the .set file
    - 'setup': a dictionary of key (string) value (any) pairs contained in the
      SETUP section of the .set file. This contains metadata on the acqusition
      settings
    - 'header': 1 element numpy array with a dtype representing the .set header
      structure (see following explanation) this most importantly contains
      a code indicating the SPC card used (see `bhformats`_ )

Code examples
*************

:func:`load_spc` is flexible, assuming the .set and .spc files have the same
name other than the extension, it is easiest, and best to specify the .spc file 

.. code-block::
    
    import phconvert as phc
    data = phc.bhreader.load_spc('sample.spc')
    
however, the result will be the same as

.. code-block::
    
    data = phc.bhreader.load_spc(setfile='sample.set')

If for some reason, the .set and .spc files have different names, it is necessary
to specify both

.. code-block::
    
    data = phc.bhreader.load_spc(spcfile='sample.spc', setfile='Sample1.set')
    
.. warning::
    
    SPCM sofware never saves the .set and .spc files with different names.
    Therefore, this situation indicates that the files were manually renamed.
    Renaming the two files differently is therefore *highly* inadvisable.


Future proofing
***************

With the latest version of phconvert, detection of the spc card and therefore
record format has become automatic.

This does come with a downside: if B&H comes out with a new card, it will be
unrecognizable by phconvert versions published before the new card.

However, in anticipation of this, a work-around is provided.
(As of writing (July 20205), there should be no cards for which this is necessary)

Assuming B&H has not come up with another format (i.e. your card is new, but uses
an existing record format), :func:`load_spc` can be instructed to read the .spc
file according the the specified format.

This is done by specifying the '``SPC_type``' argument of :func:`load_spc` as a
string, see the first line of the subsections of `bhformats`_ to get the argument
to provide.

For example

.. code-block::
    
    data = phc.bhreader.load_spc('sample.spc', SPC_type='SPC-1XX')


.. _bhformats:

Becker & Hickl SPC formats
--------------------------

Becker & Hickl documentation for their file format is scattered across the
`TCSPC handbook <https://www.becker-hickl.com/literature/documents/flim/the-bh-tcspc-handbook/>`_
and in the ``SPC_data_file_structure.h`` file which can be located in the SPCM sofware
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

SPC-1XX and SPC-8XX format
**************************

:func:`load_set` can be forced to read the .spc in this format by setting
``SPC_type='SPC-1XX'``

The majority of modern B&H cards use this format, whic relies on 32 bit (4 byte)
records.
The structure of these records is as follows

+------+---------+--------------------+------+--------+------------+-----------+------------+
| Bit  | 31      | 30                 | 29   | 28     | 27-16      | 15-12     | 11-0       |
+------+---------+--------------------+------+--------+------------+-----------+------------+
| name | Invalid | Macrotime overflow | Gap  | marker | ADC        | Routing   | Macrotime  |
+------+---------+--------------------+------+--------+------------+-----------+------------+
| type | bool    | bool               | bool | bool   | int 12 bit | int 4 bit | int 12 bit |
+------+---------+--------------------+------+--------+------------+-----------+------------+

Where:
    - invalid: boolean indicating if record can be considered a photon, if 1, then
      the record either arose from an error, or if macrotime overflow is also 1, then 
      exclusively as a macrotime overflow (see below)
    - Macrotime overflow: since only 12 bits are allocated to the macrotime
      every :math:`2^{12} = 4096` macrotime unit, it is necessar to add an overflow
      event, which adds an additional 4096 to the value of the macrotime.
      So, for any given record, the number of overflow events in the records up to
      the given record must be summed, and 
      :math:`4096 * no.\:of\:overflows` must be added to the
      macrotime clock
    - Gap: this indicates that the file buffer overflowed, and therefore there may
      be photons missed in the recording. This is rare and indicates potentially
      corupted data
    - marker indicates if the record is a marker photon, ie for FLIM, if the bit is
      1, then the record must be read differently  (see below)
    - 0: this bit is always zero and serve no purpose
    - ADC: this represents the nanotime, but as the cards record in stop-start mode
      the nanotime = 4096 - ADC
    - Routing: the index of the detector from the router, ie this indicates which
      detector the photon arrived at
    - Macrotime: the time since the last overflow event, in macrotime units

The first record in any .spc file follows the following format

+------+----+---------+------+---------+----------+----------------+
| Bit  | 31 | 30-78   | 26   | 25      | 24       | 23-0           |
+------+----+---------+------+---------+----------+----------------+
| name | 1  | unused  | raw  | markers | reserved | macrotime unit |
+------+----+---------+------+---------+----------+----------------+
| type | -- | --      | bool | bool    | --       | int  24 bit    |
+------+----+---------+------+---------+----------+----------------+

- macrotime unit is in 0.1 nanoseconds
- raw indicates if data was recorded in diatnostic mode
- markers indicated if markers are used in the photon stream

Marker records (bit 28 = 1) change certain field

- routing becomes the marker type
- ADC is irrelevant, these values are ignored
- bit 31 (Invalid) is set to 1


SPC-QC-Formats
***************
+-----+----+------------------------+-----+---------+--------+-----------+--------+----------------+
| Bit | 31 | 30-27                  | 26  | 25      | 24     | 23        | 22     | 21-0           |
+-----+----+------------------------+-----+---------+--------+-----------+--------+----------------+
|     | 1  | number of routing bits | raw | markers | femto  | 6 channel | unused | macrotime unit |
+-----+----+------------------------+-----+---------+--------+-----------+--------+----------------+


SPC-QC-X06/X08 format
~~~~~~~~~~~~~~~~~~~~~

:func:`load_spc` can be forced to read the .spc file in this format by specifying
``SPC_type='SPC-QC-X04'``

QC-004 type cards store the data in the following 32-bit record format

Photon records use the following format

+------+--------------+-----------+------------+-----------+------------+
| Bit  | 31 -30       | 29-28     | 27-16      | 15-12     | 11-0       |
+------+--------------+-----------+------------+-----------+------------+
| name |  record type | channel   | nanotime   | routing   | macrotime  |
+------+--------------+-----------+------------+-----------+------------+
| type | int 2 bit    | int 2 bit | int 12 bit | int 4 bit | int 12 bit |
+------+--------------+-----------+------------+-----------+------------+

The record codes are the following

- bit 31 = 0, bit 30 = 0: normal photon record
- bit 31 = 1, bit 30 = 0: macrotime overflow, all other bits 0 by definition
- bit 31 - 0, bit 30 = 1: marker, channel and nanotime bits all 0 by definition
- bit 31 = 1, bit 30 = 1: Gap, which occurs immediately before a file in file out
  overflow, meaning potential missing data, read remaining bytes as normal photon
  
For normal photon records

- channel indicates which input channel the event occured with
- routing indicates the routing signal for the photon
- To determine detector, both channel and routing must be considered
- unlike most other B&H record formats, the nanotime is not inverted, ie 0x000 = 0 ns

For marker records:
    - routing number indicates the marker type

The first "header" records is in the following format


SPC-QC-X06/X08 format
~~~~~~~~~~~~~~~~~~~~~

:func:`load_spc` can be forced to read the .spc file in this format by specifying
``SPC_type='SPC-QC-X06'``

QC-X0(>4) cards use a 32-bit record format similar to that of QC-X04 cards, but
with some key changes due to having more channels than can be indexed with the 2
bits allocated in the QC-0X04 format.

+------+------------+----------------+------------+-----------+------------+
| Bit  | 31         | 30-28          | 27-16      | 15-12     | 11-0       |
+------+------------+----------------+------------+-----------+------------+
| name | Special    | record/channel | nanotime   | routing   | macrotime  |
+------+------------+----------------+------------+-----------+------------+
| type | bool       | int 3 bit      | int 12 bit | int 4 bit | int 12 bit |
+------+------------+----------------+------------+-----------+------------+

As in QC-004, channel is the physical channel, and routing the signal from the
router. Therefore to determine the detctor, both must be taken into account.

If Special (bit 31) = 1, then the bits 30-2 are used to determine the non-photon
type of record.


| Bit                | 31 | 30 | 29 | 28 | 27-16    | 15-12   | 11-0      |
+====================+====+====+====+====+==========+=========+===========+
| photon             | 0  | ch[2:0]      | nanotime | routing | macrotime |
+--------------------+----+--------------+----------+---------+-----------+
| macrotime overflow | 1  | 0  | 0  | 0  | 0x000    | 0x0     | 0x000     |
+--------------------+----+--------------+----------+---------+-----------+
| marker             | 1  | 0  | 1  | 0  | 0x000    | marker  | macrotime |
+--------------------+----+--------------+----------+---------+-----------+
| Gap                | 1  | 1  | ch[1:0] | nanotime | routing | macrotime |
+--------------------+----+--------------+----------+---------+-----------+

The first record follows the same format as QC-X04 cards

SPC-6XX formats
***************

Older SPC-6XX cards can use either a 48 bit or 32 bit size records.
Since there are 2 formats for these cards, the format must be infered.
This is normally done automatically, based on the initial bits of .spc file.

:func:`load_spc` can be forced to ignore the .set file, and perform the inference
as an SPC-6XX card by specifying ``SPC_type='SPC-6XX'``


The 48 bit format has the following bit assigments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`load_spc` can be forced to read .spc files in this fomat by specifying
``SPC_type='SPC-6XX-48bit'``

+------+------------------+-----------+-------------------+----+------+--------------------+---------+------------+
| Bit  | 47-32            | 31-24     | 23-16             | 15 | 14   | 13                 | 12      | 11-0       |
+------+------------------+-----------+-------------------+----+------+--------------------+---------+------------+
| name | macrotime[15:0]  | routing   | marcrotime[23:12] | 0  | Gap  | macrotime overflow | Invalid | ADC        |
+------+------------------+-----------+-------------------+----+------+--------------------+---------+------------+
| type | int 24 bit[15:0] | int 8 bit | int 24 bit[23:12] | -- | bool | bool               | bool    | int 12 bit |
+------+------------------+-----------+-------------------+----+------+--------------------+---------+------------+


The 32 bit fomat has the following bit assigments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`load_spc` can be forced to read .spc files in this fomat by specifying
``SPC_type='SPC-6XX-48bit'``

+------+---------+--------------------+------+----+-----------+------------+-----------+
| Bit  | 31      | 30                 | 29   | 28 | 27        | 26-8       | 7-0       |
+------+---------+--------------------+------+----+-----------+------------+-----------+
| name | Invalid | macrotime overflow | GAP  | 0  | routing   | macrotime  | ADC       |
+------+---------+--------------------+------+----+-----------+------------+-----------+
| type | bool    | bool               | bool | -- | int 3 bit | int 17 bit | int 8 bit |
+------+---------+--------------------+------+----+-----------+------------+-----------+

Where

- invalid: boolean indicating if record can be considered a photon, if 1, then
  the record either arose from an error, or if macrotime overflow is also 1, then 
  exclusively as a macrotime overflow (see below)
- Routing: the index of the detector from the router, ie this indicates which
  detector the photon arrived at
- Gap: this indicates that the file buffer overflowed, and therefore there may
  be photons missed in the recording. This is rare and indicates potentially
  corupted data
- Macrotime overflow: due to the limited number of bit available to the macrotime,
  every :math:`2^{num\:bits}` macrotime unit, it is necessar to add an overflow
  event, which adds an additional :math:`2^{num\:bits}` to the value of the macrotime.
  So, for any given record, the number of overflow events in the records up to
  the given record must be summed, and 
  :math:`2^{num\:bits} * no.\:of\:overflows` must be added to the
  macrotime clock
- Macrotime: the time since the last overflow event, in macrotime units
- ADC: this represents the nanotime, but as the cards record in stop-start mode
  the nanotime = :math:`max(ADC) - ADC`
  
.. note::
    
    For the 48 bit format, the macrotime is stored in 2 blocks, which must be 
    combined, the 47-32 block stores the smaller  values, but otherwise all 
    integer values stored in little endian format

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
    timestamps = _mask_shift(records, timestamps_bitmask, timestamps_shift, '<i8')
    ovfl       = _mask_shift(records, 0xC0000000, 30, '<u1')
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
    routing = _mask_shift(records,   routing_bitmask,   routing_shift, '<u1')
    adc     = _mask_shift(records, nanotimes_bitmask, nanotimes_shift, '<u2')
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
    timestamps = data['b'].astype('<i8')
    np.left_shift(timestamps, 16, out=timestamps)
    timestamps += data['a']

    # extract the 13-th bit from data['field0']
    overflow = np.bitwise_and(np.right_shift(data['field0'], 13), 1)
    overflow = np.cumsum(overflow, dtype='<i8')

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
    timestamps = np.bitwise_and(records, 0x00000FFF).astype('<i8')
    # adjust timestamps for overflows
    ovfl_mask = phtype == 0b10
    timestamps += np.left_shift(np.cumsum(ovfl_mask, dtype='<i8'), 12)
    ovfl_mask = ~ovfl_mask
    records = records[ovfl_mask]
    timestamps = timestamps[ovfl_mask]
    phtype = phtype[ovfl_mask]
    del ovfl_mask # save memory
    # read nanotimes
    nanotimes = _mask_shift(records, 0x0FFF0000, 16, 'u2')
    # combine CH bits and routing bits for detectors channel
    route_ch = _mask_shift(records, 0x30000000, 28, '<u1')
    route_ch += _mask_shift(records, 0x0000F000, 10, '<u1')
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
    phtype = _mask_shift(records, 0xF0000000, 28, '<u1')
    # read timestamps
    timestamps = np.bitwise_and(records, 0x00000FFF).astype('<i8')
    # adjust for macrotime overflows
    overflow_mask = phtype == 0b1000
    timestamps += np.left_shift(np.cumsum(overflow_mask, dtype='<i8'), 12)
    # remove overflow photons from records
    overflow_mask = ~overflow_mask
    records = records[overflow_mask]
    timestamps = timestamps[overflow_mask]
    phtype = phtype[overflow_mask]
    del overflow_mask # save memory
    # start building router array
    route_ch = _mask_shift(records, 0x0000F000, 9, '<u1')
    route_ch[np.bitwise_and(phtype, 0b1000)==0b0000] += np.bitwise_and(phtype, 0b0111) # normal photon
    route_ch[np.bitwise_and(phtype, 0b1100)==0b1100] += np.bitwise_and(phtype, 0b0011) # GAP photon
    # identify and shift marker photons
    marker_mask = phtype == 0b1010
    route_ch[marker_mask] += 2**int(np.max(route_ch)).bit_length()
    marker_ids = np.unique(route_ch[marker_mask])
    # get nanotimes
    nanotimes = _mask_shift(records, 0x0FFF0000, 16, '<u2')
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