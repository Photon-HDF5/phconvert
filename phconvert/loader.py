#
# phconvert - Reference library to read and save Photon-HDF5 files
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains functions to load each supported data format.
Each loader function loads data from a third-party formats into a python
dictionary which has the structure of a Photon-HDF5 file.
These dictionaries can be passed to :func:`phconvert.hdf5.save_photon_hdf5`
to save the data in Photon-HDF5 format.

The loader module contains high-level functions which "fill" the dictionary
with the appropriate arrays. The actual decoding of the input binary files
is performed by low-level functions in other modules
(smreader.py, pqreader.py, bhreader.py). When trying to
decode a new file format, these modules can provide useful examples.

"""

import os
import warnings
import time
import numpy as np

from . import smreader
from . import bhreader
from . import pqreader


def loadfile_sm(filename:str, software:str='LabVIEW Data Acquisition usALEX', 
                       warn:bool=False, print_warning:bool=True)->dict:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'file: {filename} does not exist')
    timestamps, detectors, labels = smreader.load_sm(filename, 
                                                     return_labels=True)
    unique_detectors = np.unique(detectors)
    if len(labels) < unique_detectors.size:
        if warn:
            warnings.warn("Number of labels and detector types inconsistent, check labels")
        if print_warning:
            print("Number of labels and detector types inconsistent, check labels")

    photon_data = dict(
        timestamps=timestamps, 
        timestamps_specs = dict(timestamps_unit=12.5e-9),
        detectors=detectors,
        measurement_specs = dict(
            measurement_type = None,
            detectors_specs = dict(spectral_polarization_split_chN=np.unique(detectors)))
        )
    setup = dict(
        num_pixels = unique_detectors.size,
        num_spots = 1,
        num_spectral_ch = None,
        num_polarization_ch = None,
        num_split_ch = None,
        modulated_excitation = None,
        lifetime = False,
        excitation_wavelengths = None,
        excitation_cw = None,
        detection_wavelengths = None,
        excitation_alternated=None,
        )
    if len(labels) == unique_detectors.size:
        setup['detectors'] = {'id':unique_detectors, 
                              'label':np.array([''.join(chr(c) for c in chan) 
                                                for chan in  labels])}
    provenance = dict(filename=filename, software=software)
    acquisition_duration = (timestamps[-1] - timestamps[0]) * 12.5e-9
    identity = dict(author=None, author_affiliation=None)
    data = dict(
        _filename = filename,
        description = None,
        acquisition_duration = round(acquisition_duration),
        photon_data = photon_data,
        setup = setup,
        provenance = provenance,
        identity = identity)
    return data


def _load_spc_infer(filename:str, metadata:dict)->tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    identity = metadata['identification']
    spc_model = ''
    # initial inference
    for module in ('134', '144', '154', '830', '630', '600'):
        if module in identity['Contents']:
            spc_model = f'SPC-{module}'
            break
    if spc_model == '':
        spc_model = 'SPC-630' if '12' in identity['Revision'] else 'SPC-151'
    timestamps, detectors, nanotimes, timestamps_unit = bhreader.load_spc(filename, 
                                                                          spc_model=spc_model)
    if np.any(np.diff(timestamps) < 0):
        spc_model = 'SPC-151' if '6' in spc_model else 'SPC-630'
        timestampsn, detectorsn, nanotimesn, timestamps_unitn = bhreader.load_spc(filename, 
                                                                              spc_model=spc_model)
        if (np.diff(timestampsn) < 0).sum() < (np.diff(timestamps) < 0).sum():
            timestamps, detectors, nanotimes, timestamps_unit = timestampsn, detectorsn, nanotimesn, timestamps_unitn
    return timestamps, detectors, nanotimes, timestamps_unit


def loadfile_bh(filename:str, setfilename:str=None, spc_model:str='infer')->tuple[dict, dict]:
    software = 'Becker & Hickl SPCM'
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File: {filename} does not exist')
    if setfilename is None:
        setfilename = filename[:-3] + 'set'
        if not os.path.isfile(setfilename):
            setfilename = filename[:-3] + 'SET'
            if not os.path.isfile(setfilename):
                raise FileNotFoundError(f'Set file for {filename} could not be '
                                        'located with .set or .SET extension '
                                        'manually specify setfile=... if set '
                                        'file has different extension')
    elif not os.path.isfile(setfilename):
        raise FileNotFoundError(f'Set file {setfilename} does not exist')
    metadata = bhreader.load_set(setfilename)
    if spc_model == 'infer':
        timestamps, detectors, nanotimes, timestamps_unit = _load_spc_infer(filename, metadata)
    else:
        timestamps, detectors, nanotimes, timestamps_unit = bhreader.load_spc(filename, 
                                                                            spc_model=spc_model)
    det_ids, det_counts = np.unique(detectors, return_counts=True)
    provenance = dict(filename=filename, software=software)
    if 'identification' in metadata:
        identification = metadata['identification']
        date_str = identification['Date']
        time_str = identification['Time']
        creation_time = date_str + ' ' + time_str
        provenance.update({'creation_time': creation_time})
    if 'sys_params' in metadata:
        print('TCSPC parameters retrived from the .SET file.')
        sys_params = metadata['sys_params']
        tcspc_num_bins = int(sys_params['SP_ADC_RE'])
        tcspc_unit = float(sys_params['SP_TAC_TC'])/float(sys_params['SP_TAC_G'])
        #tcspc_range = sys_params['SP_TAC_R']  # redundant info, not corrected for gain
        tcspc_range = tcspc_num_bins * tcspc_unit
    else:
        tcspc_num_bins = None
        tcspc_unit = None
        tcspc_range = None

    photon_data = dict(
        timestamps=timestamps, 
        timestamps_specs=dict(timestamps_unit=timestamps_unit),
        detectors=detectors,
        nanotimes = nanotimes,
        measurement_specs = dict(
            measurement_type = None,
            laser_repetition_rate = np.array([sys_params['SP_TAC_R']]),
            detectors_specs = dict(spectral_polarization_split_chN=np.unique(detectors),
                                   )),
        nanotimes_specs = dict(
            tcspc_unit = tcspc_unit,
            tcspc_range = tcspc_range,
            tcspc_num_bins = tcspc_num_bins),
        )

    setup = dict(
        num_pixels = np.unique(detectors).size,
        num_spots = 1,
        num_spectral_ch = None,
        num_polarization_ch = None,
        num_split_ch = None,
        modulated_excitation = True,
        lifetime = True,
        excitation_wavelengths = None,
        excitation_cw = None,
        detection_wavelengths = None,
        excitation_alternated = None,
        detectors = dict(id=det_ids, counts=det_counts, label=None),
        laser_repetition_rates = np.array([sys_params['SP_TAC_R'], ])
        )

    acquisition_duration = ((timestamps.max() - timestamps.min()) *
                            timestamps_unit)
    identity = dict(author=None, author_affiliation=None)

    data = dict(
        _filename = filename,
        acquisition_duration = round(acquisition_duration),
        photon_data = photon_data,
        setup = setup,
        provenance = provenance,
        identity = identity,
        user=dict(becker_hickl=metadata))
    return data, metadata


def loadfile_ptu(filename:str)->tuple[dict, dict]:
    load_pq = {'ptu': pqreader.load_ptu, 'ht3': pqreader.load_ht3,
               'pt3': pqreader.load_pt3}[filename[-3:]]
    
    times, dets, dtime, metadata, marker_ids = load_pq(filename)
    # Creation time from the file header
    creation_time = metadata.pop('creation_time')
    
    software = metadata.pop('software')
    software_version = metadata.pop('software_version')
    
    provenance = dict(
        filename=filename,
        creation_time=creation_time,
        software=software,
        software_version=software_version,
    )
    
    timestamps_unit = float(metadata.pop('timestamps_unit'))
    acquisition_duration = float(metadata.pop('acquisition_duration'))
    
    photon_data = dict(
        timestamps=times,
        timestamps_specs=dict(timestamps_unit=timestamps_unit),
        detectors=dets,
        
        measurement_specs=dict(
            measurement_type=None,
            detectors_specs=dict(spectral_polarization_split_chN=
                                 np.setdiff1d(dets, marker_ids),
                                 )
            ),
    )
    det_ids, det_counts = np.unique(dets, return_counts=True)

    setup = dict(
        num_pixels = np.unique(dets).size - marker_ids.size,
        num_spots = 1,
        num_spectral_ch = None,
        num_polarization_ch = None,
        num_split_ch = None,
        modulated_excitation = True,
        lifetime = dtime is not None,
        excitation_wavelengths = None,
        excitation_cw = None,
        detection_wavelengths = None,
        excitation_alternated = None,
        detectors = dict(id=det_ids, counts=det_counts, label=None),
        )
    identity = dict(author=None, author_affiliation=None)

    if dtime is not None:
        laser_repetition_rate = float(metadata.pop('laser_repetition_rate'))
        tcspc_unit = float(metadata.pop('nanotimes_unit'))
        tcspc_num_bins = 4096
        tcspc_range = tcspc_num_bins * tcspc_unit
        photon_data['nanotimes'] = dtime
        photon_data['measurement_specs']['laser_repetition_rate'] = laser_repetition_rate,
        photon_data['nanotimes_specs'] = dict(
            tcspc_unit=tcspc_unit,
            tcspc_num_bins = tcspc_num_bins,
            tcspc_range = tcspc_range
            )
        setup['laser_repetition_rates'] = np.array([laser_repetition_rate, ])
    
    if marker_ids.size != 0:
        photon_data['measurement_specs']['detectors_specs']['markersN'] = marker_ids

    data = dict(
        _filename = filename,
        acquisition_duration = acquisition_duration,
        photon_data = photon_data,
        setup = setup,
        provenance = provenance,
        identity = identity,
        user=dict(picoquant=metadata))
    return data, metadata



def usalex_sm(
        filename, donor=0, acceptor=1, alex_period=4000, alex_offset=750,
        alex_period_donor=(2850, 580), alex_period_acceptor=(930, 2580),
        excitation_wavelengths=(532e-9, 635e-9),
        detection_wavelengths=(580e-9, 680e-9),
        software='LabVIEW Data Acquisition usALEX'):
    """Load a .sm us-ALEX file and returns a dictionary.

    This dictionary can be passed to the :func:`phconvert.hdf5.save_photon_hdf5`
    function to save the data in Photon-HDF5 format.
    """
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, labels = smreader.load_sm(filename,
                                                     return_labels=True)
    print(" [DONE]\n")

    photon_data = dict(
        timestamps = timestamps,
        timestamps_specs = dict(timestamps_unit=12.5e-9),
        detectors = detectors,
        measurement_specs = dict(
            measurement_type = 'smFRET-usALEX',
            alex_period = alex_period,
            alex_offset = alex_offset,
            alex_excitation_period1 = alex_period_donor,
            alex_excitation_period2 = alex_period_acceptor,
            detectors_specs = dict(spectral_ch1 = np.atleast_1d(donor),
                                   spectral_ch2 = np.atleast_1d(acceptor))),
    )

    setup = dict(
        num_pixels = 2,
        num_spots = 1,
        num_spectral_ch = 2,
        num_polarization_ch = 1,
        num_split_ch = 1,
        modulated_excitation = True,
        lifetime = False,
        excitation_wavelengths = excitation_wavelengths,
        excitation_cw = [True, True],
        detection_wavelengths = detection_wavelengths,
        excitation_alternated=[True, True])

    provenance = dict(filename=filename, software=software)
    acquisition_duration = (timestamps[-1] - timestamps[0]) * 12.5e-9
    data = dict(
        _filename = filename,
        acquisition_duration = round(acquisition_duration),
        photon_data = photon_data,
        setup = setup,
        provenance = provenance)

    return data


def nsalex_bh(filename_spc,
              donor = 4,
              acceptor = 6,
              alex_period_donor = (10, 1500),
              alex_period_acceptor = (2000, 3500),
              excitation_wavelengths = (532e-9, 635e-9),
              detection_wavelengths = (580e-9, 680e-9),
              allow_missing_set = False,
              tcspc_num_bins=None, tcspc_unit=None):
    """Load a .spc and (optionally) .set files for ns-ALEX and return 2 dict.

    The first dictionary can be passed to the
    :func:`phconvert.hdf5.save_photon_hdf5` function to save the data in
    Photon-HDF5 format.

    Returns:
        Two dictionaries: the first contains the main photon data (timestamps,
        detectors, nanotime, ...); the second contains the raw data from the
        .set file (it can be saved in a user group in Photon-HDF5).
    """
    software = 'Becker & Hickl SPCM'
    # Load .SPC file
    assert os.path.isfile(filename_spc), "File '%s' not found." % filename_spc
    print(" - Loading '%s' ... " % filename_spc)
    timestamps, detectors, nanotimes, timestamps_unit = \
        bhreader.load_spc(filename_spc)
    print(" [DONE]\n")

    # Load .SET file
    filename_set = filename_spc[:-3] + 'set'
    if os.path.isfile(filename_set):
        metadata = bhreader.load_set(filename_set)
    elif allow_missing_set:
        metadata = {}
        msg = 'SET file not found. You need to pass "%s".'
        assert tcspc_num_bins is not None, msg % 'tcspc_num_bins'
        assert tcspc_unit is not None, msg % 'tcspc_unit'
        print('SET file not found. Using passed TCSPC parameters.')
    else:
        raise IOError("File '%s' not found." % filename_set)

    # Extract the creation time from the .SET file metadata as it will be
    # more reliable than the creation time from the file system
    provenance = dict(filename=filename_spc, software=software)
    if 'identification' in metadata:
        identification = metadata['identification']
        date_str = identification['Date']
        time_str = identification['Time']
        creation_time = date_str + ' ' + time_str
        provenance.update({'creation_time': creation_time})

    if 'sys_params' in metadata:
        print('TCSPC parameters retrived from the .SET file.')
        sys_params = metadata['sys_params']
        tcspc_num_bins = int(sys_params['SP_ADC_RE'])
        tcspc_unit = float(sys_params['SP_TAC_TC'])
        #tcspc_range = sys_params['SP_TAC_R']  # redundant info
    tcspc_range = tcspc_num_bins * tcspc_unit

    photon_data = dict(
        timestamps = timestamps,
        timestamps_specs = dict(timestamps_unit=timestamps_unit),
        detectors = detectors,
        nanotimes = nanotimes,

        nanotimes_specs = dict(
            tcspc_unit = tcspc_unit,
            tcspc_range = tcspc_range,
            tcspc_num_bins = tcspc_num_bins),
        measurement_specs = dict(
            measurement_type = 'smFRET-nsALEX',
            laser_repetition_rate = 1 / timestamps_unit,
            alex_excitation_period1 = alex_period_donor,
            alex_excitation_period2 = alex_period_acceptor,
            detectors_specs = dict(spectral_ch1 = np.atleast_1d(donor),
                                   spectral_ch2 = np.atleast_1d(acceptor))),
    )

    setup = dict(
        num_pixels = 2,
        num_spots = 1,
        num_spectral_ch = 2,
        num_polarization_ch = 1,
        num_split_ch = 1,
        modulated_excitation = True,
        lifetime = True,
        excitation_wavelengths = excitation_wavelengths,
        excitation_cw = [False, False],
        detection_wavelengths = detection_wavelengths,
        excitation_alternated = [False, False])

    acquisition_duration = ((timestamps.max() - timestamps.min()) *
                            timestamps_unit)

    data = dict(
        _filename = filename_spc,
        acquisition_duration = round(acquisition_duration),
        photon_data = photon_data,
        setup = setup,
        provenance = provenance)

    return data, metadata


def nsalex_pq(filename,
              donor=0,
              acceptor=1,
              alex_period_donor=(150, 1500),
              alex_period_acceptor=(1540, 3050),
              excitation_wavelengths=(523e-9, 628e-9),
              detection_wavelengths=(580e-9, 680e-9)):
    """Load PicoQuant PTU, HT3 or PT3 files containing ns-ALEX data.

    This function returns a dictionary that can be passed to
    :func:`phconvert.hdf5.save_photon_hdf5` to save a Photon-HDF5 file.
    """
    file_type = filename.lower()[-3:]
    assert file_type in ('ptu', 'ht3', 'pt3')
    load_pq = {'ptu': pqreader.load_ptu, 'ht3': pqreader.load_ht3,
               'pt3': pqreader.load_pt3}
    assert os.path.isfile(filename), "File '%s' not found." % filename
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, nanotimes, metadata, marker_ids = load_pq[file_type](filename)
    print(" [DONE]\n")

    software = metadata.pop('software')
    software_version = metadata.pop('software_version')
    laser_repetition_rate = float(metadata.pop('laser_repetition_rate'))
    acquisition_duration = float(metadata.pop('acquisition_duration'))
    timestamps_unit = float(metadata.pop('timestamps_unit'))
    tcspc_unit = float(metadata.pop('nanotimes_unit'))
    tcspc_num_bins = 4096
    tcspc_range = tcspc_num_bins * tcspc_unit

    # Creation time from the file header
    creation_time = metadata.pop('creation_time')

    provenance = dict(
        filename=filename,
        creation_time=creation_time,
        software=software,
        software_version=software_version,
    )

    photon_data = dict(
        timestamps=timestamps,
        timestamps_specs=dict(timestamps_unit=timestamps_unit),
        detectors=detectors,
        nanotimes=nanotimes,

        nanotimes_specs=dict(
            tcspc_unit=tcspc_unit,
            tcspc_range=tcspc_range,
            tcspc_num_bins=tcspc_num_bins),

        measurement_specs=dict(
            measurement_type='smFRET-nsALEX',
            laser_repetition_rate=laser_repetition_rate,
            alex_excitation_period1=alex_period_donor,
            alex_excitation_period2=alex_period_acceptor,
            detectors_specs=dict(spectral_ch1=np.atleast_1d(donor),
                                 spectral_ch2=np.atleast_1d(acceptor))),
    )

    setup = dict(
        num_pixels=2,
        num_spots=1,
        num_spectral_ch=2,
        num_polarization_ch=1,
        num_split_ch=1,
        modulated_excitation=True,
        lifetime=True,
        excitation_wavelengths=excitation_wavelengths,
        excitation_cw=[False, False],
        detection_wavelengths=detection_wavelengths,
        excitation_alternated=[False, False])

    data = dict(
        _filename=filename,
        acquisition_duration=acquisition_duration,
        photon_data=photon_data,
        setup=setup,
        provenance=provenance)

    return data, metadata


def nsalex_ht3(filename,
               donor=0,
               acceptor=1,
               alex_period_donor=(150, 1500),
               alex_period_acceptor=(1540, 3050),
               excitation_wavelengths=(523e-9, 628e-9),
               detection_wavelengths=(580e-9, 680e-9)):
    """Load a .ht3 file containing ns-ALEX data and return a dict.

    WARNING: This function is deprecated. Please use :func:`nsalex_pq` instead.
    """
    return nsalex_pq(filename, donor=donor, acceptor=acceptor,
                     alex_period_donor=alex_period_donor,
                     alex_period_acceptor=alex_period_acceptor,
                     excitation_wavelengths=excitation_wavelengths,
                     detection_wavelengths=detection_wavelengths)


def nsalex_pt3(filename,
               donor=0,
               acceptor=1,
               alex_period_donor=(150, 1500),
               alex_period_acceptor=(1540, 3050),
               excitation_wavelengths=(523e-9, 628e-9),
               detection_wavelengths=(580e-9, 680e-9)):
    """Load a .pt3 file containing ns-ALEX data and return a dict.

    WARNING: This function is deprecated. Please use :func:`nsalex_pq` instead.
    """
    return nsalex_pq(filename, donor=donor, acceptor=acceptor,
                     alex_period_donor=alex_period_donor,
                     alex_period_acceptor=alex_period_acceptor,
                     excitation_wavelengths=excitation_wavelengths,
                     detection_wavelengths=detection_wavelengths)


def nsalex_t3r(filename,
               donor = 0,
               acceptor = 1,
               alex_period_donor = (150, 1500),
               alex_period_acceptor = (1540, 3050),
               excitation_wavelengths = (523e-9, 628e-9),
               detection_wavelengths = (580e-9, 680e-9)):
    """Load a .t3r file containing ns-ALEX data and return a dict.

    This dictionary can be passed to the :func:`phconvert.hdf5.save_photon_hdf5`
    function to save the data in Photon-HDF5 format.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename
    print(" - Loading '%s' ... " % filename)
    timestamps, detectors, nanotimes, metadata = pqreader.load_t3r(filename)
    print(" [DONE]\n")

    timestamps_unit = float(metadata.pop('timestamps_unit'))
    tcspc_unit = float(metadata.pop('nanotimes_unit'))
    tcspc_num_bins = 4096
    tcspc_range = tcspc_num_bins * tcspc_unit
    laser_repetition_rate = float(metadata['ttmode']['SyncRate'])
    acquisition_duration = float(metadata['header']['AcquisitionTime'][0] * 1e-3)
    software = str(metadata['header']['SoftwareVersion'][0])
    software_version = str(metadata['header']['HardwareVersion'][0])

    # Extract the creation time from the PT3 file header as it will be
    # more reliable than the creation time from the file system
    ctime_t = time.strptime(metadata['header']['FileTime'][0].decode(),
                            "%d-%m-%y %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)

    provenance = dict(
        filename = filename,
        creation_time = creation_time,
        software = software,
        software_version = software_version,
    )

    photon_data = dict(
        timestamps = timestamps,
        timestamps_specs = dict(timestamps_unit=timestamps_unit),
        detectors = detectors,
        nanotimes = nanotimes,

        nanotimes_specs = dict(
            tcspc_unit = tcspc_unit,
            tcspc_range = tcspc_range,
            tcspc_num_bins = tcspc_num_bins),

        measurement_specs = dict(
            measurement_type = 'smFRET-nsALEX',
            laser_repetition_rate = laser_repetition_rate,
            alex_excitation_period1 = alex_period_donor,
            alex_excitation_period2 = alex_period_acceptor,
            detectors_specs = dict(spectral_ch1 = np.atleast_1d(donor),
                                   spectral_ch2 = np.atleast_1d(acceptor)))
    )

    setup = dict(
        num_pixels = 2,
        num_spots = 1,
        num_spectral_ch = 2,
        num_polarization_ch = 1,
        num_split_ch = 1,
        modulated_excitation = True,
        lifetime = True,
        excitation_wavelengths = excitation_wavelengths,
        excitation_cw = [False, False],
        detection_wavelengths = detection_wavelengths,
        excitation_alternated = [False, False])

    data = dict(
        _filename=filename,
        acquisition_duration = acquisition_duration,
        photon_data = photon_data,
        setup = setup,
        provenance = provenance)

    return data, metadata
