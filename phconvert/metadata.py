# -*- coding: utf-8 -*-
#
# phconvert - Convert files to Photon-HDF5 format
#
# Copyright (C) 2014-2015 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module defines the string descriptions for all the fields in the
**Photon-HDF5** format.
"""

from collections import OrderedDict


official_fields_descr = OrderedDict([
    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Root fields
    ##
    ('/acquisition_time',
     'Measurement duration in seconds.'),

    ('/comment',
     'A user defined comment for the data file.'),

    ## Photon data group
    ('/photon_data',
     'Group containing arrays of photon-data (one element per photon)'),

    ('/photon_data/timestamps',
     'Array of photon timestamps.'),

    ('/photon_data/detectors',
     'Array of detector IDs for each timestamp.'),

    ('/photon_data/nanotimes',
     'TCSPC photon arrival time (nanotimes).'),

    ('/photon_data/particles',
     'Particle IDs (integer) for each timestamp.'),

    ('/photon_data/timestamps_specs',
     'Specifications for timestamps.'),

    ('/photon_data/timestamps_specs/timestamps_unit',
     'Time in seconds of 1-unit increment in timestamps.'),

    ('/photon_data/nanotimes_specs',
     'Group for nanotime-specific data.'),

    ('/photon_data/nanotimes_specs/tcspc_unit',
     'TCSPC time bin duration in seconds (nanotimes unit).'),

    ('/photon_data/nanotimes_specs/tcspc_num_bins',
     'Number of TCSPC bins.'),

   ('/photon_data/nanotimes_specs/tcspc_range',
    'TCSPC full-scale range in seconds.'),

    ('/photon_data/nanotimes_specs/time_reversed',
     ('True (i.e. 1) if nanotimes contains the time elapsed between a photon '
      'and the next laser pulse. False (i.e. 0) if it contains the time '
      'elapsed between a laser pulse and a photon.')),

    ('/photon_data/measurement_specs',
     ('Metadata necessary for interpretation of the particular type of '
      'measurement.')),

    ('/photon_data/measurement_specs/measurement_type',
     'Name of the measurement the data represents.'),

    ('/photon_data/measurement_specs/alex_period',
     ('Period of laser alternation in us-ALEX measurements in timestamps '
      'units.')),

    ('/photon_data/measurement_specs/laser_pulse_rate',
     'Repetition rate of the pulsed excitation laser.'),

    ('/photon_data/measurement_specs/alex_period_spectral_ch1',
     ('Value pair identifing the range of spectral_ch1 photons in one '
      'period of laser alternation or interleaved pulses.')),

    ('/photon_data/measurement_specs/alex_period_spectral_ch2',
     ('Value pair identifing the range of spectral_ch2 photons in one '
      'period of laser alternation or interleaved pulses.')),

    ('/photon_data/measurement_specs/detectors_specs',
     'Mapping between the detector IDs and the detection channels.'),

    ('/photon_data/measurement_specs/detectors_specs/spectral_ch1',
     ('Pixel IDs for the first spectral channel (i.e. donor in a '
      '2-color smFRET measurement).')),

    ('/photon_data/measurement_specs/detectors_specs/spectral_ch2',
     ('Pixel IDs for the first spectral channel (i.e. acceptor in a '
      '2-color smFRET measurement).')),

    ('/photon_data/measurement_specs/detectors_specs/polarization_ch1',
     'Pixel IDs for the first polarization channel.'),

    ('/photon_data/measurement_specs/detectors_specs/polarization_ch2',
     'Pixel IDs for the second polarization channel.'),

    ('/photon_data/measurement_specs/detectors_specs/split_ch1',
     ('Pixel IDs for the first channel splitted through a '
     'non-polarizing beam splitter.')),

    ('/photon_data/measurement_specs/detectors_specs/split_ch2',
     ('Pixel IDs for the second channel splitted through a '
      'non-polarizing beam splitter.')),

    ('/photon_data/measurement_specs/detectors_specs/labels',
     ('User defined labels for each pixel IDs. In smFRET it is strongly '
      'suggested to use "donor" and "acceptor" for the respective '
      'pixel IDs.')),

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Setup group
    ##
    ('/setup', 'Information about the experimental setup.'),
    ('/setup/num_pixels',
     'Total number of detector pixels.'),

    ('/setup/num_spots',
     'Number of excitation (or detection) "spots" in the sample.'),

    ('/setup/num_spectral_ch',
     'Number of distinct spectral bands that are acquired.'),

    ('/setup/num_polarization_ch',
     'Number of distinct polarization states that are acquired.'),

    ('/setup/num_split_ch',
     ('Number of distinct detection channels detecting the same '
      'spectral band and polarization. These are channels are > 1 when '
      'using a non-polarizing beam splitter.')),

    ('/setup/modulated_excitation',
     ('True (or 1) if there is any form of excitation modulation either in '
      'the wavelength space (as in us-ALEX or PAX) or in the polarization '
      'space. This field is also True for pulse-interleaved excitation (PIE) '
      'or ns-ALEX measurements.')),

    ('/setup/lifetime',
     ('True (or 1) if the measurements includes a nanotimes array of '
      '(usually sub-ns resolution) photon arrival times with respect to a '
      'laser pulse (as in TCSPC measurements).')),

    ('/setup/excitation_wavelengths',
     ('List of excitation wavelengths (center wavelength if broad-band) in '
      'increasing order (unit: meter).')),

    ('/setup/excitation_cw',
     ('For each excitation source, this field indicates whether excitation '
      'is continuous wave (CW), True (i.e. 1), or pulsed, False (i.e. 0).')),

    ('/setup/excitation_polarizations',
     'List of polarization angles (in degrees) for each excitation source.'),

    ('/setup/excitation_input_powers',
     ('Excitation power in Watts for each excitation source. This is the '
      'excitation power entering the optical system.')),

    ('/setup/excitation_intensity',
     ('Excitation intensity in the sample for each excitation source (units: '
      'Watts/meters^2). In the case of confocal excitation this is the peak '
      'PSF intensity.')),

    ('/setup/detection_wavelengths',
     'Reference wavelengths (m) for each detected spectral band.'),

    ('/setup/detection_polarizations',
     'Polarization angles for each detected polarization.'),

    ('/setup/detection_split_ch_ratios',
     ('Power fraction detected by each "beam-split" channel (i.e. '
      'independent detection channels obtained through a non-polarizing '
      'beam splitter).')),

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Identity group
    ##
    ('/identity', 'Information about the Photon-HDF5 data file.'),

    ('/identity/author',
     'Author of the current data file.'),

    ('/identity/author_affiliation',
     'Company or institution the author is affiliated with.'),

    ('/identity/creator',
     'Creator of the current Photon-HDF5 file.'),

    ('/identity/creator_affiliation',
     'Company or institution the creator is affiliated with.'),

    ('/identity/url',
     'URL that allow to download the Photon-HDF5 data file.'),

    ('/identity/doi',
     'Digital Object Identifier (DOI) for the Photon-HDF5 data file.'),

    ('/identity/filename',
     ('Original file name of the current Photon-HDF5 file '
      '(i.e. file name at creation time).')),

    ('/identity/filename_full',
     ('Original file name (with full path) of the current Photon-HDF5 file '
      '(i.e. full file name at creation time).')),

    ('/identity/creation_time',
     'Creation time of the current Photon-HDF5 file.'),

    ('/identity/software',
     'Name of the software used to create the current Photon-HDF5 file.'),

    ('/identity/software_version',
     'Version of the software used to create current the Photon-HDF5 file.'),

    ('/identity/format_name',
     'Name of the file format.'),

    ('/identity/format_version',
     'Version for the Photon-HDF5 format.'),

    ('/identity/format_url',
     'Official URL for the Photon-HDF5 format.'),

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Provenance group
    ##
    ('/provenance',
     'Information about the original data file.'),

    ('/provenance/filename',
     'File name of the original data file before conversion to Photon-HDF5.'),

    ('/provenance/filename_full',
     ('File name (with full path) of the original data file before conversion '
      'to Photon-HDF5.')),

    ('/provenance/creation_time',
     'The creation time of the original data file.'),

    ('/provenance/modification_time',
     'Time of last modification of the original data file.'),

    ('/provenance/software',
     'Software used to save the original data file.'),

    ('/provenance/software_version',
     'Version of the software used to save the original data file.'),

    ## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ## Sample group
    ##
    ('/sample',
     'Information about the measured sample.'),

    ('/sample/num_dyes',
     'Number of different dyes present in the samples.'),

    ('/sample/dye_names',
     'List of dye names present in the sample.'),

    ('/sample/buffer_name',
     'A descriptive name for the buffer.'),

    ('/sample/sample_name',
     'A descriptive name for the sample.'),

])
