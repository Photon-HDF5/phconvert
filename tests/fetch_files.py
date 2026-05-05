# Author: Paul David Harris
# Created: 04/05/2026
# Purpose: Download files from Zenodo repository

import pooch

DATASET_DIR = u'../notebooks/data'

repo = pooch.create(path=DATASET_DIR, base_url='doi:10.5281/zenodo.20035771')
repo.load_registry_from_doi()

files = ('161128_DM1_50pM_pH74.ptu', '20161027_DM1_1nM_pH7_20MHz1.ptu', 'Cy3+Cy5_diff_PIE-FRET.ptu',
         'TestFile_2.ptu', 'trace_T2_300s_1_coincidence.ptu', 'RawData.ptu', 
         'Pre.ht3', 'topfluorPE_2_1_1_1.pt3', 'nanodiamant_histo.phu', 'DNA_FRET_0.5nM.pt3', 
         'Point_11_s23d9A15_70_25sucrose_55_25.t3r',
         '0023uLRpitc_NTP_20dT_0.5GndCl.sm', 
         'test_noise.spc', 'test_noise.set', 'test_noise.asc', 
         'dsdna_d7d17_50_50_1.spc', 'dsdna_d7d17_50_50_1.set')

for file in files:
    repo.fetch(file)
