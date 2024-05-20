import os
import numpy as np

import pytest
import phconvert as phc

DATADIR = 'notebooks/data/'


def dataset1():
    fn = '161128_DM1_50pM_pH74.ptu'  # 'rtHydraHarp2T3'
    fname = DATADIR + fn
    return fname


def dataset2():
    fn = '20161027_DM1_1nM_pH7_20MHz1.ptu'  # 'rtHydraHarp2T3'
    fname = DATADIR + fn
    return fname


def dataset3():
    fn = 'Cy3+Cy5_diff_PIE-FRET.ptu'  # 'rtPicoHarpT3'
    fname = DATADIR + fn
    return fname


def dataset4():
    fn = 'TestFile_2.ptu'  # 'rtHydraHarpT3'
    fname = DATADIR + fn
    return fname

def dataset5():
    fn = 'trace_T2_300s_1_coincidence.ptu'  # 'rtTimeHarp260PT2'
    fname = DATADIR + fn
    return fname


@pytest.fixture(scope="module",
                params=[dataset1, dataset2, dataset3, dataset4, dataset5])
def filename(request):
    fname = request.param()
    return fname


def test_read_ptu_recordtype(filename):
    """Test for PTU header decoding."""
    assert os.path.isfile(filename), 'File not found: %s' % filename
    t3records, record_type, tags = \
        phc.pqreader.ptu_reader(filename)
    assert list(tags.keys())[-1] == "Header_End"
    ptu_rec_code = {0x00010304: 'rtHydraHarpT3',
                    0x01010304: 'rtHydraHarp2T3',
                    0x00010303: 'rtPicoHarpT3',
                    0x00010206: 'rtTimeHarp260PT2'
                    }
    rec_type2 = ptu_rec_code[tags['TTResultFormat_TTTRRecType']['value']]
    assert rec_type2 == record_type


def test_ptu_rtHydraHarp2T3_overflow_correction():
    """Test PTU overflow correction for rtHydraHarp2T3 records."""
    filename = dataset2()
    assert os.path.isfile(filename), 'File not found: %s' % filename
    t3records, record_type, tags = \
        phc.pqreader.ptu_reader(filename)
    if record_type == 'rtHydraHarp2T3':
        det, ts, nanot = phc.pqreader.process_t3records(
            t3records[:1000000], time_bit=10, dtime_bit=15, ch_bit=6,
            special_bit=True, ovcfunc=phc.pqreader._correct_overflow_nsync)
        det2, ts2, nanot2 = phc.pqreader.process_t3records(
            t3records[:1000000], time_bit=10, dtime_bit=15, ch_bit=6,
            special_bit=True, ovcfunc=phc.pqreader._correct_overflow_nsync_naive)
        assert (det == det2).all()
        assert (ts == ts2).all()
        assert (nanot == nanot2).all()


def test_ptu_rtHydraHarpT3_overflow_correction():
    """Test PTU overflow correction for rtHydraHarpT3 records."""
    filename = dataset4()
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_ptu(filename)
    timestamps2, detectors2, nanotimes2, meta2 = phc.pqreader.load_ptu(
        filename, ovcfunc=phc.pqreader._correct_overflow2)
    assert (timestamps == timestamps2).all()


def test_load_ptu(filename):
    """Test consistency of data loaded from PTU files."""
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_ptu(filename)
    # Test metadata
    tags = meta['tags']
    acq_duration = tags['MeasDesc_AcquisitionTime']['value'] * 1e-3
    acq_duration_exp = ((timestamps[-1] - timestamps[0])
                        * meta['timestamps_unit'])
    assert abs(acq_duration - acq_duration_exp) < 0.1
    for field in ('software', 'software_version', 'creation_time',
                  'hardware_name'):
        assert field in meta
        assert isinstance(field, str)
    # Test photon data
    valid = detectors != 127
    timestamps = timestamps[valid]
    assert (np.diff(timestamps) >= 0).all()
    if nanotimes is not None:
        assert 'laser_repetition_rate' in meta
        assert 1e6 < meta['laser_repetition_rate'] > 10e6
        assert ((nanotimes >= 0) * (nanotimes < 4096)).all()


def test_load_ht3():
    """Test loading HT3 files."""
    fn = 'Pre.ht3'
    filename = DATADIR + fn
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_ht3(filename)
    acq_duration = meta['header']['Tacq'] * 1e-3
    acq_duration2 = (timestamps[-1] - timestamps[0]) * meta['timestamps_unit']
    assert abs(acq_duration - acq_duration2) < 0.1
    assert (np.diff(timestamps) >= 0).all()


def test_load_pt3():
    """Test loading PT3 files."""
    fn = 'topfluorPE_2_1_1_1.pt3'
    filename = DATADIR + fn
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_pt3(filename)
    acq_duration = meta['header']['AcquisitionTime'] * 1e-3
    acq_duration2 = (timestamps[-1] - timestamps[0]) * meta['timestamps_unit']
    # The two acquisition times should match. TODO: find out why they don't
    #assert abs(acq_duration - acq_duration2) < 0.1   ## BROKEN TEST!
    assert (np.diff(timestamps) >= 0).all()


def test_load_phu():
    """Test loading PT3 files."""
    fn = 'nanodiamant_histo.phu'
    filename = DATADIR + fn
    assert os.path.isfile(filename), 'File not found: %s' % filename
    hist, bin_size, meta = phc.pqreader.load_phu(filename)
    
    # test acquisition duration
    acq_duration = meta['tags']['MeasDesc_AcquisitionTime']['value'] * 1e-3
    acq_duration2 = meta['acquisition_duration']
    assert acq_duration == acq_duration2 == 10
    
    # test hist.shape
    tags = meta['tags']
    num_curves = tags['HistoResult_NumberOfCurves']['value']
    num_bins = [tag['value'] for tag in tags['HistResDscr_HistogramBins']]
    assert hist.shape[1] == num_bins[0]
    assert hist.shape[0] == len(bin_size) == num_curves

    # all value in `num_bins` and in `bin_size` must be the same
    def all_values_equal(x):
        return all(np.array(x) == x[0])
    assert all_values_equal(num_bins)
    assert all_values_equal(bin_size)

    # test `bin_size`
    bin_size2 = [tag['value'] for tag in tags['HistResDscr_MDescResolution']]
    assert np.array_equal(bin_size, bin_size2)
    

   
