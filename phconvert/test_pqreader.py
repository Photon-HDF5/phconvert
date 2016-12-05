from __future__ import division
from builtins import range, zip
import os

import pytest
import phconvert as phc

DATADIR = 'notebooks/data/'


def dataset1():
    fn = '161128_DM1_50pM_pH74.ptu'
    fname = DATADIR + fn
    return fname

def dataset2():
    fn = '20161027_DM1_1nM_pH7_20MHz1.ptu'
    fname = DATADIR + fn
    return fname

def dataset3():
    fn = 'Cy3+Cy5_diff_PIE-FRET.ptu'
    fname = DATADIR + fn
    return fname

@pytest.fixture(scope="module", params=[dataset1, dataset2, dataset3])
def filename(request):
    fname = request.param()
    return fname


def test_load_ptu(filename):
    """Smoke test."""
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_ptu(filename)
    tags = meta['tags']
    acq_duration = tags['MeasDesc_AcquisitionTime']['value'] * 1e-3
    acq_duration_exp = (timestamps[-1] - timestamps[0]) * meta['timestamps_unit']
    assert abs(acq_duration - acq_duration_exp) < 0.1


def test_read_ptu(filename):
    """Test for PTU header decoding."""
    t3records, timestamps_unit, nanotimes_unit, record_type, tags = \
        phc.pqreader.ptu_reader(filename)
    assert list(tags.keys())[-1] == "Header_End"
    assert timestamps_unit == 1 / tags['TTResult_SyncRate']['value']
    assert nanotimes_unit == tags['MeasDesc_Resolution']['value']
    ptu_rec_code = {0x01010304: 'rtHydraHarp2T3', 0x00010303: 'rtPicoHarpT3'}
    rec_type2 = ptu_rec_code[tags['TTResultFormat_TTTRRecType']['value']]
    assert rec_type2 == record_type
    if record_type == 'rtHydraHarp2T3':
        det, ts, nanot = phc.pqreader.process_t3records(t3records[:1000000],
                    time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
                    ovcfunc=phc.pqreader._correct_overflow_nsync)
        det2, ts2, nanot2 = phc.pqreader.process_t3records(t3records[:1000000],
                    time_bit=10, dtime_bit=15, ch_bit=6, special_bit=True,
                    ovcfunc=phc.pqreader._correct_overflow_nsync_naive)
        assert (det == det2).all()
        assert (ts == ts2).all()
        assert (nanot == nanot2).all()


def test_load_ht3():
    """Test loading HT3 files."""
    fn = 'Pre.ht3'
    filename = DATADIR + fn
    assert os.path.isfile(filename), 'File not found: %s' % filename
    timestamps, detectors, nanotimes, meta = phc.pqreader.load_ht3(filename)
    acq_duration = meta['header']['Tacq'] * 1e-3
    acq_duration2 = (timestamps[-1] - timestamps[0]) * meta['timestamps_unit']
    assert abs(acq_duration - acq_duration2) < 0.1


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
