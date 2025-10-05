from timanda.tserie import TSerie
import numpy as np

def test_create_empty_tserie():
    ts = TSerie()
    assert ts.label == ''
    assert len(ts.mjd_tab) == 0
    assert len(ts.val_tab) == 0
    assert len(ts.pps_tab) == 0

def test_create_tserie_with_data():
    mjd = [1.0, 2.0, 3.0]
    val = [10.0, 20.0, 30.0]
    ts = TSerie(label="Test Series", mjd=mjd, val=val)
    assert ts.label == "Test Series"
    assert np.array_equal(ts.mjd_tab, np.array(mjd))
    assert np.array_equal(ts.val_tab, np.array(val))
    assert np.array_equal(ts.pps_tab, np.ones(len(mjd), dtype=int))

def test_create_tserie_with_pps():
    mjd = [1.0, 2.0, 3.0]
    val = [10.0, 20.0, 30.0]
    pps = [1, 2, 3]
    ts = TSerie(label="Test Series", mjd=mjd, val=val, pps=pps)
    assert ts.label == "Test Series"
    assert np.array_equal(ts.mjd_tab, np.array(mjd))
    assert np.array_equal(ts.val_tab, np.array(val))
    assert np.array_equal(ts.pps_tab, np.array(pps))

def test_create_tserie_invalid_pps_length():
    mjd = [1.0, 2.0, 3.0]
    val = [10.0, 20.0, 30.0]
    pps = [1, 2]  # Incorrect length
    try:
        TSerie(label="Invalid PPS", mjd=mjd, val=val, pps=pps)
        assert False, "Expected ValueError due to mismatched lengths"
    except ValueError as e:
        assert str(e) == "Length of pps must be equal to the length of mjd and val"

def test_create_tserie_invalid_mjd_val_length():
    mjd = [1.0, 2.0]
    val = [10.0, 20.0, 30.0]  # Incorrect length
    try:
        TSerie(label="Invalid MJD/VAL", mjd=mjd, val=val)
        assert False, "Expected ValueError due to mismatched lengths"
    except ValueError as e:
        assert str(e) == "Length of mjd and val must be equal"