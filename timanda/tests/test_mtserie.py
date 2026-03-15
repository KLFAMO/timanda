from timanda.mtserie import MTSerie
from timanda.tserie import TSerie
import numpy as np

def test_create_empty_mtserie():
    mts = MTSerie()
    assert mts.label == ''
    assert mts.color == 'green'
    assert len(mts.dtab) == 0

def test_mtserie_with_tseries():
    ts1 = TSerie(label="TSerie 1", mjd=[1.0, 2.0], val=[10.0, 20.0])
    ts2 = TSerie(label="TSerie 2", mjd=[3.0, 4.0], val=[30.0, 40.0])
    mts = MTSerie(label="Test MTSerie", tseries=[ts1, ts2])
    
    assert mts.label == "Test MTSerie"
    assert len(mts.dtab) == 2
    assert mts.dtab[0].label == "TSerie 1"
    assert mts.dtab[1].label == "TSerie 2" 

def test_create_mtserie_with_mjd_val():
    mts = MTSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    assert len(mts.dtab) == 1
    assert mts.dtab[0].mjd_tab.tolist() == [1.0, 2.0, 3.0]
    assert mts.dtab[0].val_tab.tolist() == [10.0, 20.0, 30.0]

def test_mtserie_with_txt_file(mocker):
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="1.000001 10.0\n1.000002 20.0\n1.000003 30.0\n1.000004 40.0\n")
    )
    mts = MTSerie(txtFileName="test_file.txt")
    
    assert len(mts.dtab) == 1
    assert np.array_equal(mts.dtab[0].mjd_tab, np.array([1.000001, 1.000002, 1.000003, 1.000004]))
    assert np.array_equal(mts.dtab[0].val_tab, np.array([10.0, 20.0, 30.0, 40.0]))

def test_rm_nans():
    """Test the rm_nans method to ensure it removes NaN values correctly."""
    mjd = [1.0, 2.0, 3.0, 4.0]
    val = [10.0, np.nan, 30.0, np.nan]
    pps = [1, 2, 3, 4]
    
    ts = TSerie(mjd=mjd, val=val, pps=pps)
    ts.rm_nans()

    # Expected results after removing NaN values
    expected_mjd = np.array([1.0, 3.0])
    expected_val = np.array([10.0, 30.0])
    expected_pps = np.array([1, 3])

    assert np.array_equal(ts.mjd_tab, expected_mjd), "MJD values do not match expected results"
    assert np.array_equal(ts.val_tab, expected_val), "VAL values do not match expected results"
    assert np.array_equal(ts.pps_tab, expected_pps), "PPS values do not match expected results"

# split

def test_mtserie_split_with_flags():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000200,  # gap
        60000.000210,
        60000.000220,
    ]

    val = [10, 11, 12, 13, 14, 15]
    flags = [1, 0, 1, 1, 0, 1]

    ts = TSerie(
        mjd=mjd,
        val=val,
        flags=flags,
        use_flags=True
    )

    mts = MTSerie(tseries=[ts], use_flags=True)

    mts.split(min_gap_s=8)

    assert len(mts.dtab) == 2

    ts1 = mts.dtab[0]
    ts2 = mts.dtab[1]

    assert ts1.use_flags is True
    assert ts2.use_flags is True

    assert np.array_equal(ts1.mjd_tab, np.array(mjd[:3]))
    assert np.array_equal(ts1.val_tab, np.array(val[:3], dtype=float))
    assert np.array_equal(ts1.flags, np.array(flags[:3]))

    assert np.array_equal(ts2.mjd_tab, np.array(mjd[3:]))
    assert np.array_equal(ts2.val_tab, np.array(val[3:], dtype=float))
    assert np.array_equal(ts2.flags, np.array(flags[3:]))

def test_mtserie_split_without_flags():
    mjd = [
        60000.0,
        60000.00001,
        60000.00002,
        60000.00020,
        60000.00021,
    ]

    val = [1, 2, 3, 4, 5]

    ts = TSerie(mjd=mjd, val=val)

    mts = MTSerie(tseries=[ts])

    mts.split(min_gap_s=8)

    assert len(mts.dtab) == 2

    assert mts.dtab[0].flags is None
    assert mts.dtab[1].flags is None

# getrange

def test_mtserie_getrange_flags():
    mjd = [1,2,3,4,5]
    val = [10,11,12,13,14]
    flags = [1,0,1,1,0]

    ts = TSerie(mjd=mjd, val=val, flags=flags, use_flags=True)
    mts = MTSerie(tseries=[ts], use_flags=True)

    out = mts.getrange(2,4)

    assert out.use_flags is True
    assert len(out.dtab) == 1
    assert np.array_equal(out.dtab[0].flags, np.array([0,1,1]))


def test_mtserie_rmrange_with_flags():
    mjd = [1,2,3,4,5,6]
    val = [10,11,12,13,14,15]
    flags = [1,0,1,1,0,1]

    ts = TSerie(mjd=mjd, val=val, flags=flags, use_flags=True)
    mts = MTSerie(tseries=[ts], use_flags=True)

    mts.rmrange(2.5,4.5)

    assert len(mts.dtab) == 2
    assert np.array_equal(mts.dtab[0].flags, np.array([1,0]))
    assert np.array_equal(mts.dtab[1].flags, np.array([0,1]))

#-----

def test_mtserie_set_flags_in_range():
    ts1 = TSerie(
        mjd=[1.0, 2.0, 3.0, 4.0],
        val=[10.0, 11.0, 12.0, 13.0],
    )
    ts2 = TSerie(
        mjd=[5.0, 6.0, 7.0],
        val=[20.0, 21.0, 22.0],
    )

    mts = MTSerie(tseries=[ts1, ts2], use_flags=False)

    mts.set_flags_in_range(2.5, 6.5, 0)

    assert mts.use_flags is True

    assert np.array_equal(mts.dtab[0].flags, np.array([1, 1, 0, 0]))
    assert np.array_equal(mts.dtab[1].flags, np.array([0, 0, 1]))

    assert mts.dtab[0].use_flags is True
    assert mts.dtab[1].use_flags is True