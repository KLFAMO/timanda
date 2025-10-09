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

# __str__ method tests

def test_tserie_str_empty():
    ts = TSerie()
    result = str(ts)
    assert result == '\tEmpty'

def test_tserie_str_small_data():
    ts = TSerie(
        label="Test Series",
        mjd=[1.0, 2.0, 3.0],
        val=[10.0, 20.0, 30.0]
    )
    result = str(ts)
    expected = (
        "TSerie:\tlabel: Test Series\tlength: 3\tlen_mjd: 2.000000\n"
        "\t1.000000\t10.000000\t1.000000\n"
        "\t2.000000\t20.000000\t1.000000\n"
        "\t3.000000\t30.000000\t1.000000\n"
    )
    assert result == expected

def test_tserie_str_large_data():
    ts = TSerie(
        label="Large Series",
        mjd=list(range(1, 21)),
        val=list(range(10, 30))
    )
    result = str(ts)
    expected = (
        "TSerie:\tlabel: Large Series\tlength: 20\tlen_mjd: 19.000000\n"
        "\t1.000000\t10.000000\t1.000000\n"
        "\t2.000000\t11.000000\t1.000000\n"
        "\t3.000000\t12.000000\t1.000000\n"
        "\t4.000000\t13.000000\t1.000000\n"
        "\t5.000000\t14.000000\t1.000000\n"
        "\t...\n"
        "\t16.000000\t25.000000\t1.000000\n"
        "\t17.000000\t26.000000\t1.000000\n"
        "\t18.000000\t27.000000\t1.000000\n"
        "\t19.000000\t28.000000\t1.000000\n"
        "\t20.000000\t29.000000\t1.000000\n"
    )
    assert result == expected


# Math operations

def test_tserie_add_scalar():
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts + 5
    assert np.array_equal(result.val_tab, np.array([15.0, 25.0, 35.0]))
    assert np.array_equal(result.mjd_tab, ts.mjd_tab)

def test_tserie_sub_scalar():
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts - 5
    assert np.array_equal(result.val_tab, np.array([5.0, 15.0, 25.0]))
    assert np.array_equal(result.mjd_tab, ts.mjd_tab)

def test_tserie_mul_scalar():
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts * 2
    assert np.array_equal(result.val_tab, np.array([20.0, 40.0, 60.0]))
    assert np.array_equal(result.mjd_tab, ts.mjd_tab)

def test_tserie_truediv_scalar():
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts / 2
    assert np.array_equal(result.val_tab, np.array([5.0, 10.0, 15.0]))
    assert np.array_equal(result.mjd_tab, ts.mjd_tab)

def test_tserie_add_tserie_not_supported():
    ts1 = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    ts2 = TSerie(mjd=[1.0, 2.0, 3.0], val=[5.0, 15.0, 25.0])
    result = ts1 + ts2
    assert result is None  # Dodawanie dwóch TSerie nie jest wspierane

def test_tserie_invalid_operation():
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts + "invalid"
    assert result is None


# calc_tab

def test_calc_tab_with_valid_data():
    """Test calc_tab with valid MJD and VAL data."""
    mjd = [1.0, 2.0, 3.0]
    val = [10.0, 20.0, 30.0]
    ts = TSerie(mjd=mjd, val=val)
    ts.calc_tab()

    assert ts.len == 3
    assert ts.isempty is 0
    assert ts.mjd_start == 1.0
    assert ts.mjd_stop == 3.0
    assert ts.mean == 20.0  # Average of [10.0, 20.0, 30.0]
    assert np.array_equal(ts.s_tab, np.array([0.0, 86400.0, 172800.0]))
    assert ts.len_mjd == 2.0
    assert ts.len_s == 172800.0

def test_calc_tab_with_empty_data():
    """Test calc_tab with empty MJD and VAL data."""
    ts = TSerie()
    ts.calc_tab()

    assert ts.len == 0
    assert ts.isempty is 1
    assert ts.mjd_start == 0
    assert ts.mjd_stop == 0
    assert ts.mean is None
    assert len(ts.s_tab) == 0
    assert ts.len_mjd == 0
    assert ts.len_s == 0

def test_calc_tab_with_none_values():
    """Test calc_tab with None values in VAL data."""
    mjd = [1.0, 2.0, 3.0]
    val = [10.0, None, 30.0]
    ts = TSerie(mjd=mjd, val=val)
    assert np.array_equal(ts.val_tab, np.array([10.0, 30.0]))
    ts.calc_tab()

    assert ts.len == 2
    assert ts.isempty is 0
    assert ts.mjd_start == 1.0
    assert ts.mjd_stop == 3.0
    assert ts.mean == 20.0  # Average of [10.0, 30.0], ignoring np.nan
    assert np.array_equal(ts.pps_tab, np.array([1, 1]))
    assert ts.len_mjd == 2.0
    assert ts.len_s == 172800.0

def test_calc_tab_with_all_none_values():
    """Test calc_tab with all None values in VAL data."""
    mjd = [1.0, 2.0, 3.0]
    val = [None, None, None]
    ts = TSerie(mjd=mjd, val=val)
    ts.calc_tab()

    assert ts.len == 0
    assert ts.isempty is 1
    assert ts.mjd_start == 0
    assert ts.mjd_stop == 0
    assert ts.mean == None