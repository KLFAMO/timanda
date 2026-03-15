from timanda.tserie import TSerie
from decimal import Decimal as D
import numpy as np
import pytest

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
    assert np.array_equal(ts.s_tab, np.array([0.0, 86400.0, 172800.0]))

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
        "\t1.000000\t10.000000\t1\n"
        "\t2.000000\t20.000000\t1\n"
        "\t3.000000\t30.000000\t1\n"
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
        "\t1.000000\t10.000000\t1\n"
        "\t2.000000\t11.000000\t1\n"
        "\t3.000000\t12.000000\t1\n"
        "\t4.000000\t13.000000\t1\n"
        "\t5.000000\t14.000000\t1\n"
        "\t...\n"
        "\t16.000000\t25.000000\t1\n"
        "\t17.000000\t26.000000\t1\n"
        "\t18.000000\t27.000000\t1\n"
        "\t19.000000\t28.000000\t1\n"
        "\t20.000000\t29.000000\t1\n"
    )
    assert result == expected

def test_tserie_str_with_flags():
    ts = TSerie(
        label="Flags Series",
        mjd=[1.0, 2.0, 3.0],
        val=[10.0, 20.0, 30.0],
        flags=[1, 0, 1],
        use_flags=True,
    )
    result = str(ts)
    expected = (
        "TSerie:\tlabel: Flags Series\tlength: 3\tlen_mjd: 2.000000\n"
        "\t1.000000\t10.000000\t1\t1\n"
        "\t2.000000\t20.000000\t1\t0\n"
        "\t3.000000\t30.000000\t1\t1\n"
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
    assert ts.mean_val == 20.0  # Average of [10.0, 20.0, 30.0]
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
    assert ts.mean_val is None
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
    assert ts.mean_val == 20.0  # Average of [10.0, 30.0], ignoring np.nan
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
    assert ts.mean_val == None

def test_mean_empty_series():
    """Test mean calculation for an empty series."""
    ts = TSerie()
    assert ts.mean() is None, "Mean of an empty series should be None"

def test_mean_simple_series():
    """Test mean calculation for a simple series."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    assert ts.mean() == 20.0, "Mean of [10.0, 20.0, 30.0] should be 20.0"

def test_mean_with_decimal():
    """Test mean calculation using Decimal for higher precision."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts.mean(decimal=True, decimal_out=True)
    assert result == D("20.0"), "Mean with Decimal should be Decimal('20.0')"

def test_mean_with_decimal_float_output():
    """Test mean calculation using Decimal but returning float."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    result = ts.mean(decimal=True, decimal_out=False)
    assert result == 20.0, "Mean with Decimal but float output should be 20.0"

def test_mean_with_nan_values():
    """Test mean calculation ignoring NaN values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, np.nan, 30.0])
    ts.rm_nans()  # Remove NaN values
    assert ts.mean() == 20.0, "Mean should ignore NaN values and be 20.0"

def test_mean_with_weighted_pps():
    """Test mean calculation using pps_tab for weighted mean."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0], pps=[1, 2, 3])
    result, total_weight = ts.mean(use_pps=True)
    assert result == 23.333333333333332, "Weighted mean should be 23.333333333333332"
    assert total_weight == 6, "Total weight should be 6"

def test_mean_with_long_decimal_numbers():
    """Test mean calculation with long decimal numbers."""
    ts = TSerie(
        mjd=[1.0, 2.0, 3.0],
        val=[999999999.0000009, 999999999.0000009, 999999999.0000009]
    )
    result = ts.mean(decimal=True, decimal_out=True)
    assert result == D("999999999.000000953673"), "Mean should be Decimal('999999999.0000009')"
    """
    There is ...53673 at the end due to floating point precision issues which is present
    even when using Decimal because the input values are floats (during object creation).
    """

# max_val

def test_max_val_with_normal_values():
    """Test max_val with normal values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    assert ts.max_val() == 30.0, "Maximum value should be 30.0"

def test_max_val_with_negative_values():
    """Test max_val with negative values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[-10.0, -20.0, -5.0])
    assert ts.max_val() == -5.0, "Maximum value should be -5.0"

def test_max_val_with_nan_values():
    """Test max_val with NaN values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, np.nan, 30.0])
    assert ts.max_val() == 30.0, "Maximum value should ignore NaN and be 30.0"

def test_max_val_with_empty_series():
    """Test max_val with an empty series."""
    ts = TSerie(mjd=[], val=[])
    assert ts.max_val() is None, "Maximum value of an empty series should be None"

def test_max_val_with_all_nan_values():
    """Test max_val with all NaN values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[np.nan, np.nan, np.nan])
    assert ts.max_val() is None, "Maximum value of an empty series should be None"

def test_max_val_with_mixed_values():
    """Test max_val with mixed positive, negative, and NaN values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0, 4.0], val=[-10.0, 20.0, np.nan, 15.0])
    assert ts.max_val() == 20.0, "Maximum value should be 20.0, ignoring NaN"

# rm_dc

def test_rm_dc_with_normal_values():
    """Test rm_dc with normal values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    ts.rm_dc()
    assert np.array_equal(ts.val_tab, [-10.0, 0.0, 10.0]), "DC component was not removed correctly"

def test_rm_dc_with_nan_values():
    """Test rm_dc with NaN values."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, np.nan, 30.0])
    ts.rm_dc()
    assert np.array_equal(ts.val_tab, [-10.0, 10.0], equal_nan=True), "DC component was not removed correctly with NaN values"

def test_rm_dc_with_empty_series():
    """Test rm_dc with an empty series."""
    ts = TSerie(mjd=[], val=[])
    ts.rm_dc()
    assert len(ts.val_tab) == 0, "rm_dc should not modify an empty series"

def test_rm_dc_with_single_value():
    """Test rm_dc with a single value."""
    ts = TSerie(mjd=[1.0], val=[10.0])
    ts.rm_dc()
    assert np.array_equal(ts.val_tab, [0.0]), "rm_dc should set the single value to 0.0"

# rm_drift

def test_rm_drift_with_linear_trend():
    """Test rm_drift with a simple linear trend."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    ts.rm_drift()
    assert np.allclose(ts.val_tab, [0.0, 0.0, 0.0]), "Drift was not removed correctly"

def test_rm_drift_with_no_trend():
    """Test rm_drift with no linear trend."""
    ts = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 10.0, 10.0])
    ts.rm_drift()
    assert np.allclose(ts.val_tab, [0.0, 0.0, 0.0]), "Drift removal should result in all zeros"

def test_rm_drift_with_empty_series():
    """Test rm_drift with an empty series."""
    ts = TSerie(mjd=[], val=[])
    ts.rm_drift()
    assert len(ts.val_tab) == 0, "rm_drift should not modify an empty series"

def test_rm_drift_with_single_point():
    """Test rm_drift with a single data point."""
    ts = TSerie(mjd=[1.0], val=[10.0])
    ts.rm_drift()
    assert np.allclose(ts.val_tab, [0.0]), "Drift removal for a single point should result in 0.0"

# split

def test_split_with_flags():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000030,

        60000.000200,
        60000.000210,
        60000.000220,
        60000.000230,

        60000.000400,
        60000.000410,
        60000.000420,
        60000.000430,
    ]

    val = list(range(12))
    pps = list(range(1, 13))
    flags = [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1]

    ts = TSerie(
        mjd=mjd,
        val=val,
        pps=pps,
        flags=flags,
        use_flags=True
    )

    out = ts.split(min_gap_s=8)

    assert len(out) == 3

    assert np.array_equal(out[0].mjd_tab, np.array(mjd[0:4]))
    assert np.array_equal(out[0].flags, np.array(flags[0:4]))

    assert np.array_equal(out[1].mjd_tab, np.array(mjd[4:8]))
    assert np.array_equal(out[1].flags, np.array(flags[4:8]))

    assert np.array_equal(out[2].mjd_tab, np.array(mjd[8:12]))
    assert np.array_equal(out[2].flags, np.array(flags[8:12]))

    for seg in out:
        assert seg.use_flags is True

def test_split_without_flags():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000030,

        60000.000200,
        60000.000210,
        60000.000220,
        60000.000230,

        60000.000400,
        60000.000410,
        60000.000420,
        60000.000430,
    ]

    val = list(range(12))
    pps = list(range(1, 13))

    ts = TSerie(
        mjd=mjd,
        val=val,
        pps=pps,
        use_flags=False
    )

    out = ts.split(min_gap_s=8)

    assert len(out) == 3

    assert np.array_equal(out[0].mjd_tab, np.array(mjd[0:4]))
    assert np.array_equal(out[1].mjd_tab, np.array(mjd[4:8]))
    assert np.array_equal(out[2].mjd_tab, np.array(mjd[8:12]))

    for seg in out:
        assert seg.flags is None
        assert seg.use_flags is False


# append

def test_append_single_without_flags():
    ts = TSerie(
        mjd=[60000.0, 60000.0001],
        val=[10.0, 11.0],
        pps=[1, 2],
        use_flags=False,
    )

    ts.append(60000.0002, 12.0, pps=3, flags=0)

    assert np.array_equal(ts.mjd_tab, np.array([60000.0, 60000.0001, 60000.0002]))
    assert np.array_equal(ts.val_tab, np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3]))
    assert ts.flags is None
    assert ts.use_flags is False

def test_append_single_with_flags():
    ts = TSerie(
        mjd=[60000.0, 60000.0001],
        val=[10.0, 11.0],
        pps=[1, 2],
        use_flags=True,
    )

    ts.append(60000.0002, 12.0, pps=3, flags=0)

    assert np.array_equal(ts.mjd_tab, np.array([60000.0, 60000.0001, 60000.0002]))
    assert np.array_equal(ts.val_tab, np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3]))
    assert np.array_equal(ts.flags, np.array([1, 1, 0]))
    assert ts.use_flags is True

def test_append_array_without_flags():
    ts = TSerie(
        mjd=[60000.0, 60000.0001],
        val=[10.0, 11.0],
        pps=[1, 2],
        use_flags=False,
    )

    ts.append(
        [60000.0002, 60000.0003, 60000.0004],
        [12.0, 13.0, 14.0],
        pps=[3, 4, 5],
        flags=[0, 1, 0],
    )

    assert np.array_equal(
        ts.mjd_tab,
        np.array([60000.0, 60000.0001, 60000.0002, 60000.0003, 60000.0004])
    )
    assert np.array_equal(
        ts.val_tab,
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    assert np.array_equal(
        ts.pps_tab,
        np.array([1, 2, 3, 4, 5])
    )
    assert ts.flags is None
    assert ts.use_flags is False

def test_append_array_with_flags():
    ts = TSerie(
        mjd=[60000.0, 60000.0001],
        val=[10.0, 11.0],
        pps=[1, 2],
        use_flags=True,
    )

    ts.append(
        [60000.0002, 60000.0003, 60000.0004],
        [12.0, 13.0, 14.0],
        pps=[3, 4, 5],
        flags=[0, 1, 0],
    )

    assert np.array_equal(
        ts.mjd_tab,
        np.array([60000.0, 60000.0001, 60000.0002, 60000.0003, 60000.0004])
    )
    assert np.array_equal(
        ts.val_tab,
        np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    )
    assert np.array_equal(
        ts.pps_tab,
        np.array([1, 2, 3, 4, 5])
    )
    assert np.array_equal(
        ts.flags,
        np.array([1, 1, 0, 1, 0])
    )
    assert ts.use_flags is True

def test_append_array_with_default_pps_and_flags():
    ts = TSerie(
        mjd=[60000.0],
        val=[10.0],
        use_flags=True,
    )

    ts.append(
        [60000.0001, 60000.0002],
        [11.0, 12.0],
    )

    assert np.array_equal(
        ts.mjd_tab,
        np.array([60000.0, 60000.0001, 60000.0002])
    )
    assert np.array_equal(
        ts.val_tab,
        np.array([10.0, 11.0, 12.0])
    )
    assert np.array_equal(
        ts.pps_tab,
        np.array([1, 1, 1])
    )
    assert np.array_equal(
        ts.flags,
        np.array([1, 1, 1])
    )


def test_append_raises_for_wrong_pps_length():
    ts = TSerie(
        mjd=[60000.0],
        val=[10.0],
    )

    with pytest.raises(ValueError, match="Length of pps"):
        ts.append(
            [60000.0001, 60000.0002],
            [11.0, 12.0],
            pps=[3],
        )


def test_append_raises_for_wrong_flags_length():
    ts = TSerie(
        mjd=[60000.0],
        val=[10.0],
        use_flags=True,
    )

    with pytest.raises(ValueError, match="Length of flags"):
        ts.append(
            [60000.0001, 60000.0002],
            [11.0, 12.0],
            flags=[1],
        )


def test_append_raises_for_wrong_mjd_val_length():
    ts = TSerie(
        mjd=[60000.0],
        val=[10.0],
    )

    with pytest.raises(ValueError, match="Length of mjd and val must be equal"):
        ts.append(
            [60000.0001, 60000.0002],
            [11.0],
        )

# getrange

def test_getrange_with_flags():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000030,
        60000.000040,
        60000.000050,
        60000.000060,
    ]
    val = [10, 11, 12, 13, 14, 15, 16]
    pps = [1, 2, 3, 4, 5, 6, 7]
    flags = [1, 0, 1, 1, 0, 1, 0]

    ts = TSerie(
        mjd=mjd,
        val=val,
        pps=pps,
        flags=flags,
        use_flags=True,
    )

    out = ts.getrange(60000.000015, 60000.000055)

    assert out is not None
    assert np.array_equal(
        out.mjd_tab,
        np.array([60000.000020, 60000.000030, 60000.000040, 60000.000050])
    )
    assert np.array_equal(out.val_tab, np.array([12, 13, 14, 15], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([3, 4, 5, 6]))
    assert np.array_equal(out.flags, np.array([1, 1, 0, 1]))
    assert out.use_flags is True


def test_getrange_without_flags():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000030,
        60000.000040,
        60000.000050,
        60000.000060,
    ]
    val = [10, 11, 12, 13, 14, 15, 16]
    pps = [1, 2, 3, 4, 5, 6, 7]

    ts = TSerie(
        mjd=mjd,
        val=val,
        pps=pps,
        use_flags=False,
    )

    out = ts.getrange(60000.000015, 60000.000055)

    assert out is not None
    assert np.array_equal(
        out.mjd_tab,
        np.array([60000.000020, 60000.000030, 60000.000040, 60000.000050])
    )
    assert np.array_equal(out.val_tab, np.array([12, 13, 14, 15], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([3, 4, 5, 6]))
    assert out.flags is None
    assert out.use_flags is False


def test_getrange_outside_returns_none():
    ts = TSerie(
        mjd=[60000.0, 60000.0001, 60000.0002],
        val=[10.0, 11.0, 12.0],
        use_flags=False,
    )

    out = ts.getrange(60001.0, 60001.1)

    assert out is None


def test_getrange_clips_to_series_bounds():
    ts = TSerie(
        mjd=[60000.0, 60000.0001, 60000.0002],
        val=[10.0, 11.0, 12.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    out = ts.getrange(59999.0, 60001.0)

    assert out is not None
    assert np.array_equal(
        out.mjd_tab,
        np.array([60000.0, 60000.0001, 60000.0002])
    )
    assert np.array_equal(out.val_tab, np.array([10.0, 11.0, 12.0]))
    assert np.array_equal(out.pps_tab, np.array([1, 2, 3]))


def test_getrange_rm_first_and_last():
    mjd = [
        60000.000000,
        60000.000010,
        60000.000020,
        60000.000030,
        60000.000040,
    ]
    val = [10, 11, 12, 13, 14]

    ts = TSerie(
        mjd=mjd,
        val=val,
        use_flags=False,
    )

    # zakres zaczyna się i kończy pomiędzy punktami
    out = ts.getrange(60000.000015, 60000.000035)

    assert out is not None
    assert np.array_equal(
        out.mjd_tab,
        np.array([60000.000020, 60000.000030])
    )
    assert np.array_equal(
        out.val_tab,
        np.array([12, 13], dtype=float)
    )

# rmrange

def test_rmrange_no_overlap_returns_self():
    ts = TSerie(
        mjd=[60000.000000, 60000.000010, 60000.000020],
        val=[10.0, 11.0, 12.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    out, code = ts.rmrange(60001.0, 60001.1)

    assert code == 0
    assert out is ts


def test_rmrange_removes_all():
    ts = TSerie(
        mjd=[60000.000000, 60000.000010, 60000.000020],
        val=[10.0, 11.0, 12.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    out, code = ts.rmrange(59999.0, 60001.0)

    assert code == 1
    assert out is None


def test_rmrange_remove_left_part_without_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
        ],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        use_flags=False,
    )

    out, code = ts.rmrange(59999.0, 60000.000020)

    assert code == 2
    assert out is not None
    assert np.array_equal(out.mjd_tab, np.array([60000.000030, 60000.000040]))
    assert np.array_equal(out.val_tab, np.array([13, 14], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([4, 5]))
    assert out.flags is None
    assert out.use_flags is False


def test_rmrange_remove_right_part_without_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
        ],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        use_flags=False,
    )

    out, code = ts.rmrange(60000.000020, 60001.0)

    assert code == 3
    assert out is not None
    assert np.array_equal(out.mjd_tab, np.array([60000.000000, 60000.000010]))
    assert np.array_equal(out.val_tab, np.array([10, 11], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([1, 2]))
    assert out.flags is None
    assert out.use_flags is False


def test_rmrange_remove_middle_without_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
            60000.000050,
            60000.000060,
        ],
        val=[10, 11, 12, 13, 14, 15, 16],
        pps=[1, 2, 3, 4, 5, 6, 7],
        use_flags=False,
    )

    out, code = ts.rmrange(60000.000020, 60000.000040)

    assert code == 4
    assert isinstance(out, list)
    assert len(out) == 2

    left, right = out

    assert np.array_equal(left.mjd_tab, np.array([60000.000000, 60000.000010]))
    assert np.array_equal(left.val_tab, np.array([10, 11], dtype=float))
    assert np.array_equal(left.pps_tab, np.array([1, 2]))
    assert left.flags is None
    assert left.use_flags is False

    assert np.array_equal(right.mjd_tab, np.array([60000.000050, 60000.000060]))
    assert np.array_equal(right.val_tab, np.array([15, 16], dtype=float))
    assert np.array_equal(right.pps_tab, np.array([6, 7]))
    assert right.flags is None
    assert right.use_flags is False


def test_rmrange_remove_middle_with_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
            60000.000050,
            60000.000060,
        ],
        val=[10, 11, 12, 13, 14, 15, 16],
        pps=[1, 2, 3, 4, 5, 6, 7],
        flags=[1, 0, 1, 1, 0, 1, 0],
        use_flags=True,
    )

    out, code = ts.rmrange(60000.000020, 60000.000040)

    assert code == 4
    assert isinstance(out, list)
    assert len(out) == 2

    left, right = out

    assert np.array_equal(left.mjd_tab, np.array([60000.000000, 60000.000010]))
    assert np.array_equal(left.val_tab, np.array([10, 11], dtype=float))
    assert np.array_equal(left.pps_tab, np.array([1, 2]))
    assert np.array_equal(left.flags, np.array([1, 0]))
    assert left.use_flags is True

    assert np.array_equal(right.mjd_tab, np.array([60000.000050, 60000.000060]))
    assert np.array_equal(right.val_tab, np.array([15, 16], dtype=float))
    assert np.array_equal(right.pps_tab, np.array([6, 7]))
    assert np.array_equal(right.flags, np.array([1, 0]))
    assert right.use_flags is True


def test_rmrange_remove_left_part_with_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
        ],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        flags=[1, 0, 1, 1, 0],
        use_flags=True,
    )

    out, code = ts.rmrange(59999.0, 60000.000020)

    assert code == 2
    assert out is not None
    assert np.array_equal(out.mjd_tab, np.array([60000.000030, 60000.000040]))
    assert np.array_equal(out.val_tab, np.array([13, 14], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([4, 5]))
    assert np.array_equal(out.flags, np.array([1, 0]))
    assert out.use_flags is True


def test_rmrange_remove_right_part_with_flags():
    ts = TSerie(
        mjd=[
            60000.000000,
            60000.000010,
            60000.000020,
            60000.000030,
            60000.000040,
        ],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        flags=[1, 0, 1, 1, 0],
        use_flags=True,
    )

    out, code = ts.rmrange(60000.000020, 60001.0)

    assert code == 3
    assert out is not None
    assert np.array_equal(out.mjd_tab, np.array([60000.000000, 60000.000010]))
    assert np.array_equal(out.val_tab, np.array([10, 11], dtype=float))
    assert np.array_equal(out.pps_tab, np.array([1, 2]))
    assert np.array_equal(out.flags, np.array([1, 0]))
    assert out.use_flags is True

# rmindexes

def test_rm_indexes_without_flags():
    ts = TSerie(
        mjd=[1, 2, 3, 4, 5],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        use_flags=False,
    )

    ts.rm_indexes([1, 3])

    assert np.array_equal(ts.mjd_tab, np.array([1, 3, 5], dtype=float))
    assert np.array_equal(ts.val_tab, np.array([10, 12, 14], dtype=float))
    assert np.array_equal(ts.pps_tab, np.array([1, 3, 5]))
    assert ts.flags is None
    assert ts.len == 3


def test_rm_indexes_with_flags():
    ts = TSerie(
        mjd=[1, 2, 3, 4, 5],
        val=[10, 11, 12, 13, 14],
        pps=[1, 2, 3, 4, 5],
        flags=[1, 0, 1, 1, 0],
        use_flags=True,
    )

    ts.rm_indexes([1, 3])

    assert np.array_equal(ts.mjd_tab, np.array([1, 3, 5], dtype=float))
    assert np.array_equal(ts.val_tab, np.array([10, 12, 14], dtype=float))
    assert np.array_equal(ts.pps_tab, np.array([1, 3, 5]))
    assert np.array_equal(ts.flags, np.array([1, 1, 0]))
    assert ts.len == 3
    assert ts.use_flags is True


def test_rm_indexes_none_does_nothing():
    ts = TSerie(
        mjd=[1, 2, 3],
        val=[10, 11, 12],
        pps=[1, 2, 3],
        flags=[1, 0, 1],
        use_flags=True,
    )

    ts.rm_indexes(None)

    assert np.array_equal(ts.mjd_tab, np.array([1, 2, 3], dtype=float))
    assert np.array_equal(ts.val_tab, np.array([10, 11, 12], dtype=float))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3]))
    assert np.array_equal(ts.flags, np.array([1, 0, 1]))
    assert ts.len == 3


def test_rm_indexes_single_index():
    ts = TSerie(
        mjd=[1, 2, 3, 4],
        val=[10, 11, 12, 13],
        pps=[1, 2, 3, 4],
        flags=[1, 0, 1, 0],
        use_flags=True,
    )

    ts.rm_indexes(2)

    assert np.array_equal(ts.mjd_tab, np.array([1, 2, 4], dtype=float))
    assert np.array_equal(ts.val_tab, np.array([10, 11, 13], dtype=float))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 4]))
    assert np.array_equal(ts.flags, np.array([1, 0, 0]))
    assert ts.len == 3

# time diff to freq

import numpy as np
from timanda.tserie import TSerie


def test_time_diff_to_freq_diff_without_fill_last_point():
    ts = TSerie(
        mjd=[1.0, 1.00001, 1.00002],
        val=[10.0, 20.0, 50.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    ts.time_diff_to_freq_diff(fill_last_point=False)

    delta_s = 0.00001 * 86400.0
    expected_val = np.array([
        (20.0 - 10.0) / delta_s,
        (50.0 - 20.0) / delta_s,
    ])

    assert np.array_equal(ts.mjd_tab, np.array([1.0, 1.00001]))
    assert np.allclose(ts.val_tab, expected_val)
    assert np.array_equal(ts.pps_tab, np.array([1, 2]))
    assert ts.flags is None
    assert ts.len == 2


def test_time_diff_to_freq_diff_with_fill_last_point():
    ts = TSerie(
        mjd=[1.0, 1.00001, 1.00002],
        val=[10.0, 20.0, 50.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    ts.time_diff_to_freq_diff(fill_last_point=True)

    delta_s = 0.00001 * 86400.0
    f1 = (20.0 - 10.0) / delta_s
    f2 = (50.0 - 20.0) / delta_s

    assert np.array_equal(ts.mjd_tab, np.array([1.0, 1.00001, 1.00002]))
    assert np.allclose(ts.val_tab, np.array([f1, f2, f2]))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3]))
    assert ts.flags is None
    assert ts.len == 3


def test_time_diff_to_freq_diff_with_flags():
    ts = TSerie(
        mjd=[1.0, 1.00001, 1.00002, 1.00003],
        val=[10.0, 20.0, 50.0, 80.0],
        pps=[1, 2, 3, 4],
        flags=[1, 0, 1, 0],
        use_flags=True,
    )

    ts.time_diff_to_freq_diff(fill_last_point=True)

    delta_s = 0.00001 * 86400.0
    f1 = (20.0 - 10.0) / delta_s
    f2 = (50.0 - 20.0) / delta_s
    f3 = (80.0 - 50.0) / delta_s

    assert np.array_equal(ts.mjd_tab, np.array([1.0, 1.00001, 1.00002, 1.00003]))
    assert np.allclose(ts.val_tab, np.array([f1, f2, f3, f3]))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3, 4]))
    assert np.array_equal(ts.flags, np.array([1, 0, 1, 0]))
    assert ts.use_flags is True
    assert ts.len == 4


def test_time_diff_to_freq_diff_with_flags_without_fill_last_point():
    ts = TSerie(
        mjd=[1.0, 1.00001, 1.00002, 1.00003],
        val=[10.0, 20.0, 50.0, 80.0],
        pps=[1, 2, 3, 4],
        flags=[1, 0, 1, 0],
        use_flags=True,
    )

    ts.time_diff_to_freq_diff(fill_last_point=False)

    delta_s = 0.00001 * 86400.0
    f1 = (20.0 - 10.0) / delta_s
    f2 = (50.0 - 20.0) / delta_s
    f3 = (80.0 - 50.0) / delta_s

    assert np.array_equal(ts.mjd_tab, np.array([1.0, 1.00001, 1.00002]))
    assert np.allclose(ts.val_tab, np.array([f1, f2, f3]))
    assert np.array_equal(ts.pps_tab, np.array([1, 2, 3]))
    assert np.array_equal(ts.flags, np.array([1, 0, 1]))
    assert ts.use_flags is True
    assert ts.len == 3


def test_time_diff_to_freq_diff_too_short_series():
    ts = TSerie(
        mjd=[1.0],
        val=[10.0],
        pps=[1],
        flags=[1],
        use_flags=True,
    )

    ts.time_diff_to_freq_diff(fill_last_point=True)

    assert np.array_equal(ts.mjd_tab, np.array([1.0]))
    assert np.array_equal(ts.val_tab, np.array([10.0]))
    assert np.array_equal(ts.pps_tab, np.array([1]))
    assert np.array_equal(ts.flags, np.array([1]))
    assert ts.len == 1

# npz

def test_to_npz_payload_without_flags():
    ts = TSerie(
        label="Test Series",
        mjd=[1.0, 2.0, 3.0],
        val=[10.0, 20.0, 30.0],
        pps=[1, 2, 3],
        use_flags=False,
    )

    payload = ts.to_npz_payload()

    assert set(payload.keys()) == {"mjd", "val", "pps", "label", "use_flags"}

    assert np.array_equal(payload["mjd"], np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert np.array_equal(payload["val"], np.array([10.0, 20.0, 30.0], dtype=np.float64))
    assert np.array_equal(payload["pps"], np.array([1, 2, 3], dtype=np.int32))
    assert payload["label"] == np.array("Test Series", dtype=np.str_)
    assert payload["use_flags"] == np.array(False, dtype=np.bool_)

    assert "flags" not in payload


def test_to_npz_payload_with_flags():
    ts = TSerie(
        label="Flags Series",
        mjd=[1.0, 2.0, 3.0],
        val=[10.0, 20.0, 30.0],
        pps=[1, 2, 3],
        flags=[1, 0, 1],
        use_flags=True,
    )

    payload = ts.to_npz_payload()

    assert set(payload.keys()) == {"mjd", "val", "pps", "label", "use_flags", "flags"}

    assert np.array_equal(payload["mjd"], np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert np.array_equal(payload["val"], np.array([10.0, 20.0, 30.0], dtype=np.float64))
    assert np.array_equal(payload["pps"], np.array([1, 2, 3], dtype=np.int32))
    assert np.array_equal(payload["flags"], np.array([1, 0, 1], dtype=np.int32))
    assert payload["label"] == np.array("Flags Series", dtype=np.str_)
    assert payload["use_flags"] == np.array(True, dtype=np.bool_)


def test_to_npz_payload_with_prefix():
    ts = TSerie(
        label="Prefixed",
        mjd=[1.0, 2.0],
        val=[10.0, 20.0],
        pps=[1, 2],
        flags=[1, 0],
        use_flags=True,
    )

    payload = ts.to_npz_payload(prefix="seg0_")

    assert set(payload.keys()) == {
        "seg0_mjd",
        "seg0_val",
        "seg0_pps",
        "seg0_label",
        "seg0_use_flags",
        "seg0_flags",
    }

    assert np.array_equal(payload["seg0_mjd"], np.array([1.0, 2.0], dtype=np.float64))
    assert np.array_equal(payload["seg0_val"], np.array([10.0, 20.0], dtype=np.float64))
    assert np.array_equal(payload["seg0_pps"], np.array([1, 2], dtype=np.int32))
    assert np.array_equal(payload["seg0_flags"], np.array([1, 0], dtype=np.int32))
    assert payload["seg0_label"] == np.array("Prefixed", dtype=np.str_)
    assert payload["seg0_use_flags"] == np.array(True, dtype=np.bool_)


def test_to_npz_payload_empty_label():
    ts = TSerie(
        label="",
        mjd=[1.0],
        val=[10.0],
        pps=[1],
        use_flags=False,
    )

    payload = ts.to_npz_payload()

    assert payload["label"] == np.array("", dtype=np.str_)
    assert payload["use_flags"] == np.array(False, dtype=np.bool_)


def test_to_npz_payload_preserves_flags_even_if_use_flags_false():
    ts = TSerie(
        label="Stored Flags",
        mjd=[1.0, 2.0, 3.0],
        val=[10.0, 20.0, 30.0],
        pps=[1, 2, 3],
        flags=[1, 0, 1],
        use_flags=False,
    )

    payload = ts.to_npz_payload()

    assert "flags" in payload
    assert np.array_equal(payload["flags"], np.array([1, 0, 1], dtype=np.int32))
    assert payload["use_flags"] == np.array(False, dtype=np.bool_)