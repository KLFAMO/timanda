from timanda.timeperiod import TimePeriod, TimePeriods

def test_timeperiod_creation():
    tp = TimePeriod(5, 10)
    assert tp.start == 5
    assert tp.stop == 10

    tp_reversed = TimePeriod(10, 5)
    assert tp_reversed.start == 5
    assert tp_reversed.stop == 10

def test_timeperiod_str():
    tp = TimePeriod(5, 10)
    assert str(tp) == "5 -> 10"

def test_timeperiod_intersection():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(8, 12)
    intersection = tp1 * tp2
    assert intersection.start == 8
    assert intersection.stop == 10

    no_intersection = tp1 * TimePeriod(11, 15)
    assert no_intersection is None

def test_timeperiod_join_if_overlap():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(8, 12)
    joined = tp1.joinIfOverlap(tp2)
    assert joined.start == 5
    assert joined.stop == 12

    no_join = tp1.joinIfOverlap(TimePeriod(11, 15))
    assert no_join is None

def test_timeperiod_len():
    tp = TimePeriod(5, 10)
    assert tp.len() == 5

def test_timeperiods_creation():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(15, 20)
    tps = TimePeriods(periods=[tp1, tp2])
    assert len(tps.periods) == 2
    assert tps.periods[0].start == 5
    assert tps.periods[1].stop == 20

def test_timeperiods_str():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(15, 20)
    tps = TimePeriods(periods=[tp1, tp2])
    assert str(tps) == "| 5 -> 10 | 15 -> 20 "

def test_timeperiods_append_period():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(15, 20)
    tps = TimePeriods(periods=[tp1])
    tps.append(tp2)
    assert len(tps.periods) == 2
    assert tps.periods[1].start == 15

    tp3 = TimePeriod(8, 12)
    tps.append(tp3)
    assert len(tps.periods) == 2
    assert tps.periods[0].start == 5
    assert tps.periods[0].stop == 12

def test_timeperiods_append_periods():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(15, 20)
    tp3 = TimePeriod(25, 30)
    tps1 = TimePeriods(periods=[tp1])
    tps2 = TimePeriods(periods=[tp2, tp3])
    tps1.append(tps2)
    assert len(tps1.periods) == 3
    assert tps1.periods[1].start == 15
    assert tps1.periods[2].stop == 30

def test_timeperiods_common_part():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(8, 12)
    tp3 = TimePeriod(15, 20)
    tps = TimePeriods(periods=[tp1, tp3])
    common = tps.commonPart(tp2)
    assert len(common.periods) == 1
    assert common.periods[0].start == 8
    assert common.periods[0].stop == 10

def test_timeperiods_total_time_without_gaps():
    tp1 = TimePeriod(5, 10)
    tp2 = TimePeriod(15, 20)
    tps = TimePeriods(periods=[tp1, tp2])
    assert tps.totalTimeWithoutGaps() == 10