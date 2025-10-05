class TimePeriod:
    """
        Class for managing single continuous time period.
        Includes information only about start and stop mjd.
    """
    
    def __init__(self, start, stop):
        self.start = min(start, stop)
        self.stop = max(start, stop)

    def __str__(self):
        return str(self.start) + " -> " + str(self.stop)

    def __mul__(self, b):
        """
            Intersection of TimePeriods
        """
        
        if isinstance(b, TimePeriod):
            start = max(self.start, b.start)
            stop = min(self.stop, b.stop)
            if start < stop:
                return TimePeriod(start, stop)
        return None

    def joinIfOverlap(self, b):
        if isinstance(b, TimePeriod):
            if self*b is not None:
                start = min(self.start, b.start)
                stop = max(self.stop, b.stop)
                return TimePeriod(start, stop)
        return None

    def len(self):
        return self.stop-self.start


class TimePeriods:
    """
    Class for managing multiple time periods

    :Changes:
        2023-09-21 by Piotr Morzy?ski: Firs version
    """
    def __init__(self, periods=None):
        """
        :param periods - list of elements of type TimePeriod
        """
        self.periods = []
        if periods is not None:
            for period in periods:
                self.periods.append(period)

    def __str__(self):
        return "".join(['| '+x.__str__()+' ' for x in self.periods])

    def appendPeriod(self, periodToAdd):
        outputPeriods = []
        insertIndex = None
        for c, x in enumerate(self.periods):
            if x*periodToAdd is None:
                outputPeriods.append(x)
                if (insertIndex is None and
                        periodToAdd.stop < x.start):
                    insertIndex = c
            else:
                if insertIndex is None:
                    insertIndex = c
                periodToAdd = periodToAdd.joinIfOverlap(x)
        if insertIndex is None:
            insertIndex = len(outputPeriods)
        outputPeriods.insert(insertIndex, periodToAdd)
        self.periods = outputPeriods

    def appendPeriods(self, TimePeriodsToAdd):
        for x in TimePeriodsToAdd.periods:
            self.appendPeriod(x)

    def append(self, b):
        if isinstance(b, TimePeriod):
            self.appendPeriod(b)
        if isinstance(b, TimePeriods):
            self.appendPeriods(b)

    def commonPart(self, b):
        if isinstance(b, TimePeriod):
            outPeriods = []
            for x in self.periods:
                xb = x*b
                if xb is not None:
                    outPeriods.append(xb)
            if len(outPeriods) == 0:
                return None
            return TimePeriods(periods=outPeriods)
        if isinstance(b, TimePeriods):
            outPeriods = []
            for x in b.periods:
                cp = self.commonPart(x)
                if cp is not None:
                    outPeriods.append(cp)
            if len(outPeriods) == 0:
                return None
            return TimePeriods(periods=outPeriods)
        return None

    def totalTimeWithoutGaps(self):
        acc = 0
        for x in self.periods:
            acc = acc+x.len()
        return acc
    