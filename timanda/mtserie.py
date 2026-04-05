from timanda.tserie import TSerie
from typing import Optional, Union, Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from decimal import Decimal as D
import decimal as dec
import allantools as al
from timanda.timeperiod import TimePeriods, TimePeriod
s2mjd = 1/(60*60*24)  # seconds to MJD conversion factor
from timanda.tserie import mjd2s
import json
from pathlib import Path
import logging


class MTSerie:
    """
    Class for handling multiple time series (TSerie)
    """
    
    def __init__(
        self,
        label: str = '',
        tseries: Optional[list[TSerie]] = None,
        color: str = 'green',
        txtFileName: Optional[str] = None,
        plot_label: str = '',
        plot_ref_val: Union[int, float] = 0,
        mjd: Optional[list[float]] = None,
        val: Optional[list[float]] = None,
        flags: Optional[list[int]] = None,
        use_flags: bool = False,
        split: bool = False
    ) -> None:
        self.label: str = label
        self.plot_label: str = plot_label
        self.plot_ref_val: Union[int, float] = plot_ref_val
        self.dtab: list[TSerie] = []
        self.color: str = color
        self.use_flags: bool = use_flags

        if txtFileName:
            self.importFromTxtFile(txtFileName)
        elif tseries:
            for tserie in tseries:
                self.add_TSerie(tserie)
        elif mjd is not None and val is not None:
            ts = TSerie(mjd=mjd, val=val, flags=flags, use_flags=use_flags)
            self.add_TSerie(ts)

        self._sync_use_flags()

        if split:
            self.split()

    def importFromTxtFile(self, fileName: str, delimiter: str = ' ') -> None:
        """
        Imports MTSerie from txt file
        Args:
            fileName (str): Path to the text file.
            delimiter (str): Delimiter used in the file to separate columns.
        """
        mjd_t = []
        val_t = []
        with open(fileName, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    parts = line.split(delimiter)
                    mjd_t.append(float(parts[0]))
                    val_t.append(float(parts[1]))
        
        self.add_TSerie(TSerie(
            mjd=mjd_t,
            val=val_t,
            use_flags=self.use_flags,
        ))


    def _sync_use_flags(self):
        for ts in self.dtab:
            ts.use_flags = self.use_flags

            if self.use_flags and ts.flags is None:
                ts._create_flags_list()


    def copy(self):
        out = MTSerie(
            label=self.label,
            color=self.color,
            plot_label=self.plot_label,
            plot_ref_val=self.plot_ref_val,
            use_flags=self.use_flags,
        )

        for ts in self.dtab:
            out.add_TSerie(ts.copy())

        return out


    def __str__(self):
        s = f"MTSerie {self.label}:\n"
        for x in self.dtab:
            s = s+x.__str__()
        return s

    def __iadd__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i]+b
        return self

    def __add__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]+b)
        return out

    def __sub__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]-b)
        elif isinstance(b, (MTSerie)):
            ts = TSerie()
            tp1 = self.getTimePeriods()
            tp2 = b.getTimePeriods()
            tc = tp1.commonPart(tp2)
            for per in tc.periods:
                mjds = np.arange(per.start, per.stop, 1/(60*60*24))
                for mjd in mjds:
                    ts.append(mjd, self.mjd2val(mjd)-b.mjd2val(mjd))
            out = MTSerie()
            out.add_TSerie(TSerie(mjd=ts.mjd_tab, val=ts.val_tab))
            out.split()
        return out

    def __imul__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i] * b
        return self

    def __mul__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]*b)
        return out

    def __truediv__(self, b):
        out = MTSerie()
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                out.add_TSerie(self.dtab[i]/b)
        return out

    def __idiv__(self, b):
        if isinstance(b, (int, float)):
            for i in range(0, len(self.dtab)):
                self.dtab[i] = self.dtab[i] / b
        return self

    def isempty(self):
        for x in self.dtab:
            if x.len > 0:
                return False
        return True

    def calc_tabs(self):
        for x in self.dtab:
            x.calc_tab()

    def rmemptyseries(self):
        """
        Removes empty TSeries from MTSerie
        """

        self.calc_tabs()
        self.dtab = [x for x in self.dtab if x.isempty == 0]

    def getTimePeriods(self):
        out = TimePeriods()
        for x in self.dtab:
            if len(x.mjd_tab) > 0:
                out.appendPeriod(TimePeriod(x.mjd_tab[0], x.mjd_tab[-1]))
        return out

    def getTotalTimeWithoutGaps(self):
        tp = self.getTimePeriods()
        if tp:
            return tp.totalTimeWithoutGaps()
        else:
            return None

    def alphnorm(self, atom='88Sr'):
        d = {
            # ref: Safranova RevModPhys 2018  page 8
            '88Sr': {'fabs': dec.Decimal('429228066418000'), 'K': 1.06},
            '87Sr': {'fabs': dec.Decimal('429228004229873'), 'K': 1.06},
            '171Yb': {'fabs': dec.Decimal('518295836590865'), 'K': 1.31},
            '171Yb+': {'fabs': dec.Decimal('688358979309307'), 'K': 2.03},
            # dla E2
            # 1-5.95 dla E3
        }
        for x in d.keys():
            d[x]['factor'] = float(
                d[x]['fabs']*dec.Decimal(d[x]['K'])*dec.Decimal('1e-18')
            )
        if atom in d.keys():
            k = d[atom]['factor']
        else:
            return None
        for x in self.dtab:
            x.val_tab = x.val_tab / k

    def rm_drift_each(self):
        for x in self.dtab:
            x.rm_drift()
    
    def rm_drift(self):
        fit = np.polyfit(self.mjd_tab(), self.val_tab(), 1)
        _mean = self.mean()
        for x in self.dtab:
            x.val_tab = (
                x.val_tab - (x.mjd_tab*fit[0]+fit[1] + _mean)
            )
    
    def get_drift(self):
        fit = np.polyfit(self.mjd_tab(), self.val_tab(), 1)
        return fit[0]


    def add_TSerie(self, ser: TSerie):
        ser.use_flags = self.use_flags

        if self.use_flags and ser.flags is None:
            ser._create_flags_list()

        self.dtab.append(ser)


    def add_mjdf_data(self, mjd, f):
        tmp = TSerie(mjd=mjd, val=f, use_flags=self.use_flags)
        self.add_TSerie(tmp)  


    def add_mjdf_from_file(self, file_name):
        raw = np.load(file_name, allow_pickle=True)

        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("Input file must contain a 2D array with at least two columns: mjd and val")

        self.add_mjdf_data(raw[:, 0], raw[:, 1])


    def add_mjdf_from_datfile(self, file_name, delimiter=' ', skiprows=0):
        try:
            raw = np.loadtxt(file_name, delimiter=delimiter, skiprows=skiprows)

            if raw.size == 0:
                logging.warning(f"File '{file_name}' is empty.")
                return

            raw = np.atleast_2d(raw)

            self.add_mjdf_data(raw[:, 0], raw[:, 1])

        except FileNotFoundError:
            logging.error(f"File '{file_name}' doesn't exist.")

        except ValueError as e:
            logging.error(f"Error while reading file '{file_name}': {e}")

        except Exception as e:
            logging.error(f"Unexpected error during reading file '{file_name}': {e}")


    def plot(self, color='', show=1, ax=None, zorder=1, marker=".", linestyle='none',
             nolabels=False, time_unit='mjd'):
        for x in self.dtab:
            if time_unit == 'mjd':
                xs = x.mjd_tab
            elif time_unit == 's':
                xs = (x.mjd_tab - x.mjd_tab[0]) * 24 * 60 * 60
                
            if color == '':
                color = self.color
            if ax is None:
                plt.figure()
                plt.plot(xs, x.val_tab-self.plot_ref_val,
                          color=color, marker=marker,
                         linestyle=linestyle, zorder=zorder)
            else:
                ax.plot(xs, x.val_tab-self.plot_ref_val,
                        color=color, marker=marker,
                        linestyle=linestyle, zorder=zorder)
            if not nolabels:
                plt.ylabel(self.plot_label)
        if show == 1:
            plt.show()
    


    def plotf(self, color='', show=1, ax=None, zorder=1, marker=".", linestyle='none',
             nolabels=False, time_unit='mjd'):
        for x in self.dtab:
            if time_unit == 'mjd':
                xs = x.mjd_tab
            elif time_unit == 's':
                xs = (x.mjd_tab - x.mjd_tab[0]) * 24 * 60 * 60

            ys = x.val_tab - self.plot_ref_val

            if x.flags is not None and len(x.flags) == len(xs):
                mask1 = x.flags == 1
                mask0 = x.flags == 0

                if ax is None:
                    plt.figure()
                    if np.any(mask1):
                        plt.plot(xs[mask1], ys[mask1], color='green', marker=marker,
                                 linestyle=linestyle, zorder=zorder)
                    if np.any(mask0):
                        plt.plot(xs[mask0], ys[mask0], color='red', marker=marker,
                                 linestyle=linestyle, zorder=zorder)
                else:
                    if np.any(mask1):
                        ax.plot(xs[mask1], ys[mask1], color='green', marker=marker,
                                linestyle=linestyle, zorder=zorder)
                    if np.any(mask0):
                        ax.plot(xs[mask0], ys[mask0], color='red', marker=marker,
                                linestyle=linestyle, zorder=zorder)
            else:
                if color == '':
                    color = self.color
                if ax is None:
                    plt.plot(xs, ys, color=color, marker=marker,
                             linestyle=linestyle, zorder=zorder)
                else:
                    ax.plot(xs, ys, color=color, marker=marker,
                            linestyle=linestyle, zorder=zorder)

            if not nolabels:
                plt.ylabel(self.plot_label)
        if show == 1:
            plt.show()

    def hist(self, bins=10, orientation='vertical'):
        v = self.val_tab()
        plt.hist(v, bins=bins, orientation=orientation)

    def plot_pqg_widget(self, minusmjd=0, widget=None):
        if widget is None:
            self.widget = pg.PlotWidget()
        else:
            self.widget = widget
        for x in self.dtab:
            if len(x.mjd_tab) > 0:
                print("---------")
                print(x)
                self.widget.plot(x.mjd_tab-minusmjd, x.val_tab)
        if widget is None:
            self.widget.setBackground('w')
            self.widget.setTitle(self.label)
        return self.widget

    def plot_allan(self, atom=None, ref_val=None, rate=1, taus=None, allan_method='adev'):
        if atom == '88Sr':
            ref_val = 429228066418012.0
        if ref_val:
            ref = ref_val
        else:
            ref = 1
        y = self.val_tab()/ref
        # y = y.flatten()
        # print('y: ', y)
        if taus is None:
            taus = np.power(10, np.arange(0, int(np.log10(len(y)/rate))+0.1, 0.1))
        a = al.Dataset(data=y, rate=rate, data_type="freq", taus=taus)
        a.compute(allan_method)
        b = al.Plot()
        b.plot(a, errorbars=True, grid=True)
        b.show()


    def sew(self, grid_s=1):
        """
        Stitches values from all TSerie segments into a single NumPy array.

        Returns:
            numpy.ndarray: Array containing values from all TSerie objects in `dtab`.
        """

        g = grid_s*s2mjd
        out = []
        for x in self.dtab:
            for mjd in np.arange(x.mjd_tab[0], x.mjd_tab[-1], g):
                out.append(x.mjd2val(mjd))
        return np.array(out)


    def split(self, min_gap_s=8):
        """
        Splits all TSerie objects in `dtab` based on time gaps.

        Each TSerie is split using `TSerie.split()`. The resulting segments
        replace the original series in `self.dtab`.

        Args:
            min_gap_s (float): Minimum gap in seconds used to split segments.
        """

        tmp_tab = []
        for a in self.dtab:
            spl = a.split(min_gap_s)
            tmp_tab.extend(spl)
        self.dtab = tmp_tab


    def rm_dc_each(self):
        for x in self.dtab:
            x.rm_dc()

    def rm_dc(self):
        mean = self.mean()
        for i in range(0, len(self.dtab)):
            self.dtab[i] = self.dtab[i]-mean

    def high_gauss_filter_each(self, stddev=50, rm_dc=True):
        i = 0
        for x in self.dtab:
            i = i+1
            # print('filter '+str(i))
            if x.len < stddev*10:
                pass
                # print('to short')
                # self.dtab.remove(x)
            else:
                x.high_gauss_filter(stddev=stddev, rm_dc=rm_dc)

    def time_shift_each(self, sec):
        for x in self.dtab:
            x.time_shift(sec)

    def mjd2tabNo(self, mjd):
        for num in range(0, len(self.dtab)):
            if (
                len(self.dtab[num].mjd_tab) > 0 and
                mjd < self.dtab[num].mjd_tab[0]
            ):
                return num-1
        return len(self.dtab)-1

    def mjd2tabNoandindex(self, mjd, init_index=None):
        tabNo = self.mjd2tabNo(mjd)
        if tabNo == -1:
            index = None
        else:
            index = self.dtab[tabNo].mjd2index(mjd, init_index=init_index)
        return (tabNo, index)

    def mjd2val(self, mjd, mode=0):
        (t, i) = self.mjd2tabNoandindex(mjd)
        if i is None:
            return None
        else:
            if mode == 1:
                v = self.dtab[t].val_tab[i]
                dv = self.dtab[t].val_tab[i+1]-v
                dm = self.dtab[t].mjd_tab[i+1]-self.dtab[t].mjd_tab[i]
                xm = mjd - self.dtab[t].mjd_tab[i]
                return v + dv * xm / dm
            else:
                return self.dtab[t].val_tab[i]

    def getrange(self, from_mjd, to_mjd):
        """
        Gets a range of MTSerie between from_mjd and to_mjd
        Args:
            from_mjd: float
                starting MJD
            to_mjd: float
                ending MJD
        Returns:
            MTSerie
                MTSerie object with data in the given range
        """
        ft, fN = self.mjd2tabNoandindex(from_mjd)
        tt, tN = self.mjd2tabNoandindex(to_mjd)
        if fN is not None:
            fN = fN+1
        if (ft == tt and fN is None):
            return None
        out = MTSerie(use_flags=self.use_flags)
        if ft == -1:
            ft = 0
        if tt == -1:
            tt = 0

        for tab in self.dtab[ft:tt+1]:
            tmp = tab.getrange(
                from_mjd,
                to_mjd,
            )
            if tmp is not None:
                out.add_TSerie(tmp)
        if len(out.dtab) == 0:
            return None
        return out

    def getrange_on_self(self, fmjd, tmjd):
        """
        Keeps only the data in the range [fmjd, tmjd] by removing data outside the range.
        """
        self.rmrange(0, fmjd)
        self.rmrange(tmjd, np.inf)

    def rmrange(self, fmjd, tmjd):
        ft, fN = self.mjd2tabNoandindex(fmjd)
        tt, tN = self.mjd2tabNoandindex(tmjd)
        if fN is not None:
            fN = fN+1
        if (ft == tt and fN is None):
            return self
        if ft == -1:
            ft = 0
        if tt == -1:
            tt = 0
        i = tt
        while i >= ft:
            x = self.dtab[i]
            (a, b) = x.rmrange(fmjd, tmjd)
            if b == 1:
                del self.dtab[i]
            if b in (2, 3):
                self.dtab[i] = a
            if b == 4:
                del self.dtab[i]
                self.dtab.insert(i, a[1])
                self.dtab.insert(i, a[0])
            i = i-1

    def rm_indexes(self, indexes):
        for i, tab in enumerate(self.dtab):
            tab.rm_indexes(indexes[i])

    def rmoutlayers(self, max_iterations=4, target=None, maxdiff=None):
        i = 0
        no_rm = 0
        indexes_in_while_to_delete = []
        while (i < max_iterations and no_rm == 0):
            i += 1
            input_len = len(self.mjd_tab())
            if target is None:
                target = self.mean()
            if maxdiff is None:
                maxdiff = 3*self.std()
            indexes_in_mts_to_delete = []
            for x in self.dtab:
                indexes_in_mts_to_delete.append(
                    x.rmOutlayersOfTarget(
                        target=target,
                        maxDifference=maxdiff,
                    )
                )
            indexes_in_while_to_delete.append(indexes_in_mts_to_delete)
            if input_len == len(self.mjd_tab()):
                no_rm = 1
        return indexes_in_while_to_delete
    
    def rm_value(self, value, get_indexes_only=False):
        indexes_in_mts_to_delete = []
        for x in self.dtab:
            indexes_in_mts_to_delete.append(
                x.rm_value(value, get_indexes_only=get_indexes_only)
            )
        return indexes_in_mts_to_delete

    def rm_area(self, x1, y1, x2, y2, get_indexes_only=False):
        indexes_in_mts_to_delete = []
        for x in self.dtab:
            indexes_in_mts_to_delete.append(
                x.rm_area(x1, y1, x2, y2, get_indexes_only=get_indexes_only)
            )
        return indexes_in_mts_to_delete

    def mean(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        totlen = 0
        for x in self.dtab:
            if len(x.mjd_tab)>0:
                tab.append([x.mean(), x.len])
                totlen = totlen+x.len
        out = 0
        for x in tab:
            out = out + x[0]*x[1]/totlen
        return out

    def mean_use_pps(self, decimal=False, decimal_out=False):
        if len(self.dtab) == 0:
            return None
        tab = []
        total_points=0
        for x in self.dtab:
            if len(x.mjd_tab)>0:
                mean, ts_points = x.mean_use_pps(
                    decimal=decimal,
                    decimal_out=decimal_out
                )
                tab.append((mean, ts_points))
                total_points=total_points+ts_points
        if decimal:
            out = D('0')
            for x in tab:
                out = out+D(x[0])*D(x[1])/D(total_points)
        else:
            out = 0
            for x in tab:
                out = out + x[0]*x[1]/total_points
        if decimal_out:
            return D(out)
        else:
            return float(out)

    def std(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        for x in self.dtab:
            tab = np.concatenate((tab, x.val_tab), axis=0)
        return np.std(tab)

    def stdm(self):
        if len(self.dtab) == 0:
            return None
        tab = []
        totlen = 0
        for x in self.dtab:
            totlen = totlen + x.len
            tab = np.concatenate((tab, x.val_tab), axis=0)
        return np.std(tab)/np.sqrt(totlen)

    def len_mjd_eff(self):
        length = 0
        for x in self.dtab:
            length = length+x.len_mjd

    def mjd_tab(self):
        if len(self.dtab) > 0:
            tmp = [
                np.array(x.mjd_tab) for x in self.dtab
                if np.array(x.mjd_tab).ndim > 0
            ]
            if len(tmp) > 0:
                return np.concatenate(tmp)
        return []

    def val_tab(self):
        if len(self.dtab) > 0:
            tmp = [
                np.array(x.val_tab) for x in self.dtab
                if np.array(x.val_tab).ndim > 0
            ]
            if len(tmp) > 0:
                return np.concatenate(tmp)
        return []

    def saveToTxtFile(self, fileName):
        f = open(fileName, 'w')
        for x in self.dtab:
            i = 0
            length = len(x.mjd_tab)
            f.write('#\n')
            while i < length:
                # f.write('%f\t%f\n' % (x.mjd_tab[i], x.val_tab[i]))
                f.write(f'{x.mjd_tab[i]:.10f}\t{x.val_tab[i]:.10f}\n')
                i = i+1
        f.close()
    
    def first_mjd(self):
        """
        Returns the first MJD of MTSerie.
        """

        return self.dtab[0].first_mjd()
    
    def last_mjd(self):
        """
        Returns the last MJD of MTSerie.
        """
        
        return self.dtab[-1].last_mjd()
        

    def resample2(self, *args, **kwargs):
        """Backward-compatible alias for align_to_grid_zoh()."""
        return self.align_to_grid_zoh(*args, **kwargs)

    def align_to_grid_zoh(
            self,
            period_s: float | int = 1,
            sh_s: float | int = 0.05,
            snap_s: float | int = 0,
            tol_s: float | int = 7,
            start_mjd: float = None,
            stop_mjd: float = None,
            hold_last: bool = False,
        ):
        """
        Resample / align the time series to a regular time grid using a
        zero-order hold (nearest-past-sample) strategy.

        For each grid point t_n on the new time grid:

        1. All original samples with timestamps <= t_n + sh_s are consumed
        (in chronological order), and the most recent of them becomes the
        "last known value".
        2. If a last known value exists, it is used for this grid point
        (zero-order hold).
        3. If no last known value exists yet (we are still before the first
        original sample), but the first future sample is closer than
        tol_s seconds to t_n, this future sample is used instead.
        4. Otherwise, the grid point is marked as missing in `rm_mask`
        and its value in `nv` is set to 0.0.

        This algorithm runs in O(N + M) time, where N is the number of
        original samples and M is the number of points on the new grid.

        Parameters
        ----------
        period_s : float or int, optional
            Sampling period of the target time grid in seconds.
        sh_s : float or int, optional
            Time shift (in seconds) applied to the grid when deciding which
            original samples belong to a given grid point.
            Useful, for example, when a counter nominally produces data at
            integer seconds, but the actual readings arrive a few milliseconds
            later.
        tol_s : float or int, optional
            Tolerance window (in seconds) for matching a grid point with the
            first future original sample when no past sample is available yet.
        start_mjd : float, optional
            Start time of the target grid in MJD. If None, it is computed
            from the first MJD of the series and aligned to the grid period.
        stop_mjd : float, optional
            Stop time of the target grid in MJD. If None, it is computed
            from the last MJD of the series and aligned to the grid period.

        Returns
        -------
        nmts : MTSerie
            Resampled time series (single TSerie inside MTSerie) defined
            on the regular time grid.
        rm_mask : np.ndarray of bool
            Boolean mask of shape (len(grid),) which is True for grid points
            that could not be matched to any original sample within tol_s
            (only applicable before the first valid sample).
        """

        tol_mjd = tol_s*s2mjd
        sh_mjd = sh_s*s2mjd
        snap_mjd = snap_s*s2mjd

        first_mjd = self.first_mjd()
        first_mjd_int = np.floor(first_mjd)
        last_mjd = self.last_mjd()
        last_mjd_int = np.floor(last_mjd)
        period_mjd = period_s*s2mjd

        # find the last grid point before the first MJD
        if start_mjd is None:
            start_mjd = np.floor((first_mjd % 1)/period_mjd)*period_mjd + first_mjd_int

        # find the first grid point after the last MJD
        if stop_mjd is None:
            stop_mjd = np.ceil((last_mjd % 1)/period_mjd)*period_mjd + last_mjd_int

        nm = np.arange(start_mjd, stop_mjd, period_mjd)
        nv = np.zeros_like(nm, dtype=float)
        rm_mask = np.zeros_like(nm, dtype=bool)
        ov = self.val_tab()
        om = self.mjd_tab()

        # main iterations for all mts
        oi = 0
        v = None

        for ni in range(len(nm)):
            matched = False
            dif = 1
            while (oi < len(om)-1 and om[oi] < nm[ni] + sh_mjd):
                dif = nm[ni]-om[oi]
                v = ov[oi]
                oi = oi+1
                matched = True
            
            # --- NOWE: "snap" do siatki ---
            # Jeżeli kolejny (pierwszy nie-skonsumowany) punkt jest bardzo blisko bieżącej
            # chwili siatki (±snap_s), to użyj go od razu dla tego grid-pointu.
            if oi < len(om) and abs(om[oi] - nm[ni]) <= snap_mjd:
                v = ov[oi]
                matched = True
                # opcjonalnie: skonsumuj go, żeby nie został użyty ponownie
                if oi < len(om)-1:
                    oi = oi+1
            # --- koniec NOWE ---

            if hold_last:
                if v is not None:
                    nv[ni] = v
                else:
                    dt = om[oi] - nm[ni]
                    if 0 <= dt < tol_mjd:
                        nv[ni] = ov[oi]
                    else:
                        nv[ni] = 0
                        rm_mask[ni] = True
            else:
                # NEW behavior: do NOT fill gaps using last value.
                # Keep only points that matched a real sample for this grid point.
                if matched:
                    nv[ni] = v
                else:
                    # Optional: allow "first future sample within tol" for early start
                    # (only if you want it in no-hold mode as well).
                    dt = om[oi] - nm[ni]
                    if v is None and oi < len(om) and 0 <= dt < tol_mjd:
                        nv[ni] = ov[oi]
                        # Consume it to avoid reusing the same sample on multiple grid points
                        if oi < len(om)-1:
                            oi = oi+1
                    else:
                        nv[ni] = 0
                        rm_mask[ni] = True

                # IMPORTANT: do not carry state forward in no-hold mode
                v = None  # <-- NEW: prevents holding last value


        nts = TSerie(mjd=nm, val=nv)
        nmts = MTSerie()
        nmts.add_TSerie(nts)
        return nmts, rm_mask

    
    def resample(
            self,
            fun: str = 'mean',
            period_s: float | int = 60,
            points_ratio: float = 0.7,
            get_empty_mjd_ranges: bool = False
        ):
        """
        Resamples the time series to a given period in seconds.

        Args:
            fun: str
                function to calculate the value of the resampled point
                'mean' - mean value
                'slope' - slope
                'slope_s' - 
            period_s: float | int
                period in seconds
            points_ratio: float
                
        """
        
        self.rmemptyseries()
        first_mjd = self.first_mjd()
        first_mjd_int = np.floor(first_mjd)
        last_mjd = self.last_mjd()
        last_mjd_int = np.floor(last_mjd)
        period_mjd = period_s*s2mjd
        
        # find the last grid point before the first MJD
        start_mjd = np.floor((first_mjd % 1)/period_mjd)*period_mjd + first_mjd_int
        # find the first grid point after the last MJD
        stop_mjd = np.ceil((last_mjd % 1)/period_mjd)*period_mjd + last_mjd_int
        mjd_grid = np.arange(start_mjd, stop_mjd+1e-7, period_mjd)

        return self.resample_to_mjd_array(
            mjd_grid=mjd_grid,
            grid_period_s=period_s,
            fun=fun,
            points_ratio=points_ratio,
            get_empty_mjd_ranges=get_empty_mjd_ranges
        )


    def resample_to_mjd_array(
        self,
        mjd_grid,
        grid_period_s, # maybe not needed ?
        fun='mean',
        points_ratio=0.7,
        none_fields=False,
        none_val=None,
        get_empty_mjd_ranges=False,
    ):
        """
        params:
            mjd_grid: np.array 1D
                array of MJDs to resample to
            grid_period_s: float
                period of the grid in seconds
            fun: str
                function to calculate the value of the resampled point
                'mean' - mean value
                'slope' - slope
                'slope_s' -
            points_ratio: float
                ratio of points in the subseries to the expected number of points
                in the resampled series
            none_fields: bool
                if True, the resampled series will contain None values for the
                points that were not resampled
            none_val: any
                value to use for the None fields
        """

        sample_period_s = self.get_sample_period_s()
        expected_number_of_points = grid_period_s/sample_period_s
        period_mjd = grid_period_s/(24*60*60)
        ts=TSerie()
        empty_mjd_ranges = []
        for i in range(0, len(mjd_grid)-1):
            mjd = mjd_grid[i]
            sub_mts = self.getrange(mjd_grid[i], mjd_grid[i+1])
            if sub_mts:
                sub_mean = sub_mts.mean()
            else:
                sub_mean = None
            if (
                # sub_mts.get_number_of_points() > expected_number_of_points*points_ratio and
                sub_mean is not None
            ):
                if fun=='mean':
                    calc = sub_mean
                if fun=='slope':
                    calc = sub_mts.slope()
                if fun=='slope_s':
                    calc = sub_mts.slope_s()
                if calc is not None:
                    ts.append(
                        mjd=mjd,
                        val=calc,
                        pps=sub_mts.get_number_of_points()
                    )
                    # ts.calc_tab()
                else:
                    ts.append(
                        mjd=mjd,
                        val=none_val,
                        pps=sub_mts.get_number_of_points()
                    )
            else:
                if none_fields:
                    ts.append(
                        mjd=mjd,
                        val=none_val,
                    )
                    # ts.calc_tab()
                else:
                    if get_empty_mjd_ranges:
                        empty_mjd_ranges.append((mjd_grid[i], mjd_grid[i+1]))

        ts.calc_tab()
        out_mts = MTSerie(tseries=[ts])
        out_mts.split(min_gap_s=grid_period_s*1.5)
        # out_mts.rmemptyseries()
        if get_empty_mjd_ranges:
            return out_mts, empty_mjd_ranges
        else:
            return out_mts

    def resample_to_mts_grid(
        self,
        mts,
        grid_period_s,
        fun='mean',
        points_ratio=0.7,
        none_fields=False,
        none_val=None,
    ):
        mjd_array = mts.mjd_tab()
        return self.resample_to_mjd_array(
            mjd_grid = mjd_array,
            grid_period_s=grid_period_s,
            fun=fun,
            points_ratio=points_ratio,
            none_fields=none_fields,
            none_val=none_val,
        )

    def get_sample_period_s(self):
        """
        Returns the average sample period in seconds.
        Based on the length of the time series and the number of points.
        """

        time = self.getTotalTimeWithoutGaps()
        points = self.get_number_of_points()
        return mjd2s*time/points

    def get_number_of_points(self):
        num_of_points = 0
        for x in self.dtab:
            num_of_points += len(x.mjd_tab)
        return num_of_points
    
    def slope(self):
        number_of_points = self.get_number_of_points()
        if number_of_points < 2:
            return np.nan
        slope, intercept, r_value, p_value, str_err = linregress(
            self.mjd_tab(), self.val_tab()
        )
        return slope

    def slope_s(self):
        return self.slope()/(24*60*60)
    
    def time_diff_to_freq_diff(self):
        for x in self.dtab:
            x.time_diff_to_freq_diff()

    def add_val_offset_from_mts(self, mts):
        from_mjd = mts.dtab[0].mjd_tab[0]
        to_mjd = mts.dtab[-1].mjd_tab[-1]
        first_tab, first_index = self.mjd2tabNoandindex(from_mjd)
        last_tab, last_index = self.mjd2tabNoandindex(to_mjd)
        if last_index is None:
            last_index = len(self.dtab[last_tab].mjd_tab)-1
        if first_index is not None:
            first_index = first_index+1
        if (first_tab == last_tab and first_index is None):
            return self
        it = first_tab
        ii = first_index
        while (it<=last_tab or ii<=last_index):
            logging.info(
                f"{self.dtab[it].val_tab[ii]}, {mts.mjd2val(self.dtab[it].mjd_tab[ii])}"
            )
            if mts.mjd2val(self.dtab[it].mjd_tab[ii]) is not None:
                self.dtab[it].val_tab[ii]+=mts.mjd2val(self.dtab[it].mjd_tab[ii])
            if ii==len(self.dtab[it].mjd_tab)-1:
                ii=0
                it+=1
            else:
                ii+=1
    
    def add_sin(self, amplitude=0, omega=0):
        """
        Add amplitude*sin(omega*t) to existing data
        """

        for ts in self.dtab:
            ts.add_sin(amplitude=amplitude, omega=omega)


    def _compute_mts_stats_from_segments(self, segments: list[dict]) -> dict:
        # ---- X (time) ----
        x_min = min(seg["stats"]["x"]["min_mjd"] for seg in segments)
        x_max = max(seg["stats"]["x"]["max_mjd"] for seg in segments)
        span_s = (x_max - x_min) * mjd2s

        # ---- Y ----
        y_min = min(seg["stats"]["y"]["min"] for seg in segments)
        y_max = max(seg["stats"]["y"]["max"] for seg in segments)

        return {
            "x": {
                "min_mjd": x_min,
                "max_mjd": x_max,
                "span_s": float(span_s),
            },
            "y": {
                "min": y_min,
                "max": y_max,
            },
            "n_segments": len(segments),
        }


    def to_dict(self, *, include_stats: bool = True) -> dict:
        segments = [ts.to_dict(include_stats=include_stats) for ts in self.dtab]

        d = {
            "schema": "timanda-tsplot",
            "version": 1,
            "type": "MTS",
            "use_flags": self.use_flags,
            "segments": segments,
        }

        if include_stats:
            d["stats"] = self._compute_mts_stats_from_segments(segments)

        return d


    def to_json(self) -> str:
        """
        Minimal JSON export (v1).
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    def dump_npz(
            self,
            path,
            *,
            compress=False,
            meta=None,
            overwrite=True,
        ):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            return path

        save_fn = np.savez_compressed if compress else np.savez

        payload = {
            "ts_count": np.array(len(self.dtab), dtype=np.int64),
            "mts_use_flags": np.array(self.use_flags, dtype=np.bool_),
            "meta_txt": np.array(
                json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta else "",
                dtype=np.str_,
            ),
        }

        for i, ts in enumerate(self.dtab):
            payload.update(ts.to_npz_payload(prefix=f"ts{i}_"))

        save_fn(path, **payload)
        return path
    
    def append_npz(self, path, *, sort_after=False):

        path = Path(path)

        with np.load(path, allow_pickle=False) as z:

            # jeśli plik zawiera mts_use_flags to synchronizujemy
            if "mts_use_flags" in z:
                self.use_flags = bool(z["mts_use_flags"])

            ts_count = int(z["ts_count"])

            for i in range(ts_count):

                use_flags = bool(z[f"ts{i}_use_flags"]) if f"ts{i}_use_flags" in z else False
                flags = z[f"ts{i}_flags"] if f"ts{i}_flags" in z else None

                ts = TSerie(
                    label=str(z[f"ts{i}_label"]),
                    mjd=z[f"ts{i}_mjd"].tolist(),
                    val=z[f"ts{i}_val"].tolist(),
                    pps=z[f"ts{i}_pps"].tolist(),
                    flags=flags.tolist() if flags is not None else None,
                    use_flags=use_flags,
                )

                self.add_TSerie(ts)

        if sort_after:
            self.dtab.sort(
                key=lambda ts: ts.mjd_tab[0] if len(ts.mjd_tab) else float("inf")
            )

        return ts_count

    # def to_npz_payload(self, prefix=""):
    #     payload = {
    #         f"{prefix}ts_count": np.array(len(self.dtab), dtype=np.int64),
    #         f"{prefix}use_flags": np.array(self.use_flags, dtype=np.bool_),
    #     }

    #     for i, ts in enumerate(self.dtab):
    #         payload.update(ts.to_npz_payload(prefix=f"{prefix}ts{i}_"))

    #     return payload


    def to_npz_payload(self, prefix=""):
        payload = {
            f"{prefix}ts_count": np.array(len(self.dtab), dtype=np.int64),
            f"{prefix}use_flags": np.array(self.use_flags, dtype=np.bool_),
            f"{prefix}label": np.array(self.label if self.label is not None else "", dtype=np.str_),
            f"{prefix}plot_label": np.array(self.plot_label if self.plot_label is not None else "", dtype=np.str_),
            f"{prefix}plot_ref_val": np.array(self.plot_ref_val, dtype=np.float64),
            f"{prefix}color": np.array(self.color if self.color is not None else "green", dtype=np.str_),
        }

        for i, ts in enumerate(self.dtab):
            payload.update(ts.to_npz_payload(prefix=f"{prefix}ts{i}_"))

        return payload


    @classmethod
    def from_npz_payload(cls, z, prefix=""):
        from timanda.tserie import TSerie

        mts_use_flags = bool(z[f"{prefix}use_flags"]) if f"{prefix}use_flags" in z else False
        mts_label = str(z[f"{prefix}label"]) if f"{prefix}label" in z else ""
        mts_plot_label = str(z[f"{prefix}plot_label"]) if f"{prefix}plot_label" in z else ""
        mts_plot_ref_val = float(z[f"{prefix}plot_ref_val"]) if f"{prefix}plot_ref_val" in z else 0.0
        mts_color = str(z[f"{prefix}color"]) if f"{prefix}color" in z else "green"

        mts = cls(
            label=mts_label,
            color=mts_color,
            plot_label=mts_plot_label,
            plot_ref_val=mts_plot_ref_val,
            use_flags=mts_use_flags,
        )

        ts_count = int(z[f"{prefix}ts_count"])

        for j in range(ts_count):
            ts_prefix = f"{prefix}ts{j}_"

            mjd = z[f"{ts_prefix}mjd"]
            val = z[f"{ts_prefix}val"]
            pps = z[f"{ts_prefix}pps"]
            label = str(z[f"{ts_prefix}label"])
            ts_use_flags = bool(z[f"{ts_prefix}use_flags"]) if f"{ts_prefix}use_flags" in z else False
            flags = z[f"{ts_prefix}flags"] if f"{ts_prefix}flags" in z else None

            ts = TSerie(
                label=label,
                mjd=mjd.tolist(),
                val=val.tolist(),
                pps=pps.tolist(),
                flags=flags.tolist() if flags is not None else None,
                use_flags=ts_use_flags,
            )

            mts.add_TSerie(ts)

        return mts

    def set_flags_in_range(self, from_mjd, to_mjd, flag_value):
        """
        Sets flag_value for all points in the given MJD range.
        """
        if from_mjd > to_mjd:
            from_mjd, to_mjd = to_mjd, from_mjd

        self.use_flags = True

        for ts in self.dtab:
            ts.set_flags_in_range(from_mjd, to_mjd, flag_value)
    
    
    def set_flags_in_area(self, x1, y1, x2, y2, flag_value):
        """
        Sets flag_value for all points in the given rectangular area.
        """
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        self.use_flags = True

        for ts in self.dtab:
            ts.set_flags_in_area(x1, y1, x2, y2, flag_value)


    def flag_filter(self, allowed_flag: int = 1):
        """
        Keeps only points with flags in allowed_flags. Removes all other points.
        """
        for ts in self.dtab:
            ts.flag_filter(allowed_flag=allowed_flag)


    def merged_with(self, other: "MTSerie", *, sort_segments: bool = True) -> "MTSerie":
        """
        Zwraca nowe MTSerie będące sklejeniem self i other.

        Na start:
        - kopiuje wszystkie TSerie z self
        - kopiuje wszystkie TSerie z other
        - opcjonalnie sortuje segmenty po pierwszym MJD
        """
        if not isinstance(other, MTSerie):
            raise TypeError(f"Expected MTSerie, got {type(other).__name__}")

        out = MTSerie(
            label=self.label or other.label,
            color=self.color,
            plot_label=self.plot_label or other.plot_label,
            plot_ref_val=self.plot_ref_val,
            use_flags=self.use_flags or other.use_flags,
        )

        for ts in self.dtab:
            out.add_TSerie(ts.copy())

        for ts in other.dtab:
            out.add_TSerie(ts.copy())

        if sort_segments:
            out.dtab.sort(
                key=lambda ts: ts.mjd_tab[0] if len(ts.mjd_tab) > 0 else float("inf")
            )

        return out


    def extend_from(self, other: "MTSerie", *, sort_segments: bool = True) -> None:
        """
        Dokleja segmenty z other do self.
        """
        if not isinstance(other, MTSerie):
            raise TypeError(f"Expected MTSerie, got {type(other).__name__}")

        self.use_flags = self.use_flags or other.use_flags

        if not self.label:
            self.label = other.label
        if not self.plot_label:
            self.plot_label = other.plot_label

        for ts in other.dtab:
            self.add_TSerie(ts.copy())

        if sort_segments:
            self.dtab.sort(
                key=lambda ts: ts.mjd_tab[0] if len(ts.mjd_tab) > 0 else float("inf")
            )