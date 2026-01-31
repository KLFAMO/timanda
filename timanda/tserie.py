from astropy.time import Time
from decimal import Decimal as D
from decimal import getcontext
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
import allantools as al
from astropy.convolution import Gaussian1DKernel, convolve
import json

logging.basicConfig(
    level=logging.INFO, format='%(levelname)s - timanda - %(message)s'
)

mjd2s = 24*60*60
s2mjd = 1./mjd2s

class TSerie:
    """
    Time series class
    TSerie is a set of (mjd, val) pairs represnted as continuous lists
    Attributes:
    label - label of the series
    mjd_tab - numpy array of mjd values
    val_tab - numpy array of values
    pps_tab - numpy array of points per sample values, used when resampling
    s_tab - numpy array of seconds from the start of the series
    """

    def __init__(
            self, label: str = '', mjd: list[float] = None,
            val: list[float] = None, pps: list[int] = None
        ):
        self.label = label
        self.mjd_tab = np.array(mjd, dtype=float) if mjd is not None else np.empty(0, dtype=float)
        self.val_tab = np.array(val, dtype=float) if val is not None else np.empty(0, dtype=float)
        mjd_len = len(self.mjd_tab)
        if len(self.val_tab) != mjd_len:
            raise ValueError("Length of mjd and val must be equal")

        # points per sample, used when resampling
        if pps is None:
            self.pps_tab = np.ones(len(self.mjd_tab), dtype=int)
        else:
            if len(pps) != len(self.mjd_tab):
                raise ValueError("Length of pps must be equal to the length of mjd and val")
            self.pps_tab = np.array(pps)

        self.len = len(self.mjd_tab)
        self.calc_tab()

    def __str__(self):
        if len(self.mjd_tab) == 0:
            return '\tEmpty'
        
        self.calc_tab()
        s = f'TSerie:\tlabel: {self.label}\tlength: {self.len}\tlen_mjd: {self.len_mjd:.6f}\n'
        s += self._format_data()
        return s
    
    def _format_data(self):
        if self.len <= 10:
            return ''.join(
                f'\t{self.mjd_tab[i]:.6f}\t{self.val_tab[i]:f}\t{self.pps_tab[i]:f}\n'
                for i in range(self.len)
            )
        else:
            first_part = ''.join(
                f'\t{self.mjd_tab[i]:.6f}\t{self.val_tab[i]:f}\t{self.pps_tab[i]:f}\n'
                for i in range(5)
            )
            last_part = ''.join(
                f'\t{self.mjd_tab[i]:.6f}\t{self.val_tab[i]:f}\t{self.pps_tab[i]:f}\n'
                for i in range(self.len - 5, self.len)
            )
            return first_part + '\t...\n' + last_part

    def _apply_operation(self, b, operation):
        if isinstance(b, (int, float)):
            val = operation(self.val_tab, b)
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, TSerie):
            print(f'TSerie {operation.__name__} TSerie is not supported yet')
            return None
        else:
            return None

    def __add__(self, b):
        return self._apply_operation(b, np.add)

    def __sub__(self, b):
        return self._apply_operation(b, np.subtract)

    def __mul__(self, b):
        return self._apply_operation(b, np.multiply)

    def __truediv__(self, b):
        return self._apply_operation(b, np.divide)
    
    def rm_nans(self):
        not_nan_indexes = np.where(~np.isnan(self.val_tab))[0]
        self.mjd_tab = self.mjd_tab[not_nan_indexes]
        self.val_tab = self.val_tab[not_nan_indexes]
        self.pps_tab = self.pps_tab[not_nan_indexes]

    def calc_tab(self):
        """
        Calculates derived attributes of the series
        
        Derived attributes:
        len - length of the series (number of points)
        isempty - 1 if the series is empty, 0 otherwise
        mjd_start - starting MJD of the series
        mjd_stop - ending MJD of the series
        mean - mean value of the series
        s_tab - numpy array of seconds from the start of the series
        len_mjd - length of the series in MJD
        len_s - length of the series in seconds
        """
        self.rm_nans()
        self.len = len(self.mjd_tab)
        if self.len > 0:
            self.isempty = 0
            self.mjd_start = self.mjd_tab[0]
            self.mjd_stop = self.mjd_tab[-1]
            filtered = [x for x in self.val_tab if x is not None]
            self.mean_val = np.mean(filtered)
            # self.mean = np.mean(self.val_tab)
        else:
            self.isempty = 1
            self.mjd_start = 0
            self.mjd_stop = 0
            self.mean_val = None
        self.s_tab = (self.mjd_tab - self.mjd_start)*mjd2s
        # self.t_type = 't_type'
        self.len_mjd = self.mjd_stop-self.mjd_start
        self.len_s = self.len_mjd*mjd2s

    def len(self):
        return len(self.mjd_tab)

    def cp(self):
        out = TSerie(label=self.label+'_cp',
                     mjd=self.mjd_tab,
                     val=self.val_tab)
        return out

    def mean(self, decimal=False, decimal_out=False, use_pps=False):
        """
        Returns mean value of the series

        Params:
            decimal (bool): if True, uses Decimal for calculations
            decimal_out (bool): if True, returns Decimal, otherwise float
            use_pps (bool): if True, uses pps_tab for weighted mean calculation
        
        Returns:
            float | Decimal | None: The mean value of the series, or None if the series is empty.

        Note:
            Decimal calculations helps only partially, because input values are float.
        """
        if len(self.val_tab) == 0:  # TODO: check if self.len == 0  is ok
            return None

        if use_pps:
            return self.mean_use_pps(decimal=decimal, decimal_out=decimal_out)
        
        if decimal:
            getcontext().prec = 21
            darr = [D(x) for x in self.val_tab]
            mean_value = sum(darr) / len(darr)
            return mean_value if decimal_out else float(mean_value)
            
        return np.mean(self.val_tab)

    
    def mean_use_pps(self, decimal=False, decimal_out=False):
        """
        Returns mean value of the series using pps_tab for weighted mean calculation
        Params:
            decimal (bool): if True, uses Decimal for calculations
            decimal_out (bool): if True, returns Decimal, otherwise float
        """
        if len(self.val_tab) == 0:
            return None
        
        if decimal:
            getcontext().prec = 21
            d_pps_tab = [D(str(x)) for x in self.pps_tab]
            d_val_tab = [D(str(x)) for x in self.val_tab]
            d_pps_val_tab = [x*y for x, y in zip(d_pps_tab, d_val_tab)]
            d_sum_points = sum(d_pps_tab)
            d_mean = sum(d_pps_val_tab)/d_sum_points
            if decimal_out:
                return d_mean, d_sum_points
            else:
                return float(d_mean), int(d_sum_points)
        
        pps_val_tab = self.pps_tab*self.val_tab
        sum_points = np.sum(self.pps_tab)
        mean_out = np.sum(pps_val_tab)/sum_points
        return mean_out, int(sum_points)
        
    def max_val(self):
        """
        Returns the maximum value in val_tab.

        Returns:
            float | None: The maximum value in val_tab, or None if val_tab is empty.
        """
        if len(self.val_tab) == 0:
            return None  # Return None if the array is empty
        return np.nanmax(self.val_tab)  # Use nanmax to ignore NaN values

    def rm_dc(self):
        """
        Removes the DC (mean) component from val_tab.

        This method subtracts the mean value of val_tab from each element in val_tab.
        If val_tab is empty, the method does nothing.

        Note:
            NaN values in val_tab are ignored when calculating the mean.

        Returns:
            None
        """
        if len(self.val_tab) == 0:
            return  # Do nothing if val_tab is empty
        
        self.val_tab = self.val_tab - self.mean()

    def rm_drift(self):
        """
        Removes the linear drift from val_tab.

        This method fits a linear trend to the data in val_tab as a function of mjd_tab
        and subtracts the fitted line from val_tab. If mjd_tab or val_tab are empty,
        the method does nothing.

        Returns:
            None
        """
        if len(self.val_tab) == 0:
            return  # Do nothing if data is empty

        try:
            fit = np.polyfit(self.mjd_tab, self.val_tab, 1)
            self.val_tab = self.val_tab - (self.mjd_tab*fit[0]+fit[1])
        except ValueError as e:
            logging.error(f"ValueError in rm_drift: {e}")
        except Exception as e:
            logging.error(f"Unexpected error in rm_drift: {e}")

    def split(self, min_gap_s=8):
        """
        Splits the time series into segments based on gaps in time.

        This method identifies gaps in `s_tab` larger than `min_gap_s` seconds
        and splits the time series into multiple `TSerie` objects.

        Args:
            min_gap_s (float): Minimum gap in seconds to split the series.

        Returns:
            list[TSerie]: A list of `TSerie` objects representing the split segments.
        """
        if self.len == 0:
            return []
        out_tab = []
        tab_i = []
        tab_i.append(0)
        for i in range(0, len(self.s_tab)-1):
            if self.s_tab[i+1]-self.s_tab[i] > min_gap_s:
                tab_i.append(i+1)
        tab_i.append(len(self.s_tab)-1)
        for j in range(0, len(tab_i)-1):
            out_tab.append(TSerie(
                mjd=self.mjd_tab[tab_i[j]:tab_i[j+1]],
                val=self.val_tab[tab_i[j]:tab_i[j+1]],
                pps=self.pps_tab[tab_i[j]:tab_i[j+1]],
            ))
        return out_tab

    def append(self, mjd, val, pps=None):
        self.mjd_tab = np.append(self.mjd_tab, mjd)
        self.val_tab = np.append(self.val_tab, val)
        if pps:
            self.pps_tab = np.append(self.pps_tab, pps)
        else:
            self.pps_tab = np.append(self.pps_tab, 1)

    def last(self):
        out = TSerie()
        out.append(self.mjd_tab[-1], self.val_tab[-1], self.pps_tab[-1])
        return out

    def mjd2index(self, mjd, init_index=None):
        """Returns index of tab corresponding to mjd
        in: mjd, init_index - initial index for searching
        out: index of the nearest mjd <= input_mjd
             None if mjd is out of table
        """
        # self.__str__() # removed for speedup, but not sure if not necessary
        if mjd < self.mjd_start or mjd > self.mjd_stop:
            setattr(self, "_last_mjd_index", None)
            return None

        if init_index is not None:
            N = init_index
        else:
            # try to use last cached index
            cached = getattr(self, "_last_mjd_index", None)
            if cached is not None:
                N = cached
            else:
                if self.len_mjd == 0:
                    return None
                N = int((self.len / self.len_mjd) * (mjd - self.mjd_start))
        
        if N < 0:
            N = 0
        if N >= self.len:
            N = self.len-1
        while True:
            if self.mjd_tab[N] > mjd:
                N = N-1
            else:
                if N+1 >= self.len or self.mjd_tab[N+1] > mjd:
                    setattr(self, "_last_mjd_index", N)
                    return N
                else:
                    N = N+1

    def mjd2val(self, mjd, init_index=None):
        return self.val_tab[self.mjd2index(mjd, init_index=init_index)]

    def getrange(self, fmjd, tmjd, fmjd_init=None, tmjd_init=None):
        if (fmjd > self.mjd_stop or tmjd < self.mjd_start):
            return None
        if fmjd < self.mjd_start:
            fmjd = self.mjd_start
        if tmjd > self.mjd_stop:
            tmjd = self.mjd_stop
        fN = self.mjd2index(fmjd, init_index=fmjd_init)
        tN = self.mjd2index(tmjd, init_index=tmjd_init)
        if tN is None:
            return None
        s = TSerie(
            mjd=self.mjd_tab[fN:tN+1],
            val=self.val_tab[fN:tN+1]
        )
        if s.mjd_tab[0] < fmjd:
            s.rm_first(1)
        if len(s.mjd_tab) == 0:
            return None
        if s.mjd_tab[-1] > tmjd:
            s.rm_last(1)
        return s

    def rmrange(self, fmjd, tmjd):
        if (fmjd > self.mjd_stop or tmjd < self.mjd_start):
            return (self, 0)
        if (fmjd <= self.mjd_start and tmjd >= self.mjd_stop):
            return (None, 1)
        fN = self.mjd2index(fmjd)
        tN = self.mjd2index(tmjd)
        if (fmjd <= self.mjd_start and tmjd < self.mjd_stop):
            return (TSerie(
                mjd=self.mjd_tab[(tN+1):],
                val=self.val_tab[(tN+1):],
                pps=self.pps_tab[(tN+1):],
                ),
                2
            )
        if (fmjd > self.mjd_start and tmjd >= self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            return (TSerie(
                mjd=self.mjd_tab[:fN],
                val=self.val_tab[:fN],
                pps=self.pps_tab[:fN],
                ),
                3)
        if (fmjd > self.mjd_start and tmjd < self.mjd_stop):
            if fmjd != self.mjd_tab[fN]:
                fN = fN+1
            left = TSerie(
                mjd=self.mjd_tab[:fN],
                val=self.val_tab[:fN],
                pps=self.pps_tab[:fN],
            )
            right = TSerie(
                mjd=self.mjd_tab[(tN+1):],
                val=self.val_tab[(tN+1):],
                pps=self.pps_tab[(tN+1):],
            )
            return ([left, right], 4)
    
    def rm_indexes(self, indexes):
        if indexes is None:
            return
        self.mjd_tab = np.delete(self.mjd_tab, indexes)
        self.val_tab = np.delete(self.val_tab, indexes)
        self.pps_tab = np.delete(self.pps_tab, indexes)
        self.len = len(self.mjd_tab)

    def rm_outlayers_singledelta(self, max_delta):
        self.calc_tab()
        for i in range(1, self.len):
            if abs(self.val_tab[i] - self.val_tab[i-1]) > max_delta:
                print('outlayer detected')
                self.val_tab[i] = self.val_tab[i-1]

    def rmOutlayersOfTarget(self, target, maxDifference, get_indexes_only=False):
        indexes_to_delete = []
        if target is None:
            target = self.mean()
        if maxDifference is None:
            maxDifference = 5*self.std()
        i = 0
        self.len=len(self.mjd_tab)
        while i < self.len:
            if abs(self.val_tab[i]-target) > maxDifference:
                indexes_to_delete.append(i)
            i = i+1
        if not get_indexes_only:
            self.rm_indexes(indexes_to_delete)
        return indexes_to_delete
    
    def rm_value(self, value, get_indexes_only=False):
        indexes_to_delete = []
        i = 0
        while i < self.len:
            if self.val_tab[i] == value:
                indexes_to_delete.append(i)
            i = i+1
        if not get_indexes_only:
            self.rm_indexes(indexes_to_delete)
        return indexes_to_delete

    def time_shift(self, sec):
        self.mjd_tab = self.mjd_tab+sec/(24*60*60)
        self.calc_tab()

    def show(self):
        N = 5
        if len(self.mjd_tab) > 2*N:
            for x in range(0, 5):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
            print('...')
            for x in range(self.length-6, self.length-1):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
        elif len(self.mjd_tab) > 0:
            for x in range(0, 2*N-1):
                print(self.mjd_tab[x], self.val_tab[x], self.pps_tab[x])
        else:
            print('Empty')

    def plot(self):
        plt.figure()
        plt.title(self.label)
        plt.plot(self.mjd_tab, self.val_tab)
        # plt.scatter(self.mjd_tab, self.val_tab*0, s=700, marker="|")
        plt.show()

    def plot_pqg(self, minusmjd):
        w = pg.plot(self.mjd_tab-minusmjd, self.val_tab)
        w.setBackground('w')
        w.setTitle(self.label)
        w.show()
        return w

    def plot_pqg_widget(self, minusmjd=0):
        self.widget = pg.PlotWidget()
        self.widget.plot(self.mjd_tab-minusmjd, self.val_tab)
        self.widget.setBackground('w')
        self.widget.setTitle(self.label)
        return self.widget

    def plot_nice(self, fig=0, ax_site='left'):
        if fig == 0:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.mjd_tab, self.val_tab)
        plt.show()
        return fig, ax

    def plot_allan_pqg(self, atom='88Sr'):
        if atom == '88Sr':
            fabs = 400e12
        w = pg.PlotWidget()
        t = np.power(10, np.arange(1, int(np.log10(self.len_s))+0.1, 0.1))
        # t=np.logspace(np.log10(20),np.log10(self.len_s),50)
        y = self.val_tab
        r = self.len_s/self.len
        (t2, ad, ade, adn) = al.adev(y, rate=r, data_type="freq", taus=t)
        w.plot(t2, ad/fabs, symbol='o')
        w.showGrid(True, True)
        w.setLogMode(True, True)
        w.setBackground('w')
        w.setTitle(self.label+'_ADEV')
        return w

    def plot_allan(self, **kwargs):
        """
        Plots allan deviation using allantools

        Params:
            atom (str): name of atom ('88Sr') used to set fabs if not None
            fabs (num): absolute frequency used to calculate relative values
            out_file_name (str): name of the output file (without extension) with the plot,
            it None - the plot is displayed on screen
            method (str): method of calculation ('adev', 'mdev', 'oadev')

        Return:
            dictionary: with data - keys: 'taus', 'stat', 'stat_err', 'stat_n', 'stat_id'
        """
        out_file_name = kwargs.pop("out_file_name", None)
        a = self._compute_allan(**kwargs)
        b = al.Plot()
        b.plot(a, errorbars=True, grid=True)
        if out_file_name:
            plt.savefig(out_file_name)
            plt.close()
        else:
            b.show()

    def compute_allan(self, **kwargs):
        """
        Computes allan deviation using allantools

        Params:
            atom (str): name of atom ('88Sr') used to set fabs if not None
            fabs (num): absolute frequency used to calculate relative values
            method (str): method of calculation ('adev', 'mdev', 'oadev')

        Return:
            dictionary: with keys 'taus', 'stat', 'stat_err', 'stat_n', 'stat_id'
        """
        a = self._compute_allan(**kwargs)
        return a.out

    def _compute_allan(self, atom=None, fabs=1, method='adev', **kwargs):
        """
        Computes allan deviation using allantools

        Params:
            atom (str): name of atom ('88Sr') used to set fabs if not None
            fabs (num): absolute frequency used to calculate relative values
            method (str): method of calculation ('adev', 'mdev', 'oadev')

        Return:
            allantools object
        """
        if atom == '88Sr':
            fabs = 429228066418012
        # t = np.power(10, np.arange(1, int(np.log10(self.len_s))+0.1, 0.1))
        y = self.val_tab / fabs
        r = self.len_s / self.len
        a = al.Dataset(data=y, rate=r, data_type="freq")
        a.compute(method)
        return a

    def scatter(self):
        plt.scatter(self.mjd_tab, self.val_tab)
        plt.show()

    def rm_first(self, n=1):
        self.val_tab = np.delete(self.val_tab, np.s_[0:n], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[0:n], None)
        self.pps_tab = np.delete(self.pps_tab, np.s_[0:n], None)
        self.calc_tab()

    def rm_last(self, n=1):
        self.val_tab = np.delete(self.val_tab, np.s_[-n:], None)
        self.mjd_tab = np.delete(self.mjd_tab, np.s_[-n:], None)
        self.pps_tab = np.delete(self.pps_tab, np.s_[-n:], None)
        self.calc_tab()

    def gauss_filter(self, stddev=50):
        g = Gaussian1DKernel(stddev=stddev)
        self.val_tab = convolve(self.val_tab, g)
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)

    def high_gauss_filter(self, stddev=50, rm_dc=True):
        if not rm_dc:
            dc = self.mean()
        g = Gaussian1DKernel(stddev=stddev)
        tmp = convolve(self.val_tab, g)
        self.val_tab = self.val_tab - tmp
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)
        if not rm_dc:
            self.val_tab = self.val_tab + dc

    def toMTSerie(self):
        from timanda.mtserie import MTSerie
        return MTSerie(tseries=[self])
    
    def first_mjd(self):
        return self.mjd_tab[0]
    
    def last_mjd(self):
        return self.mjd_tab[-1]
    
    def time_diff_to_freq_diff(self, fill_last_point=True):
        t_mjd=list()
        t_val=list()
        for i in range(0, len(self.mjd_tab)-1):
            delta_mjd_s = (self.mjd_tab[i+1]-self.mjd_tab[i])*(24*60*60)
            f = (self.val_tab[i+1]-self.val_tab[i])/delta_mjd_s
            t_mjd.append(self.mjd_tab[i])
            t_val.append(f)
        if fill_last_point:
            t_val.append(t_mjd[-1])
            t_mjd.append(self.mjd_tab[-1])
        self.mjd_tab=np.array(t_mjd)
        self.val_tab=np.array(t_val)


    def compute_plot_stats(
        self,
        *,
        qy: float = 0.95,
        linthresh_min_y: float | None = None,
        linthresh_max_y: float | None = None,
    ) -> dict:
        """
        Compute simple plot-oriented stats for this TSerie.

        Returns
        -------
        dict:
          {
            "x": {
              "n": int,
              "min_mjd": float,
              "max_mjd": float,
              "span_s": float,
              "dt_s_median": float | None,
              "dt_s_min": float | None
            },
            "y": {
              "n": int,
              "min": float,
              "max": float,
              "center": float,
              "linthresh": float,
              "linear_low": float,
              "linear_high": float,
              "qy": float,
              "spike_fraction": float
            }
          }
        """
        # ---------- X stats (time) ----------
        x = np.asarray(self.mjd_tab, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("TSerie.compute_plot_stats: no finite samples in mjd_tab.")

        x_min_mjd = float(np.min(x))
        x_max_mjd = float(np.max(x))
        span_s = float((x_max_mjd - x_min_mjd) * mjd2s)

        if x.size >= 2:
            xs = np.sort(x)
            dx = np.diff(xs)
            dx = dx[np.isfinite(dx)]
            dx = dx[dx > 0]
            if dx.size > 0:
                dt_s_median = float(np.median(dx) * mjd2s)
                dt_s_min = float(np.min(dx) * mjd2s)
            else:
                dt_s_median = None
                dt_s_min = None
        else:
            dt_s_median = None
            dt_s_min = None

        x_dict = {
            "n": int(x.size),
            "min_mjd": x_min_mjd,
            "max_mjd": x_max_mjd,
            "span_s": span_s,
            "dt_s_median": dt_s_median,
            "dt_s_min": dt_s_min,
        }

        # ---------- Y stats ----------
        y = np.asarray(self.val_tab, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            raise ValueError("TSerie.compute_plot_stats: no finite samples in val_tab.")

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        center = float(np.median(y))
        dev = np.abs(y - center)

        linthresh = float(np.quantile(dev, qy))

        # default minimal linthresh to avoid zero-width band
        if linthresh_min_y is None:
            linthresh_min_y = max(1e-12 * max(1.0, abs(center)), 0.0)

        linthresh = max(linthresh, float(linthresh_min_y))
        if linthresh_max_y is not None:
            linthresh = min(linthresh, float(linthresh_max_y))

        linear_low = float(center - linthresh)
        linear_high = float(center + linthresh)

        spike_fraction = float(np.mean(dev > linthresh))

        y_dict = {
            "n": int(y.size),
            "min": y_min,
            "max": y_max,
            "center": center,
            "linthresh": linthresh,
            "linear_low": linear_low,
            "linear_high": linear_high,
            "qy": float(qy),
            "spike_fraction": spike_fraction,
        }

        return {"x": x_dict, "y": y_dict}
    
    
    def add_sin(self, amplitude=0, omega=0):
        mjd2s = 24*60*60
        for i,v in enumerate(self.val_tab):
            t = (self.mjd_tab[i])*mjd2s
            self.val_tab[i] += amplitude*np.sin(omega*t)


    def to_dict(self, *, include_stats: bool = True) -> dict:
        """
        Minimal dict export of a single TSerie segment.
        Exports raw mjd/val arrays only.
        """
        mjd = list(self.mjd_tab) if self.mjd_tab is not None else []
        val = list(self.val_tab) if self.val_tab is not None else []
        n = min(len(mjd), len(val))
        d = {
            "type": "tserie",
            "mjd": mjd[:n],
            "val": val[:n],
        }
        if include_stats:
            d["stats"] = self.compute_plot_stats()
        return d

    def to_json(self) -> str:
        """
        Minimal JSON export of a single TSerie.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_npz_payload(self, prefix: str = "") -> dict:
        """
        Prepares data for saving to NPZ file.
        Params:
            prefix (str): prefix to add to the keys in the output dictionary
        Returns:
            dict: dictionary with keys:
                f"{prefix}mjd" - numpy array of mjd values
                f"{prefix}val" - numpy array of val values
                f"{prefix}pps" - numpy array of pps values
                f"{prefix}label" - numpy array with a single unicode string (the label)
        """
        mjd = np.asarray(self.mjd_tab, dtype=np.float64)
        val = np.asarray(self.val_tab, dtype=np.float64)

        pps = np.asarray(self.pps_tab, dtype=np.int32) if hasattr(self, "pps_tab") else np.ones(len(mjd), dtype=np.int32)

        # label jako unicode scalar (bez pickla)
        label = np.array(self.label if self.label is not None else "", dtype=np.str_)

        return {
            f"{prefix}mjd": mjd,
            f"{prefix}val": val,
            f"{prefix}pps": pps,
            f"{prefix}label": label,
        }
