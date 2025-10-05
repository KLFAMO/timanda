from astropy.time import Time
from scipy.stats import linregress
import decimal as dec  # TODO: remove after removing alphanorm
from decimal import Decimal as D
from decimal import getcontext
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
import allantools as al
from astropy.convolution import Gaussian1DKernel, convolve

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
    """

    def __init__(
            self, label: str = '', mjd: list[float] = None,
            val: list[float] = None, pps: list[int] = None
        ):
        self.label = label
        self.mjd_tab = np.array(mjd or [])
        self.val_tab = np.array(val or [])
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
            s = '\tEmpty'
        else:
            self.calc_tab()
            s = 'TSerie:\tlabel: %s\tlength: %d\tlen_mjd: %.6f\n' % (
                self.label,
                self.len,
                self.len_mjd
            )
            if self.len <= 10:
                for i in range(0, self.len):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i],
                    )
            else:
                for i in range(0, 5):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i]
                    )
                s = s+'\t...\n'
                for i in range(self.len-5, self.len):
                    s = s+'\t%.6f\t%f\t%f\n' % (
                        self.mjd_tab[i],
                        self.val_tab[i],
                        self.pps_tab[i],
                    )
        return s

    def __add__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab + b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('Adding TSerie to TSerie is  not supported yet')
            return None
        else:
            return None

    def __sub__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab - b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print(' TSerie - TSerie is  not supported yet')
            return None
        else:
            return None

    def __mul__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab * b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('TSerie * TSerie is  not supported yet')
            return None
        else:
            return None

    def __truediv__(self, b):
        if isinstance(b, (int, float)):
            val = self.val_tab / b
            return TSerie(val=val, mjd=self.mjd_tab)
        elif isinstance(b, (TSerie)):
            print('TSerie / TSerie is  not supported yet')
            return None
        else:
            return None

    def calc_tab(self):  # if not empty
        self.len = len(self.mjd_tab)
        if self.len > 0:
            self.isempty = 0
            self.mjd_start = self.mjd_tab[0]
            self.mjd_stop = self.mjd_tab[-1]
            filtered = [x for x in self.val_tab if x is not None]
            self.mean = np.mean(filtered)
        else:
            self.isempty = 1
            self.mjd_start = 0
            self.mjd_stop = 0
        mjd_s = 24*60*60
        self.s_tab = (self.mjd_tab - self.mjd_start)*mjd_s
        self.t_type = 't_type'
        self.len_mjd = self.mjd_stop-self.mjd_start
        self.len_s = self.len_mjd*mjd_s

    def len(self):
        return len(self.mjd_tab)

    def cp(self):
        out = TSerie(label=self.label+'_cp',
                     mjd=self.mjd_tab,
                     val=self.val_tab)
        return out

    def mean_use_pps(self, decimal=False, decimal_out=False):
        if len(self.val_tab) > 0:
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
            else:
                pps_val_tab = self.pps_tab*self.val_tab
                sum_points = sum(pps_val_tab)
                mean_out = sum(pps_val_tab)/float(sum_points)
                return mean_out, int(sum_points)
        else:
            return None

    def mean(self, decimal=False, decimal_out=False, use_pps=False):
        if use_pps:
            return self.mean_use_pps(decimal=decimal, decimal_out=decimal_out)
        if len(self.val_tab) > 0:
            if decimal:
                getcontext().prec = 21
                darr = [D(x) for x in self.val_tab]
                if decimal_out:
                    return sum(darr)/len(darr)
                else:
                    return float(sum(darr)/len(darr))
            else:
                return np.mean(self.val_tab)
        else:
            return None
        
    def max_val(self):
        return np.max(self.val_tab)

    def rm_dc(self):
        self.val_tab = self.val_tab - self.mean

    def rm_drift(self):
        try:
            fit = np.polyfit(self.mjd_tab, self.val_tab, 1)
            self.val_tab = (
                self.val_tab - (self.mjd_tab*fit[0]+fit[1])
            )
        except:  # TODO: add exception name here
            pass
            # print('rm drift problem')

    def split(self, min_gap_s=8):
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
        self.__str__()
        if mjd < self.mjd_start or mjd > self.mjd_stop:
            return None
        if init_index is None:
            if self.len_mjd == 0:
                return None
            N = int((self.len/self.len_mjd)*(mjd-self.mjd_start))
        else:
            N = init_index
        if N < 0:
            N = 0
        if N >= self.len:
            N = self.len-1
        while 1:
            if self.mjd_tab[N] > mjd:
                N = N-1
            else:
                if N+1 >= self.len or self.mjd_tab[N+1] > mjd:
                    return N
                else:
                    N = N+1

    def mjd2val(self, mjd, init_index=None):
        return self.val_tab[self.mjd2index(mjd, init_index=init_index)]

    def getrange(self, fmjd, tmjd):
        if (fmjd > self.mjd_stop or tmjd < self.mjd_start):
            return None
        if fmjd < self.mjd_start:
            fmjd = self.mjd_start
        if tmjd > self.mjd_stop:
            tmjd = self.mjd_stop
        fN = self.mjd2index(fmjd)
        tN = self.mjd2index(tmjd)
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
            dc = self.mean
        g = Gaussian1DKernel(stddev=stddev)
        tmp = convolve(self.val_tab, g)
        self.val_tab = self.val_tab - tmp
        self.rm_first(stddev*5)
        self.rm_last(stddev*5)
        if not rm_dc:
            self.val_tab = self.val_tab + dc

    def toMTSerie(self):
        from timanda.mtserie import MTSerie
        return MTSerie(TSerie=self)
    
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
    
    def add_sin(self, amplitude=0, omega=0):
        mjd2s = 24*60*60
        for i,v in enumerate(self.val_tab):
            t = (self.mjd_tab[i])*mjd2s
            self.val_tab[i] += amplitude*np.sin(omega*t)
             

class MGserie:
    def __init__(self, ser1, ser2, grid_s, fmjd=None, tmjd=None, grid_mode=1):
        self.dtab = []
        self.grid_s = grid_s
        self.grid_mjd = grid_s/(24*60*60)
        self.mjd_min = min(ser1.mjd_tab()[0], ser2.mjd_tab()[0])
        self.mjd_max = max(ser1.mjd_tab()[-1], ser2.mjd_tab()[-1])
        if fmjd is not None:
            self.mjd_min = max(self.mjd_min, fmjd)
        if tmjd is not None:
            self.mjd_max = min(self.mjd_max, tmjd)

        grid_tab = np.arange(self.mjd_min, self.mjd_max, self.grid_mjd)
        self.dtab = np.zeros((3, len(grid_tab)))
        for ind, mjd in enumerate(grid_tab):
            self.dtab[1][ind] = ser1.mjd2val(mjd, mode=grid_mode)
            self.dtab[2][ind] = ser2.mjd2val(mjd, mode=grid_mode)
        self.dtab[0] = grid_tab

    def corr(self, fmjd, tmjd, shift_s=None, norm=True):
        fi1 = self.mjd2index(fmjd)
        ti1 = self.mjd2index(tmjd)
        if shift_s is None:
            fi2 = fi1
            ti2 = ti1
        else:
            try:
                fi2 = fi1 + int(shift_s/self.grid_s)
                ti2 = ti1 + int(shift_s/self.grid_s)
            except:
                return np.nan
            if fi2 < 0 or ti2 > len(self.dtab[0])-1:
                return np.nan

        d1 = self.dtab[1][fi1:ti1]
        d2 = self.dtab[2][fi2:ti2]
        if len(d1) == len(d2):
            if norm is True:
                return np.correlate(d1, d2, 'valid')[0]/len(d1)
            else:
                return np.correlate(d1, d2, 'valid')[0]
        else:
            return np.nan

    def plot(self, ax=None, show=True, color='g'):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        for x in [1, 2]:
            d = np.array([
                [self.dtab[0][i], self.dtab[x][i]] for
                i in range(0, len(self.dtab[0]))
                # if not np.isnan(self.dtab[x][i])
            ]).transpose()
            ax.plot(d[0], d[1], color=color)
        if show is True:
            plt.show()
        return ax

    def mjd2index(self, mjd, init_index=None):
        leni = len(self.dtab[0])
        len_mjd = self.dtab[0][-1]-self.dtab[0][0]
        if init_index is None:
            N = int((leni/len_mjd)*(mjd-self.dtab[0][0]))
        else:
            N = init_index
        if (mjd < self.dtab[0][0] or mjd > self.dtab[0][-1]):
            return None
        while 1:
            if self.dtab[0][N] > mjd:
                N = N-1
            else:
                if self.dtab[0][N+1] > mjd:
                    return N
                else:
                    N = N+1


class GTserie:
    """
    Class storing multiple time series of type MTserie
    """
    
    def __init__(self, name):
        self.name = name
        self.mts_dict = dict()
        self.number_of_mts = 0
        self.mjd_groups = dict()
    
    def __str__(self):
        return (
            f"{self.name}:\n"
            f"\tnumber of series: {self.number_of_mts}"
        )

    def append_mtserie(self, mts_name, mts, mjd_group=''):
        """
        Append MTSerie to GTserie

        Params:
            mts_name (str): name of MTSerie
            mts (MTSerie): data
            time_group (str): name of mjd group
        """
        self.mts_dict[mts_name] = mts
        self.number_of_mts = len(self.mts_dict)
        self.mjd_groups[mts_name] = mjd_group
    
    def import_data_rocit_oc(
        self,
        info,
        name='umk',
        headers = ['date', 'time', 'frac_freq', 'confidence', 'systematics'],
    ):
        df = import_data_to_df_rocit_oc(info=info, name=name, headers=headers)
        self.append_df_as_mtseries(df, name, columns=['frac_freq', 'confidence', 'systematics'])

    def import_data_rocit_gnss(self, path='./Data_storage/sn112-nmij.dat', name='gnss'):
        df = import_data_to_df_rocit_gnss(path=path)
        self.append_df_as_mtseries(df, name, ['delta_t_ns'])
        
    def append_df_as_mtseries(self, df, name, columns, mjd_name='mjd'):
        from timanda.mtserie import MTSerie
        for column in columns:
            self.append_mtserie(
                mts_name=name+'_'+column,
                mts=MTSerie(
                    label=name+'_'+column,
                    TSerie=TSerie(mjd=df[mjd_name].to_numpy(), val=df[column].to_numpy())
                ),
                mjd_group=name+'_mjd',
            )

    def get_mtss_names(self):
        return self.mts_dict

    def first_mjd(self):
        """
        Returns the first MJD of GTserie.
        """

        first_mtss_mjds = [self.mts_dict[mts].first_mjd() for mts in self.mts_dict]
        print(first_mtss_mjds)
        return min(first_mtss_mjds)

    def last_mjd(self):
        """
        Returns the last MJD of GTserie.
        """

        last_mtss_mjds = [self.mts_dict[mts].last_mjd() for mts in self.mts_dict]
        return max(last_mtss_mjds)
    
    def print_all_mts(self):
        for a in self.mts_dict:
            print(self.mts_dict[a])
    
    def plot_mts_one_by_one(self):
        for a in self.mts_dict:
            self.mts_dict[a].plot()
    
    def plot_mts(self, mts_name):
        self.mts_dict[mts_name].plot()

    def plot(self, fig=None, axs=None, figsize=(7, 7), mts_names=None, show=1, zorder=1):
        if not mts_names:
            mts_names = self.mts_dict
        fig, axs = plt.subplots(len(mts_names),1,  constrained_layout=True, sharex=True, figsize=figsize)
        for i, mts_name in enumerate(mts_names):
            self.mts_dict[mts_name].plot(ax=axs[i], show=0, zorder=zorder)
            axs[i].grid(True)
            if self.mts_dict[mts_name].plot_label == '':
                self.mts_dict[mts_name].plot_label = mts_name
            axs[i].set_ylabel(self.mts_dict[mts_name].plot_label)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axs

    def get_mtss_from_mjd_group(self, mjd_group, exclude=None):
        keys = [key for key, value in self.mjd_groups.items() if value == mjd_group]
        if exclude in keys:
            keys.remove(exclude)
        return keys

    def rm_indexes_from_mjd_group(self, indexes, mjd_group, exclude):
        mtss = self.get_mtss_from_mjd_group(mjd_group=mjd_group, exclude=exclude)
        for mts in mtss:
            self.mts_dict[mts].rm_indexes(indexes)

    def rm_outlayers(self, mts_name, target=None, maxdiff=None):
        """
        Removes outlayers from MTSerie and from other MTseries in the same mjd group.
        """
        mts = self.mts_dict[mts_name]
        indexes_to_delete_iterated = mts.rmoutlayers(
            target=target,
            maxdiff=maxdiff,
        )
        mjd_group = self.mjd_groups[mts_name]
        for indexes_to_delete in indexes_to_delete_iterated:
            self.rm_indexes_from_mjd_group(
                indexes=indexes_to_delete,
                mjd_group=mjd_group,
                exclude=mts_name
            )
    
    def rm_value(self, mts_name, value):
        mts = self.mts_dict[mts_name]
        indexes_in_mts_to_delete = mts.rm_value(value, get_indexes_only=True)
        mjd_group = self.mjd_groups[mts_name]
        self.rm_indexes_from_mjd_group(
            indexes=indexes_in_mts_to_delete,
            mjd_group=mjd_group,
            exclude=None
        )

    def rm_range(self, from_mjd, to_mjd):
        for a in self.mts_dict:
            self.mts_dict[a].rmrange(from_mjd, to_mjd)

    def get_range(self, from_mjd, to_mjd):
        """
        Returns the range of all MTseries in the given time period.
        """
        self.rm_range(self.first_mjd(), from_mjd)
        self.rm_range(to_mjd, self.last_mjd())
    
    def split_mjd_group(self, mjd_group, min_gap_s=160):
        mtss = self.get_mtss_from_mjd_group(mjd_group)
        for mts in mtss:
            self.mts_dict[mts].__str__()
            self.mts_dict[mts].split(min_gap_s=min_gap_s)
    
    def split(self, min_gap_s):
        for mjd_group in self.mjd_groups:
            self.split_mjd_group(mjd_group=mjd_group, min_gap_s=min_gap_s)

    def resample(self, fun='mean', period_s=60, start_mjd=None, points_ratio=0.1):
        """
        Resamples all time MTseries to a given period in seconds.

        Args:
            fun: str
                function to calculate the value of the resampled point
                'mean' - mean value
                'slope' - slope
                'slope_s' -
            period_s: float | int
                period in seconds
            start_mjd: float
                start time in MJD for resampling
            points_ratio: float
        """

        for a in self.mts_dict:
            self.mts_dict[a], mjd_ranges_to_rm = self.mts_dict[a].resample(
                fun=fun,
                period_s=period_s,
                start_mjd=start_mjd,
                points_ratio=points_ratio,
                get_empty_mjd_ranges=True,
            )
            print(a)
            print(mjd_ranges_to_rm)
            for mjd_range in mjd_ranges_to_rm:
                self.rm_range(mjd_range[0], mjd_range[1])

    def resample_to_mts(
        self,
        ref_mts_name,
        grid_period_s,
        none_fields=True,
        rm_none_fields=True,
        none_val=None,
        new_gts=True,
        points_ratio=0.7,
    ):
        ref_mts = self.mts_dict[ref_mts_name]
        mjd_group = self.mjd_groups[ref_mts_name]
        # mtss_names = [mts_name for mts_name in self.mts_dict]
        if new_gts:
            g = GTserie(name='resampled')
        for mts_name in self.mts_dict:
            mts = self.mts_dict[mts_name]
            if self.mjd_groups[mts_name] == mjd_group:
                tmp = mts
            else:
                tmp = mts.resample_to_mts_grid(
                    mts=ref_mts,
                    grid_period_s=grid_period_s,
                    fun='mean',
                    points_ratio=points_ratio,
                    none_fields=none_fields,
                    none_val=none_val,
                )
            if not new_gts:
                self.mts_dict[mts_name]=tmp
                self.mts_dict[mts_name].__str__()
            else:
                g.append_mtserie(
                    mts_name=mts_name,
                    mts=tmp,
                    mjd_group=mjd_group,
                )
        if rm_none_fields:
            # for mts_name in g.mts_dict:
            #     g.rm_value(mts_name, none_val)
            print(
                two_mts_equal_mjd(
                    self.mts_dict['umk_frac_freq'],
                    self.mts_dict['nmij_frac_freq']
                )
            )
            # g.rm_value('nmij_frac_freq', none_val)
        if new_gts:
            return g     

    def resample2(self, period_s):
        first_mjd = self.first_mjd()
        first_mjd_int = np.floor(first_mjd)
        last_mjd = self.last_mjd()
        last_mjd_int = np.floor(last_mjd)
        period_mjd = period_s*s2mjd
        # find the last grid point before the first MJD
        start_mjd = np.floor((first_mjd % 1)/period_mjd)*period_mjd + first_mjd_int
        # find the first grid point after the last MJD
        stop_mjd = np.ceil((last_mjd % 1)/period_mjd)*period_mjd + last_mjd_int
        nm = np.arange(start_mjd, stop_mjd+1e-7, period_mjd)

        # prepare np.arrays for all mts
        nv = dict()
        ov = dict()
        om = dict()
        for mtn in self.mts_dict:
            nv[mtn] = np.zeros(len(s), type=float)
            ov[mtn] = self.mts_dict[mtn].mjd_val()
            om[mtn] = self.mts_dict[mtn].mjd_val()

        # main iterations for all mts
        for ni in range(len(nm)-1):
            nmjd = nm[ni]
            for mts_name in self.mts_dict:
                print(ni)

    def add_mts_to_mts(self, mts_name_1, mts_name_2, mts_name_out):
        from timanda.mtserie import MTSerie
        mts1=self.mts_dict[mts_name_1]
        mts2=self.mts_dict[mts_name_2]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            ts2 = mts2.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                val_tab.append(ts1.val_tab[i_ts]+ts2.val_tab[i_ts])
                mjd_tab.append(ts1.mjd_tab[i_ts])
            out_ts = TSerie(mjd=mjd_tab, val=val_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')

    def math_mts_and_mts(self, operation, mts_name_1, mts_name_2, mts_name_out, decimal=False):
        from timanda.mtserie import MTSerie
        mts1=self.mts_dict[mts_name_1]
        mts2=self.mts_dict[mts_name_2]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            ts2 = mts2.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            pps_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                if operation in OPERATIONS:
                    val_tab.append(OPERATIONS[operation](
                        ts1.val_tab[i_ts],
                        ts2.val_tab[i_ts]
                    ))
                mjd_tab.append(ts1.mjd_tab[i_ts])
                pps_tab.append((ts1.pps_tab[i_ts]+ts1.pps_tab[i_ts])/2)
            out_ts = TSerie(mjd=mjd_tab, val=val_tab, pps=pps_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')
    
    def math_mts_and_number(self, operation, mts_name_1, number, mts_name_out):
        from timanda.mtserie import MTSerie
        mts1=self.mts_dict[mts_name_1]
        out_mts = MTSerie(label=mts_name_out)
        for i_dtab in range(0, len(mts1.dtab)):
            ts1 = mts1.dtab[i_dtab]
            val_tab = list()
            mjd_tab = list()
            pps_tab = list()
            for i_ts in range(0,len(ts1.val_tab)):
                if operation in OPERATIONS:
                    val_tab.append(OPERATIONS[operation](
                        ts1.val_tab[i_ts],
                        number
                    ))
                mjd_tab.append(ts1.mjd_tab[i_ts])
                pps_tab.append(ts1.pps_tab[i_ts])
            out_ts = TSerie(mjd=mjd_tab, val=val_tab, pps=pps_tab)
            out_mts.add_TSerie(out_ts)
        self.append_mtserie(mts_name=mts_name_out, mts=out_mts, mjd_group='gnss_mjd')
    
    def create_comparator_file(self, filename, mts_names, formats, headers):
        """
        Creates a file with data of all MTseries in the GTserie.
        """
        with open(filename, 'w') as f:
            f.write(f"# MJD")
            for h in headers:
                f.write(f"\t{h}")
            f.write('\n')
            
            for i, mjd in enumerate(self.mts_dict[mts_names[0]].mjd_tab()):
                f.write(f"{mjd:.6f}")

                for j, mts_name in enumerate(mts_names):
                    if mts_name in self.mts_dict:
                        f.write(f"\t{self.mts_dict[mts_name].val_tab()[i]:{formats[j]}}")
                    else:
                        f.write("\tNone")

                f.write('\n')
        f.close()


def import_data_to_df_rocit_oc(
    info,
    name='umk',
    headers = ['date', 'time', 'frac_freq', 'confidence', 'systematics'],
):
    """
    Importing data to dataframe (pandas) from March 2020 Rocit campaign and creating mjd data

    Args:
        info (dict): dictionary including information about path to lab/clock data
            ex.:
                data_path = path.Path('./Data_storage/clocks_vs_maser')
                info['umk'] ={'data_dir': data_path / 'UMK_Sr1-HMAOS'}
        name: name of the lab/clock
        headers: names of columns imported from file
    Return:
        Pandas dataframe
    """
    
    df = pd.DataFrame([])
    for f in info[name]['data_dir'].iterdir():
        if f.suffix == '.dat':
            p = pd.read_csv(
                f,
                names=headers,
                skiprows=11, 
                skip_blank_lines=True, 
                sep=' |\t',
                engine='python'
            )
            df = pd.concat([df, p], ignore_index=True)
    df['mjd']=Time(pd.to_datetime(df['date']+' '+df['time'])).mjd
    return df


def import_data_to_df_rocit_gnss(path='./Data_storage/sn112-nmij.dat'):
    headers = ['mjd', 'delta_t_ns']
    p = pd.read_csv(
        path, 
        names=headers,
        skip_blank_lines=True,
        sep='\t',
    )
    df = pd.DataFrame(p)
    return df


def two_mts_equal_mjd(mts1, mts2):
    return (
        len(mts1.dtab) == len(mts2.dtab)
        and all(np.array_equal(a, b) for a, b in zip(mts1.mjd_tab(), mts2.mjd_tab()))
    )

OPERATIONS = {
    'add': lambda x, y: x + y,
    'add_d': lambda x, y: float(D(x)+D(y)),
    'multiply': lambda x, y: x * y,
    'multiply_d': lambda x, y: float(D(x)*D(y)),
    'divide': lambda x, y: x / y,
    'divide_d': lambda x, y: float(D(x)/D(y)),
}


def get_test_tserie(fmjd = 50000, tmjd=50001, period_s=1, noise_ampl=1, mean_val=0):
    mjd_tab = np.arange(fmjd, tmjd, 1/(24*60*60))
    val_tab = np.random.uniform(mean_val-noise_ampl, mean_val+noise_ampl, size=mjd_tab.shape)
    return TSerie(mjd=mjd_tab, val=val_tab)
