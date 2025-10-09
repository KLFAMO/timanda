import numpy as np
import matplotlib.pyplot as plt

class MGserie:
    def __init__(self, ser1, ser2, grid_s, fmjd=None, tmjd=None, grid_mode=1):
        self.dtab = []
        self.grid_s = grid_s
        self.grid_mjd = grid_s/(24*60*60)
        self.mjd_min = min(ser1.mjd_tab[0], ser2.mjd_tab[0])
        self.mjd_max = max(ser1.mjd_tab[-1], ser2.mjd_tab[-1])
        if fmjd is not None:
            self.mjd_min = max(self.mjd_min, fmjd)
        if tmjd is not None:
            self.mjd_max = min(self.mjd_max, tmjd)

        grid_tab = np.arange(self.mjd_min, self.mjd_max, self.grid_mjd)
        self.dtab = np.zeros((3, len(grid_tab)))
        for ind, mjd in enumerate(grid_tab):
            self.dtab[1][ind] = ser1.mjd2val(mjd)
            self.dtab[2][ind] = ser2.mjd2val(mjd)
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