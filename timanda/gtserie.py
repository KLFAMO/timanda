import numpy as np
import matplotlib.pyplot as plt
s2mjd = 1/(24*60*60)  # seconds to MJD
from timanda.utils import import_data_to_df_rocit_gnss, two_mts_equal_mjd, import_data_to_df_rocit_oc
from timanda.utils import OPERATIONS
# from timanda.mtserie import MTSerie
from timanda.tserie import TSerie
from decimal import Decimal as D 

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

    def plot(self, fig=None, axs=None, figsize=(7, 7), mts_names=None, show=1, zorder=1,
             time_unit='mjd'):
        if not mts_names:
            mts_names = self.mts_dict
        fig, axs = plt.subplots(len(mts_names),1,  constrained_layout=True, sharex=True, figsize=figsize)
        for i, mts_name in enumerate(mts_names):
            self.mts_dict[mts_name].plot(ax=axs[i], show=0, zorder=zorder, time_unit=time_unit)
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
        Resample all MTSerie objects in this GTserie onto a regular time grid
        using each MTSerie.resample(...).

        This method delegates the actual resampling to each MTSerie and then
        performs a consistency cleanup across all series by removing time ranges
        where the resampling produced "empty" output (gaps with insufficient data).

        Parameters
        ----------
        fun : str, default 'mean'
            Aggregation function used to compute resampled values inside each grid bin.
            The exact set of supported functions depends on MTSerie.resample(...).
            Typical examples:
            - 'mean'  : average value in the bin
            - 'slope' : slope estimated in the bin (if supported by MTSerie)
        period_s : float | int, default 60
            Resampling period in seconds. Defines the target grid spacing.
        start_mjd : float | None, default None
            Optional starting epoch (MJD) for the resampling grid. If None, the
            MTSerie implementation selects an appropriate start (typically derived
            from the first available sample).
        points_ratio : float, default 0.1
            Minimum fraction (or threshold) of points required in a bin to consider
            it valid (exact interpretation depends on MTSerie.resample).

        Behavior / Side Effects
        -----------------------
        - For each MTSerie in self.mts_dict:
            1) Replace it with its resampled version returned by MTSerie.resample(...).
            2) Retrieve empty MJD ranges (gaps) detected during resampling.
        - For each empty MJD range returned by any series:
            remove that time range from ALL series in the GTserie via self.rm_range(...).
        This enforces alignment across series by ensuring that no series retains
        data in periods where another series had insufficient coverage after resampling.

        Returns
        -------
        None
            The operation is performed in-place: self.mts_dict is modified.

        Notes
        -----
        This function is a "high-level" resampling pipeline:
        - It relies on MTSerie.resample(...) to compute the resampled data.
        - It additionally enforces cross-series consistency by removing empty ranges
        from every series, not only the one that detected the gap.
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


    def align_all_to_grid_zoh_and_drop_missing(
        self,
        period_s: float | int = 1,
        sh_s: float | int = 0.05,
        snap_s: float | int = 0.0,
        tol_s: float | int = 7,
        start_mjd: float | None = None,
        stop_mjd: float | None = None,
        new_gts: bool = True,
        out_name: str = "aligned_common",
        hold_last: bool = False,
    ):
        """
        Align all MTSerie objects to the same regular grid using align_to_grid_zoh()
        and remove grid points that are missing in at least one series.

        The procedure is:
        1) Resample each series to the same grid (period_s, start_mjd, stop_mjd).
        2) Combine rm_mask from all series using logical OR.
           A grid point is removed if it is missing in any series.
        3) Remove these points from all series.

        Parameters
        ----------
        period_s, sh_s, snap_s, tol_s, start_mjd, stop_mjd
            Passed identically to MTSerie.align_to_grid_zoh() for every series.
        new_gts : bool
            If True, return a new GTserie containing aligned and trimmed series.
            If False, modify this GTserie in place and return self.
        out_name : str
            Name of the returned GTserie when new_gts=True.

        Returns
        -------
        g : GTserie
            The aligned GTserie (new object if new_gts=True, otherwise self).
        common_rm_mask : np.ndarray of bool
            Boolean mask of removed grid points (True = removed).
        """

        # Create output container if requested
        if new_gts:
            g = GTserie(name=out_name)
        else:
            g = self

        common_rm_mask = None

        # First pass: align each series and build the common removal mask
        for mts_name in self.mts_dict:
            mts = self.mts_dict[mts_name]

            tmp, rm_mask = mts.align_to_grid_zoh(
                period_s=period_s,
                sh_s=sh_s,
                snap_s=snap_s,
                tol_s=tol_s,
                start_mjd=start_mjd,
                stop_mjd=stop_mjd,
                hold_last=hold_last,
            )

            # Combine masks (logical OR): remove if missing in ANY series
            common_rm_mask = (common_rm_mask | rm_mask) if common_rm_mask is not None else rm_mask.copy()

            # Store aligned series
            if new_gts:
                # Preserve mjd_group if available; otherwise use empty string
                mjd_group = self.mjd_groups.get(mts_name, "")
                g.append_mtserie(mts_name=mts_name, mts=tmp, mjd_group=mjd_group)
            else:
                g.mts_dict[mts_name] = tmp

        # Second pass: remove common missing points from all series
        for mts_name in g.mts_dict:
            g.mts_dict[mts_name].rm_indexes([common_rm_mask])

        return g, common_rm_mask


    # def resample2(self, period_s):
    #     """
    #     Experimental / work-in-progress resampling implementation based on explicit
    #     grid construction in MJD and manual bin iteration.

    #     This method attempts to:
    #     1) Compute a global regular MJD grid with spacing period_s seconds.
    #     2) Prepare per-series numpy arrays for output values.
    #     3) Iterate over consecutive grid intervals and compute resampled values.

    #     Parameters
    #     ----------
    #     period_s : float | int
    #         Resampling period in seconds used to define the grid spacing.

    #     Intended Algorithm (as implemented so far)
    #     ------------------------------------------
    #     - Convert seconds to days (MJD fractional days) using s2mjd = 1/(24*60*60).
    #     - Determine:
    #         first_mjd = earliest timestamp across all series
    #         last_mjd  = latest timestamp across all series
    #     and split them into integer day part + fractional part.
    #     - Compute:
    #         period_mjd = period_s * s2mjd
    #     - Create a grid [start_mjd, ..., stop_mjd] such that:
    #         start_mjd aligns to the last grid point <= first_mjd
    #         stop_mjd  aligns to the first grid point >= last_mjd
    #     then build nm = np.arange(start_mjd, stop_mjd, period_mjd).

    #     Current Status / Limitations
    #     ----------------------------
    #     - The function is incomplete: the main computation loop is not implemented.
    #     - There are apparent bugs/typos in the current code (e.g. use of an undefined
    #     variable 's', incorrect numpy zeros argument 'type' instead of 'dtype', and
    #     duplicate assignments for ov/om).
    #     - No value aggregation rule is defined yet (mean/slope/etc.), unlike resample().

    #     Returns
    #     -------
    #     None (currently)
    #         As written, the function does not return a resampled dataset yet.

    #     Key Differences vs resample()
    #     -----------------------------
    #     resample():
    #     - Production-oriented: delegates resampling to MTSerie.resample(...)
    #     - Removes empty MJD ranges globally across all series to keep alignment
    #     - Supports configurable aggregation via 'fun', and grid control via start_mjd
    #     - Operates in-place and fully executes end-to-end

    #     resample2():
    #     - Low-level, manual approach: constructs the grid and intends to iterate bins
    #     - Not finished and currently not equivalent in functionality
    #     - No cross-series cleanup strategy is applied yet
    #     - Intended as an alternative implementation (potentially faster or more
    #         controllable), but currently not usable as a drop-in replacement
    #     """


    #     first_mjd = self.first_mjd()
    #     first_mjd_int = np.floor(first_mjd)
    #     last_mjd = self.last_mjd()
    #     last_mjd_int = np.floor(last_mjd)
    #     period_mjd = period_s*s2mjd
    #     # find the last grid point before the first MJD
    #     start_mjd = np.floor((first_mjd % 1)/period_mjd)*period_mjd + first_mjd_int
    #     # find the first grid point after the last MJD
    #     stop_mjd = np.ceil((last_mjd % 1)/period_mjd)*period_mjd + last_mjd_int
    #     nm = np.arange(start_mjd, stop_mjd+1e-7, period_mjd)

    #     # prepare np.arrays for all mts
    #     nv = dict()
    #     ov = dict()
    #     om = dict()
    #     for mtn in self.mts_dict:
    #         nv[mtn] = np.zeros(len(s), type=float)
    #         ov[mtn] = self.mts_dict[mtn].mjd_val()
    #         om[mtn] = self.mts_dict[mtn].mjd_val()

    #     # main iterations for all mts
    #     for ni in range(len(nm)-1):
    #         nmjd = nm[ni]
    #         for mts_name in self.mts_dict:
    #             print(ni)

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