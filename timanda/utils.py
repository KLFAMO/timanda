import pandas as pd
import numpy as np
from decimal import Decimal as D
from timanda.tserie import TSerie
from timanda.mtserie import MTSerie
from astropy.time import Time
mjd2s = 24 * 60 * 60


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

def get_test_mtserie(
    n_segments: int = 2,
    fmjd: float = 60000.0,
    segment_len_s: float = 5.0,
    period_s: float = 1.0,
    gap_s: float = 0.1,
    noise_ampl: float = 1.0,
    mean_val: float = 0.0,
    mean_step: float = 0.0,          # zmiana średniej między segmentami (np. dryft skokowy)
    trend_per_day: float = 0.0,      # trend liniowy w obrębie segmentu (val/dzień)
    jitter_period_rel: float = 0.0,  # np. 0.1 => +/-10% okresu per segment
    shuffle_segments: bool = False,  # do testu sortowania / scalania
    seed: int | None = 123,
    label_prefix: str = "seg",
):
    """
    Return MTSerie composed of several TSerie segments.

    segment_len_s: length of single segment in seconds
    gap_s: gap between segments in seconds (can be 0)
    jitter_period_rel: per-segment random modification of period_s by +/- this fraction
    trend_per_day: linear trend inside segment (value change per day)
    mean_step: change of mean value between segments (simulating step drift)
    shuffle_segments: if True, randomize order of segments in MTSerie
    seed: random seed for reproducibility
    label_prefix: prefix for segment labels
    """
    rng = np.random.default_rng(seed)

    segments = []
    cur_start_mjd = fmjd

    for i in range(n_segments):
        # okres próbkowania dla segmentu (opcjonalnie jitter)
        if jitter_period_rel > 0:
            factor = 1.0 + rng.uniform(-jitter_period_rel, jitter_period_rel)
            seg_period_s = max(1e-6, period_s * factor)
        else:
            seg_period_s = period_s

        seg_len_mjd = segment_len_s / mjd2s
        step_mjd = seg_period_s / mjd2s

        seg_end_mjd = cur_start_mjd + seg_len_mjd
        mjd_tab = np.arange(cur_start_mjd, seg_end_mjd, step_mjd)

        # trend within segment
        t_days = (mjd_tab - cur_start_mjd)  # days since segment start
        seg_trend = trend_per_day * t_days

        seg_mean = mean_val + i * mean_step
        val_tab = rng.uniform(seg_mean - noise_ampl, seg_mean + noise_ampl, size=mjd_tab.shape) + seg_trend

        ts = TSerie(mjd=mjd_tab, val=val_tab)
        ts.label = f"{label_prefix}{i+1}"
        segments.append(ts)

        # next segment start
        cur_start_mjd = seg_end_mjd + (gap_s / mjd2s)

    if shuffle_segments:
        rng.shuffle(segments)

    mts = MTSerie(tseries=segments)

    return mts
