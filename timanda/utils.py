import pandas as pd
import numpy as np
from decimal import Decimal as D
from timanda.tserie import TSerie
from astropy.time import Time


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