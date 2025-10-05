from timanda.mtserie import MTSerie
from timanda.tserie import TSerie
import numpy as np

def test_create_empty_mtserie():
    mts = MTSerie()
    assert mts.label == ''
    assert mts.color == 'green'
    assert len(mts.dtab) == 0

def test_mtserie_with_tseries():
    ts1 = TSerie(label="TSerie 1", mjd=[1.0, 2.0], val=[10.0, 20.0])
    ts2 = TSerie(label="TSerie 2", mjd=[3.0, 4.0], val=[30.0, 40.0])
    mts = MTSerie(label="Test MTSerie", tseries=[ts1, ts2])
    
    assert mts.label == "Test MTSerie"
    assert len(mts.dtab) == 2
    assert mts.dtab[0].label == "TSerie 1"
    assert mts.dtab[1].label == "TSerie 2" 

def test_create_mtserie_with_mjd_val():
    mts = MTSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
    assert len(mts.dtab) == 1
    assert mts.dtab[0].mjd_tab.tolist() == [1.0, 2.0, 3.0]
    assert mts.dtab[0].val_tab.tolist() == [10.0, 20.0, 30.0]

def test_mtserie_with_txt_file(mocker):
    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="1.000001 10.0\n1.000002 20.0\n1.000003 30.0\n1.000004 40.0\n")
    )
    mts = MTSerie(txtFileName="test_file.txt")
    
    assert len(mts.dtab) == 1
    assert np.array_equal(mts.dtab[0].mjd_tab, np.array([1.000001, 1.000002, 1.000003, 1.000004]))
    assert np.array_equal(mts.dtab[0].val_tab, np.array([10.0, 20.0, 30.0, 40.0]))