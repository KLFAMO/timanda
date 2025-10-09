import numpy as np
import pytest
from timanda.gtserie import GTserie
from timanda.mtserie import MTSerie
from timanda.tserie import TSerie


def test_gtserie_initialization():
    """Test initialization of GTserie."""
    gts = GTserie(name="Test GTserie")
    assert gts.name == "Test GTserie"
    assert len(gts.mts_dict) == 0
    assert gts.number_of_mts == 0

def test_append_mtserie():
    """Test appending MTSerie to GTserie."""
    gts = GTserie(name="Test GTserie")
    mts = MTSerie(label="Test MTSerie")
    gts.append_mtserie(mts_name="mts1", mts=mts, mjd_group="group1")

    assert len(gts.mts_dict) == 1
    assert gts.number_of_mts == 1
    assert "mts1" in gts.mts_dict
    assert gts.mjd_groups["mts1"] == "group1"

def test_first_and_last_mjd():
    """Test first_mjd and last_mjd methods."""
    gts = GTserie(name="Test GTserie")
    mts1 = MTSerie(label="mts1")
    mts1.add_TSerie(TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0]))
    mts2 = MTSerie(label="mts2")
    mts2.add_TSerie(TSerie(mjd=[4.0, 5.0, 6.0], val=[40.0, 50.0, 60.0]))
    gts.append_mtserie(mts_name="mts1", mts=mts1, mjd_group="group1")
    gts.append_mtserie(mts_name="mts2", mts=mts2, mjd_group="group1")

    assert gts.first_mjd() == 1.0
    assert gts.last_mjd() == 6.0
