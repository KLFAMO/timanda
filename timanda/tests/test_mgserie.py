import numpy as np
import pytest
from timanda.mgserie import MGserie
from timanda.tserie import TSerie

# def test_mgserie_initialization():
#     """Test initialization of MGserie."""
#     ser1 = TSerie(mjd=[1.0, 2.0, 3.0], val=[10.0, 20.0, 30.0])
#     ser2 = TSerie(mjd=[1.5, 2.5, 3.5], val=[15.0, 25.0, 35.0])
#     mg = MGserie(ser1, ser2, grid_s=86400)  # Grid size = 1 day

#     assert mg.grid_s == 86400
#     assert len(mg.dtab[0]) > 0  # Grid should not be empty
#     assert len(mg.dtab[1]) == len(mg.dtab[0])
#     assert len(mg.dtab[2]) == len(mg.dtab[0])