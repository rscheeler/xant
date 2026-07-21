# tests/conftest.py
import matplotlib
import pytest
from hics.geo.dem import DEM

matplotlib.use("Agg")


@pytest.fixture(autouse=True, scope="session")
def _no_network_dem():
    DEM.local_only = True
