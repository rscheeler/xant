"""
Data structures and utility functions
"""
from pathlib import Path

from pint import Quantity
import numpy as np
import xarray as xr


_DIR = Path(__file__).parent

# Electrical Ground Constants
# NTIA Technical Report ERL 79-ITS 67
# Prediction of Tropospheric Radio Transmission Loss Over Irregular Terrain: A Computer Method - 1968
# https://www.its.bldrdoc.gov/publications/details.aspx?pub=2784
# NTIA REPORT 82-100
# A Guide to the Use of the  ITS Irregular Terrain Model in the Area Prediction Mode
# https://www.ntia.doc.gov/files/ntia/publications/ntia_82-100_20121129145031_555510.pdf
_GND_EPS = dict(poor=4, average=15, good=25, fresh_water=81, sea_water=81)
_GND_SGM = dict(poor=0.001, average=0.005, good=0.02, fresh_water=0.01, sea_water=5.0)
ITM_POLARIZATION = dict(horizontal=0, vertical=1)

def take_magnitudes(*args):
    """
    Format of inputs are a tuple where the first entry is the data to be converted the second value is the unit to
    convert to.

    args : tuple
        Format is (data (Quantity or DataArray, str or unit))
    """

    converted = []
    for a in args:
        if isinstance(a[0], Quantity):
            converted.append(a[0].to(a[1]).magnitude)
        elif isinstance(a[0], xr.DataArray):

            if isinstance(a[0].data, Quantity):
                b = a[0].copy()
                b.data = b.data.to(a[1]).magnitude
                converted.append(b)
            else:
                converted.append(a[0])
        else:
            converted.append(a[0])

    return converted


def load_climate_data():
    """
    Load climate zone data

    Climate selection
    1=equatorial,
    2=continental subtropical
    3=maritime subtropical,
    4=desert
    5=continental temperate,
    6=maritime temperate overland,
    7=maritime temperate, oversea (5 is the default)
    """
    # Create netCDF file if it doesn't exist
    if not (_DIR / "resource/itm_climate_zones.nc").exists():
        # Load txt file
        clm_zns = np.loadtxt(_DIR / "resource/itm_climate_zones.txt")
        # Create DataArray note coordinate values - see data file for details
        clm_zns = xr.DataArray(
            clm_zns,
            dims=("lat", "lon"),
            coords=dict(lat=89.75 - np.arange(360) * 0.5, lon=-179.75 + np.arange(720) * 0.5),
        )
        # Fix values zet to 0 for sea and convert to 7
        mask = clm_zns == 0
        clm_zns = xr.where(mask, 7, clm_zns)
        # Export to netCDF
        clm_zns.to_netcdf(_DIR / "resource/itm_climate_zones.nc")

    clm_zns = xr.load_dataarray(_DIR / "resource/itm_climate_zones.nc")
    return clm_zns


def load_refractivity():
    """
    Load surface refractivity data.
    """
    # Create netCDF file if it doesn't exist
    if not (_DIR / "resource/itm_refractivity.nc").exists():
        # Load txt file
        refr = np.loadtxt(_DIR / "resource/itm_refractivity.txt")
        # Create DataArray note coordinate values - see data file for details
        lons = 0 + np.arange(241) * 1.5
        refr = xr.DataArray(
            refr,
            dims=("lat", "lon"),
            coords=dict(lat=90 - np.arange(121) * 1.5, lon=lons),
        )
        # Fix longitude values
        neg_lon = refr.sel(lon=lons[120:-1])
        neg_lon = neg_lon.assign_coords(lon=neg_lon.lon - 360)
        pos_lon = refr.sel(lon=lons[:121])
        refr = xr.concat([neg_lon, pos_lon], "lon")

        # Export to netCDF
        refr.to_netcdf(_DIR / "resource/itm_refractivity.nc")

    refr = xr.load_dataarray(_DIR / "resource/itm_refractivity.nc")
    return refr


CLIMATE_ZONES = load_climate_data()
REFRACTIVITY = load_refractivity()
