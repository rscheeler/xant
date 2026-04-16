from functools import partial
from typing import Tuple, Union

import h3
import numpy as np
import xarray as xr
from hics.geo.dem import llh2geocent
from hics.utils import vector_norm
from pint import Quantity

from .. import ureg

COORDINATE_DIMS = dict(
    phitheta=("phi", "theta"),
    azel=("azimuth", "elevation"),
    uv=("u", "v"),
    arcsin=("au", "av"),
    elaz=("elevation", "azimuth"),
    uvw=("u", "v", "w"),
    llh=("lat", "lon", "h"),
    ecef=("x", "y", "z"),
    cartesian=("x", "y", "z"),
    trueview=("tvx", "tvy"),
    h3=("i", "j", "h"),
)

COORDINATE_SYSTEMS = ["ecef", "llh"]
eps = 12


def phitheta2uvw(
    phi: np.ndarray | xr.DataArray, theta: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts spherical phi and theta positions to cartesian u, v, w.

    Parameters
    ----------
    phi : np.ndarray, xr.DataArray
        Spherical coordinate phi
    theta : np.ndarray, xr.DataArray
        Spherical coordinate theta
    """
    # Shift data slightly away from poles in +/- z
    # Pint quantities are handled in a different manner
    thresh = np.pi / 2
    shift = 10 ** (-eps)
    if isinstance(theta.data, Quantity):
        if theta.data.units == "degree":
            thresh = np.rad2deg(thresh)
            shift *= theta.data.units
    theta_shift = theta.copy()
    theta_shift.data[theta.values < thresh] = theta_shift.data[theta.values < thresh] + shift
    theta_shift.data[theta.values > thresh] = theta_shift.data[theta.values > thresh] - shift

    # Perform transform
    u = np.sin(theta_shift) * np.cos(phi)
    v = np.sin(theta_shift) * np.sin(phi)
    w = np.cos(theta_shift) * xr.ones_like(phi)

    return u, v, w


def uvw2phitheta(
    u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray, w: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts cartesian coordinates u, v, w to polar spherical coordinates phi, theta.

    Parameters
    ----------
    u : np.ndarray, xr.DataArray
        Cartesian coordinate u
    v : np.ndarray, xr.DataArray
        Cartesian coordinate v
    w : np.ndarray, xr.DataArray
        Cartesian coordinate w
    """
    # Transform to phi/theta
    phi = np.arctan2(v, u)
    # Remove quantity if necessary before calculating theta
    w = w.copy()
    if isinstance(w.data, Quantity):
        if w.data.units != ureg.dimensionless:
            w.data = w.data.magnitude
    # Make sure to clip or else w values such as 1.00000000001 will result in nans
    theta = np.arccos(np.clip(w, -1, 1))

    # Need to handle pint units
    units = 1
    if isinstance(phi.data, Quantity):
        units = phi.data.units

    # Round to eliminate shift away from pole by rounding to two decimals less than the shift
    phi.data = np.around(phi.values, eps - 2) * units
    theta.data = np.around(theta.values, eps - 2) * units

    return phi, theta


def uvw2uvw(
    u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray, w: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Convenience function that just returns the inputs.
    """
    return u, v, w


def llh2uvw(lat=None, lon=None, h=None, hagl=True, reference_cs=None):
    """
    Converts LLH to relative uvw.
    """
    # Convert LLH to ECEF
    x, y, z = llh2geocent.transform(lat, lon, h, hagl=hagl)

    # Convert into a DataArray
    if isinstance(x, xr.DataArray):
        position = []
        for da, coord in zip((x, y, z), ["x", "y", "z"]):
            da = da.assign_coords(dict(position=coord))
            position.append(da)
        position = xr.concat(position, dim="position")
    else:
        position = xr.DataArray(
            [x, y, z], dims=("position",), coords=dict(position=["x", "y", "z"])
        )

    # Get position relative to reference_cs
    rel_pos = reference_cs.relative_position(position)
    # Grab units if a Quantity
    units = None
    if isinstance(rel_pos.data, Quantity):
        units = rel_pos.data.units

    # Normalize
    rel_pos /= vector_norm(rel_pos, dim="position")
    # Apply units if needed
    if units is not None:
        rel_pos.data = rel_pos.data * units

    # Set x, y, and z data
    x = rel_pos.sel(position="x")
    x = x.drop_vars("position")
    y = rel_pos.sel(position="y")
    y = y.drop_vars("position")
    z = rel_pos.sel(position="z")
    z = z.drop_vars("position")

    return x, y, z


def cs2uvw(self_cs, other_cs):
    """
    Get relative position in normalized u, v, w of other coordinate system.
    """
    # Get position relative to reference_cs
    rel_pos = self_cs.relative_position(other_cs.position)
    # Grab units if a Quantity
    units = None
    if isinstance(rel_pos.data, Quantity):
        units = rel_pos.data.units

    # Normalize
    rel_pos /= vector_norm(rel_pos, dim="position")
    # Apply units if needed
    if units is not None:
        rel_pos.data = rel_pos.data * units

    # Set x, y, and z data
    x = rel_pos.sel(position="x")
    x = x.drop_vars("position")
    y = rel_pos.sel(position="y")
    y = y.drop_vars("position")
    z = rel_pos.sel(position="z")
    z = z.drop_vars("position")

    return x, y, z


def ecef2uvw(x=None, y=None, z=None, reference_cs=None):
    """
    Converts ECEF to relative uvw.
    """
    # Convert into a DataArray
    if isinstance(x, xr.DataArray):
        position = []
        for da, coord in zip((x, y, z), ["x", "y", "z"]):
            da = da.assign_coords(dict(position=coord))
            position.append(da)
        position = xr.concat(position, dim="position")
    else:
        position = xr.DataArray(
            [x, y, z], dims=("position",), coords=dict(position=["x", "y", "z"])
        )

    # Get position relative to reference_cs
    rel_pos = reference_cs.relative_position(position)
    # Grab units if a Quantity
    units = None
    if isinstance(rel_pos.data, Quantity):
        units = rel_pos.data.units

    # Normalize
    rel_pos /= vector_norm(rel_pos, dim="position")
    # Apply units if needed
    if units is not None:
        rel_pos.data = rel_pos.data * units

    # Set x, y, and z data
    x = rel_pos.sel(position="x")
    x = x.drop_vars("position")
    y = rel_pos.sel(position="y")
    y = y.drop_vars("position")
    z = rel_pos.sel(position="z")
    z = z.drop_vars("position")

    return x, y, z


def cartesian2uvw(x=None, y=None, z=None):
    # Convert into a DataArray
    if isinstance(x, xr.DataArray):
        position = []
        for da, coord in zip((x, y, z), ["x", "y", "z"]):
            da = da.assign_coords(dict(position=coord))
            position.append(da)
        position = xr.concat(position, dim="position")
    else:
        position = xr.DataArray(
            [x, y, z], dims=("position",), coords=dict(position=["x", "y", "z"])
        )

    # Grab units if a Quantity
    units = None
    if isinstance(position.data, Quantity):
        units = position.data.units

    # Normalize
    position /= vector_norm(position, dim="position")
    # Apply units if needed
    if units is not None:
        position.data = position.data * units

    # Set x, y, and z data
    x = position.sel(position="x")
    x = x.drop_vars("position")
    y = position.sel(position="y")
    y = y.drop_vars("position")
    z = position.sel(position="z")
    z = z.drop_vars("position")

    return x, y, z


def azel2uvw(
    azimuth: np.ndarray | xr.DataArray, elevation: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts spherical azimuth and elevation positions in an azimuth over elevation coordinate frame to
    cartesian u, v, w.

    Parameters
    ----------
    azimuth : np.ndarray, xr.DataArray
        Spherical coordinate azimuth
    elevation : np.ndarray, xr.DataArray
        Spherical coordinate elevation
    """
    # Perform transform
    u = np.cos(elevation) * np.sin(azimuth)
    v = np.sin(elevation) * xr.ones_like(azimuth)
    w = np.cos(elevation) * np.cos(azimuth)

    return u, v, w


def uv2uvw(
    u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    uv-projection where u and v define theta/phi. Unit circle for upper hemisphere.
    """
    phi = np.arctan2(v, u)
    theta = np.arcsin(np.sqrt(u**2 + v**2))
    u, v, w = phitheta2uvw(phi, theta)
    return u, v, w


def arcsin2uvw(
    au: np.ndarray | xr.DataArray, av: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Arcsin projection. Unit square for upper hemisphere.
    """
    u = np.sin(au)
    v = np.sin(av)
    w = np.sqrt(u**2 + v**2)

    return u, v, w


def trueview2uvw(
    tvx: np.ndarray | xr.DataArray, tvy: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    True-view projection. Unit square for upper hemisphere.
    """
    xysr = np.sqrt(tvx**2 + tvy**2)

    u = np.sin(xysr) * np.cos(np.arctan2(tvy, tvx))
    v = np.sin(xysr) * np.sin(np.arctan2(tvy, tvx))
    w = np.cos(xysr)

    return u, v, w


def uvw2azel(
    u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray, w: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts cartesian u, v, w to spherical azimuth and elevation positions in an azimuth over elevation coordinate
    frame.

    Parameters
    ----------
    u : np.ndarray, xr.DataArray
        Cartesian coordinate u
    v : np.ndarray, xr.DataArray
        Cartesian coordinate v
    w : np.ndarray, xr.DataArray
        Cartesian coordinate w

    Returns:
    -------
    azimuth : np.ndarray, xr.DataArray
        Spherical coordinate azimuth
    elevation : np.ndarray, xr.DataArray
        Spherical coordinate elevation
    """
    elevation = np.arcsin(v)
    azimuth = np.arctan2(u, w)

    return azimuth, elevation


def elaz2uvw(
    elevation: np.ndarray | xr.DataArray, azimuth: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts spherical elevation and azimuth positions in an elevation over azimuth coordinate frame to
    cartesian u, v, w.

    Parameters
    ----------
    elevation : np.ndarray, xr.DataArray
        Spherical coordinate elevation
    azimuth : np.ndarray, xr.DataArray
        Spherical coordinate azimuth
    """
    # Perform transform
    u = np.sin(azimuth) * xr.ones_like(elevation)
    v = np.cos(azimuth) * np.sin(elevation)
    w = np.cos(azimuth) * np.cos(elevation)

    return u, v, w


def uvw2elaz(
    u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray, w: np.ndarray | xr.DataArray
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Converts cartesian u, v, w to spherical elevation and azimuth positions in an
    elevation over azimuth coordinate frame.

    Parameters
    ----------
    u : np.ndarray, xr.DataArray
        Cartesian coordinate u
    v : np.ndarray, xr.DataArray
        Cartesian coordinate v
    w : np.ndarray, xr.DataArray
        Cartesian coordinate w

    Returns:
    -------
    elevation : np.ndarray, xr.DataArray
        Spherical coordinate elevation
    azimuth : np.ndarray, xr.DataArray
        Spherical coordinate azimuth
    """
    azimuth = np.arcsin(u)
    elevation = np.arctan2(v, w)

    return elevation, azimuth


def h3grid2index(
    i: xr.DataArray, j: xr.DataArray, origin_latlong: tuple, resolution: int
) -> xr.DataArray:
    """
    Convert h3 grid specified by array in i and j and origin lat,lon to h3 cells.
    """
    # Find origin cell
    origin_cell = h3.latlng_to_cell(*origin_latlong, resolution)

    # Center i,j
    ci, cj = h3.cell_to_local_ij(origin_cell, origin_cell)

    # Generate grid of i,j
    i2, j2 = xr.broadcast(i + ci, j + cj)

    # Get cell indicies
    h3cells = xr.apply_ufunc(partial(h3.local_ij_to_cell, origin_cell), i2, j2, vectorize=True)
    # Add h3 cell index as coordinate
    h3cells = h3cells.assign_coords(h3=h3cells)

    return h3cells


def h3grid2ll(
    i: xr.DataArray, j: xr.DataArray, origin_latlong: tuple, resolution: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Convert h3 grid to latitude and longitude.
    """
    # Get cell indicies
    h3cells = h3grid2index(i, j, origin_latlong, resolution)

    # Convert to lat, lon
    lat, lon = xr.apply_ufunc(h3.cell_to_latlng, h3cells, vectorize=True, output_core_dims=[[], []])

    # Add back in units
    lat.data = lat.data * ureg.degree
    lon.data = lon.data * ureg.degree

    return lat, lon


def h32uvw(
    i: np.ndarray | xr.DataArray,
    j: np.ndarray | xr.DataArray,
    h: np.ndarray | xr.DataArray,
    origin_latlong: tuple[float, float] = (0, 0),
    resolution: int = 5,
    hagl: bool = False,
    reference_cs=None,
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    Convert h3 grid into global local uvw by first getting lat, lon, height.
    """
    # Get lat, lon
    lat, lon = h3grid2ll(i, j, origin_latlong, resolution)

    # Convert to uvw
    u, v, w = llh2uvw(lat, lon, h, hagl, reference_cs)

    return u, v, w
