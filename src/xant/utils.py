from copy import deepcopy

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity
from scipy.spatial.transform import Rotation

from numba import njit, prange



def round2base(x: float, base: float = 5) -> float:
    """
    Round number up to nearest base.

    Parameters
    ----------
    x : float
        Number to round
    base : float
        Base to round to
    """
    return base * np.ceil(x / base)


def apply_rotation(
    rotation: Rotation | xr.DataArray,
    da: xr.DataArray,
    rotation_dim: str = "position",
    inverse: bool | None = False,
):
    """
    Applies the rotation from the Rotation object to a DataArray of coordinates. Function can be used for
    rotation of polarization.

    Parameters
    ----------
    rotation : Rotation, xr.DataArray
        Rotation object to be applied
    da : xr.DataArray
        DataArray to apply the rotation to. Shall have dimension "position" with coordinate "x", "y", "z".
    rotation_dim: str="position"
        Rotation dim to apply to.
    inverse : bool
        Whether to apply the inverse of the rotation
    """
    # Get original dimensions
    olddims = deepcopy(list(da.dims))

    # Transpose so rotation dimension is last
    newdims = deepcopy(olddims)
    newdims.pop(newdims.index(rotation_dim))
    newdims = newdims + [rotation_dim]
    da = da.transpose(*newdims)

    # Grab Units
    units = None
    if isinstance(da.data, Quantity):
        units = da.data.units

    # Handle special case of non-dimnesional DataArray set rotation to be the item
    if isinstance(rotation, xr.DataArray):
        if rotation.dims == ():
            rotation = rotation.item()

    # Grab Shape and ravel
    shape = da.shape
    data = da.data.reshape(-1, 3)

    # Make sure data is type float
    if data.dtype == "object":
        data = data.astype(np.float64)

    # Rotate data
    newdata = rotation.apply(data, inverse=inverse)

    # Reshape and assign rotated data units if needed
    newdata = newdata.reshape(shape)

    # Add back in units
    if units is not None:
        newdata *= units

    # Assign back to da
    da.data = newdata

    # Make sure data is a float - sometimes it is an object array
    if isinstance(da.data, Quantity):
        if da.data.magnitude.dtype == "object":
            da.data = da.data.magnitude.astype(np.float64) * da.data.units
    elif da.data.dtype == "object":
        da.data = da.data.astype(np.float64)

    return da

@njit(parallel=True)
def fast_nearest_indices(grid_coords, request_points):
    """Parallelized nearest-neighbor search for non-uniform grids."""
    flat_pts = request_points.ravel()
    flat_idx = np.empty(flat_pts.shape, dtype=np.int64)
    
    for i in prange(len(flat_pts)):
        # Efficiently find the closest index in the coordinate array
        flat_idx[i] = np.abs(grid_coords - flat_pts[i]).argmin()
        
    return flat_idx.reshape(request_points.shape)