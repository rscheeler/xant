import numpy as np
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


@njit(parallel=True)
def fast_nearest_indices(grid_coords, request_points):
    """Parallelized nearest-neighbor search for non-uniform grids."""
    flat_pts = request_points.ravel()
    flat_idx = np.empty(flat_pts.shape, dtype=np.int64)

    for i in prange(len(flat_pts)):
        # Efficiently find the closest index in the coordinate array
        flat_idx[i] = np.abs(grid_coords - flat_pts[i]).argmin()

    return flat_idx.reshape(request_points.shape)
