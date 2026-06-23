"""Helper utilities for data encountered in xant."""

from typing import Literal

import numpy as np
import xarray as xr
from loguru import logger


def remap_antenna_pattern(
    da: xr.DataArray,
    theta_dim: str = "theta",
    phi_dim: str = "phi",
    units: str = "degree",
) -> xr.DataArray:
    """
    Remaps antenna pattern data into canonical (theta, phi) space.

    Canonical space:
        theta : [0, 180] degrees  or  [0, pi] radians
        phi   : [-180, 180) degrees  or  [-pi, pi) radians,
                ordered Q3, Q4, Q1, Q2

    Parameters
    ----------
    units : Literal['degree', 'radian']
        Either 'degree' (default) or 'radian'. Controls all internal constants
        so the quadrant boundaries and wrap logic are always correct.

    Transformation logic:
    - Points at (+theta, phi) stay at (+theta, wrap(phi))
    - Points at (-theta, phi) move to (+theta, wrap(phi + 180 degree / pi radian))
    - Standard [inclusive, exclusive) intervals handle boundary capture.

    Notes:
    -----
    This function assumes data is in a rotation-invariant basis (e.g. x, y, z).
    If your data is in a theta/phi basis, the (-theta, phi) -> (+theta, phi+180)
    remapping requires a sign flip on the field components to account for the
    180-degree phase discontinuity at the pole. Convert to cartesian first using
    ``polarization.project_polarizations`` before calling this function.
    """
    if units not in ("degree", "radian"):
        raise ValueError(f"units must be 'degree' or 'radian', got '{units}'")

    # Convert to degrees for sampling
    if units == "radian":
        da = da.assign_coords(
            {
                theta_dim: np.round(np.degrees(da[theta_dim]), decimals=9),
                phi_dim: np.round(np.degrees(da[phi_dim]), decimals=9),
            },
        )

    # All internal constants scale with units — one source of truth
    HALF = 180.0
    FULL = 2 * HALF
    QUAD = HALF / 2  # 90 degree

    # Initial cleanup & sorting to guarantee monotonic source coordinates
    da = da.sortby([theta_dim, phi_dim])
    theta_src = da[theta_dim].values
    phi_src = da[phi_dim].values

    eps = 1e-9
    has_positive = (theta_src > eps).any()
    has_negative = (theta_src < -eps).any()

    # Determine which hemisphere rules include the boundary at theta=0.0
    include_zero_in_positive = has_positive or not has_negative
    include_zero_in_negative = has_negative or not has_positive

    # specs: Maps target quadrants to source coordinate ranges.
    # We use a consistent [lower, upper) approach for phi selection.
    specs = {
        1: [  # Target: [0, QUAD)
            {"t_range": (0, HALF), "p_range": (0, QUAD)},
            {"t_range": (0, HALF), "p_range": (-FULL, -3 * QUAD)},
            {"t_range": (-HALF, 0), "p_range": (-HALF, -QUAD)},
            {"t_range": (-HALF, 0), "p_range": (HALF, 3 * QUAD)},
        ],
        2: [  # Target: [QUAD, HALF)
            {"t_range": (0, HALF), "p_range": (QUAD, HALF)},
            {"t_range": (0, HALF), "p_range": (-3 * QUAD, -HALF)},
            {"t_range": (-HALF, 0), "p_range": (-QUAD, 0)},
            {"t_range": (-HALF, 0), "p_range": (3 * QUAD, FULL)},
        ],
        3: [  # Target: [-HALF, -QUAD)
            {"t_range": (0, HALF), "p_range": (-HALF, -QUAD)},
            {"t_range": (0, HALF), "p_range": (HALF, 3 * QUAD)},
            {"t_range": (-HALF, 0), "p_range": (0, QUAD)},
            {"t_range": (-HALF, 0), "p_range": (-FULL, -3 * QUAD)},
        ],
        4: [  # Target: [-QUAD, 0)
            {"t_range": (0, HALF), "p_range": (-QUAD, 0)},
            {"t_range": (0, HALF), "p_range": (3 * QUAD, FULL)},
            {"t_range": (-HALF, 0), "p_range": (QUAD, HALF)},
            {"t_range": (-HALF, 0), "p_range": (-3 * QUAD, -HALF)},
        ],
    }

    quadrant_blocks = {}

    for q, options in specs.items():
        sub_blocks = []
        for opt in options:
            t_lo, t_hi = opt["t_range"]
            p_lo, p_hi = opt["p_range"]

            # Filter phi coordinates using standard half-open [p_lo, p_hi) interval
            p_mask = (phi_src >= p_lo - eps) & (phi_src < p_hi - eps)

            # Define theta mask based on whether we should include 0.0 in that hemisphere
            if t_lo >= 0:
                # Positive hemisphere: [0, HALF]
                if include_zero_in_positive:
                    t_mask = (theta_src >= t_lo - eps) & (theta_src <= t_hi + eps)
                else:
                    t_mask = (theta_src > eps) & (theta_src <= t_hi + eps)
            # Negative hemisphere: [-HALF, 0]
            elif include_zero_in_negative:
                t_mask = (theta_src >= t_lo - eps) & (theta_src <= t_hi + eps)
            else:
                t_mask = (theta_src >= t_lo - eps) & (theta_src < -eps)

            if t_mask.any() and p_mask.any():
                block = da.isel({theta_dim: t_mask, phi_dim: p_mask})
                p_vals = block[phi_dim].values

                if t_lo < 0:
                    # Physical Rotation: (-theta, phi) -> (+theta, phi + HALF)
                    block = block.assign_coords({theta_dim: np.abs(block[theta_dim].values)})
                    new_phi = p_vals + HALF
                else:
                    new_phi = p_vals

                # Wrap to canonical [-HALF, HALF)
                new_phi = (new_phi + HALF) % FULL - HALF
                block = block.assign_coords({phi_dim: new_phi})

                # Sort the block coordinates to prevent descending alignment issues on concat
                block = block.sortby([theta_dim, phi_dim])
                sub_blocks.append(block)

        if sub_blocks:
            # Combine fragments that landed in this quadrant
            q_combined = (
                xr.concat(sub_blocks, dim=phi_dim)
                .drop_duplicates(phi_dim, keep="first")
                .drop_duplicates(theta_dim, keep="first")
            )

            # Resolve duplicate coordinates (overlaps at reflection or boundary points)
            if len(np.unique(q_combined[theta_dim])) < len(q_combined[theta_dim]):
                q_combined = q_combined.groupby(theta_dim).mean(dim=theta_dim)

            if len(np.unique(q_combined[phi_dim])) < len(q_combined[phi_dim]):
                q_combined = q_combined.groupby(phi_dim).mean(dim=phi_dim)

            quadrant_blocks[q] = q_combined.sortby([theta_dim, phi_dim])

    if not quadrant_blocks:
        raise ValueError("No data found matching expected antenna pattern ranges.")

    # Final alignment and assembly
    ordered_keys = [k for k in [3, 4, 1, 2] if k in quadrant_blocks]

    # Create a unified theta axis across all strips
    all_thetas = np.unique(
        np.concatenate([quadrant_blocks[k][theta_dim].values for k in ordered_keys]),
    )

    snapped_strips = []
    for q in ordered_keys:
        # Reindexing ensures the strips are aligned on the same theta grid
        blk = quadrant_blocks[q].reindex({theta_dim: all_thetas}, method=None)
        snapped_strips.append(blk)

    # Concat the 90-degree strips into the final 360-degree longitudinal pattern
    result = xr.concat(snapped_strips, dim=phi_dim, join="outer")
    result = result.sortby([theta_dim, phi_dim])

    # Convert back if needed
    if units == "radian":
        result = result.assign_coords(
            {theta_dim: np.radians(result[theta_dim]), phi_dim: np.radians(result[phi_dim])},
        )

    result.attrs.update(da.attrs)
    return result


def pad_data(
    da: xr.DataArray,
    theta_dim: str = "theta",
    phi_dim: str = "phi",
    negative_points: int = 4,
) -> xr.DataArray:
    """Prepare data for map_coordinate interpolation by padding data with 4 data points in negative theta.
    Data must be complete in phi.

    Notes:
    -----
    This function assumes data is in a rotation-invariant basis (e.g. x, y, z).
    If your data is in a theta/phi basis, the (-theta, phi) -> (+theta, phi+180)
    remapping requires a sign flip on the field components to account for the
    180-degree phase discontinuity at the pole. Convert to cartesian first using
    ``polarization.project_polarizations`` before calling this function.
    """
    # Include negative_points points in negative theta so map_coordinates spline filter is accurate
    # and negate the polarization as its in the negative space
    da_neg_theta = da.isel({theta_dim: range(1, negative_points + 1)})
    # Convert coordinates - negate theta
    da_neg_theta = da_neg_theta.assign_coords(**{theta_dim: da_neg_theta.coords[theta_dim] * -1})
    # Roll -> quadrant order goes from 3, 4, 1, 2 to 1, 2, 3, 4 in negative theta space
    da_neg_theta = da_neg_theta.roll({phi_dim: da.coords[phi_dim].size // 2})
    # Concat and sort
    result = xr.concat([da_neg_theta, da], dim=theta_dim, join="outer")
    result = result.sortby([theta_dim, phi_dim])
    # Now interpolation ready, store attribute
    result.attrs["interp_ready"] = True
    return result
