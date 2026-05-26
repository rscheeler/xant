"""Helper utilities for data encountered in xant."""

import numpy as np
import xarray as xr
from loguru import logger


def remap_antenna_pattern(
    da: xr.DataArray,
    theta_dim: str = "theta",
    phi_dim: str = "phi",
) -> xr.DataArray:
    """
    Remaps antenna pattern data into canonical (theta, phi) space.

    Canonical space:
        theta : [0, 180]
        phi   : [-180, 180) ordered Q3, Q4, Q1, Q2

    Transformation logic:
    - Points at (+theta, phi) stay at (+theta, wrap(phi))
    - Points at (-theta, phi) move to (+theta, wrap(phi + 180))
    - Standard [inclusive, exclusive) intervals handle boundary capture.
    """
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
        1: [  # Target: [0, 90)
            {"t_range": (0, 180), "p_range": (0, 90)},
            {"t_range": (0, 180), "p_range": (-360, -270)},
            {"t_range": (-180, 0), "p_range": (-180, -90)},
            {"t_range": (-180, 0), "p_range": (180, 270)},
        ],
        2: [  # Target: [90, 180)
            {"t_range": (0, 180), "p_range": (90, 180)},
            {"t_range": (0, 180), "p_range": (-270, -180)},
            {"t_range": (-180, 0), "p_range": (-90, 0)},
            {"t_range": (-180, 0), "p_range": (270, 360)},
        ],
        3: [  # Target: [-180, -90)
            {"t_range": (0, 180), "p_range": (-180, -90)},
            {"t_range": (0, 180), "p_range": (180, 270)},
            {"t_range": (-180, 0), "p_range": (0, 90)},
            {"t_range": (-180, 0), "p_range": (-360, -270)},
        ],
        4: [  # Target: [-90, 0)
            {"t_range": (0, 180), "p_range": (-90, 0)},
            {"t_range": (0, 180), "p_range": (270, 360)},
            {"t_range": (-180, 0), "p_range": (90, 180)},
            {"t_range": (-180, 0), "p_range": (-270, -180)},
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
                # Positive hemisphere: [0, 180]
                if include_zero_in_positive:
                    t_mask = (theta_src >= t_lo - eps) & (theta_src <= t_hi + eps)
                else:
                    t_mask = (theta_src > eps) & (theta_src <= t_hi + eps)
            # Negative hemisphere: [-180, 0]
            elif include_zero_in_negative:
                t_mask = (theta_src >= t_lo - eps) & (theta_src <= t_hi + eps)
            else:
                t_mask = (theta_src >= t_lo - eps) & (theta_src < -eps)

            if t_mask.any() and p_mask.any():
                block = da.isel({theta_dim: t_mask, phi_dim: p_mask})
                p_vals = block[phi_dim].values

                if t_lo < 0:
                    # Physical Rotation: (-theta, phi) -> (+theta, phi + 180)
                    block = block.assign_coords({theta_dim: np.abs(block[theta_dim].values)})
                    block = block * -1.0  # Wipes out the 180-degree step phase change
                    new_phi = p_vals + 180.0
                else:
                    new_phi = p_vals

                # Wrap to canonical [-180, 180)
                new_phi = (new_phi + 180.0) % 360.0 - 180.0
                block = block.assign_coords({phi_dim: new_phi})

                # Sort the block coordinates to prevent descending alignment issues on concat
                block = block.sortby([theta_dim, phi_dim])
                sub_blocks.append(block)

        if sub_blocks:
            # Combine fragments that landed in this quadrant
            # We use drop_duplicates directly to resolve boundary duplicates gracefully
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

    result.attrs.update(da.attrs)
    return result
