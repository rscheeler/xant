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
    Remaps an xarray DataArray with antenna pattern data into canonical space:
    theta : [0, 180]    (inclusive both ends if present in source)
    phi   : [-180, 180) (split into 4 x 90-deg strips)

    Source data may span any sub-range of theta in [-180, 180] and phi in [-360, 360).

    Lookup table -- which output quadrant each (theta_row, phi_col) block maps to:

        Theta / Phi   [-360,-270) [-270,-180) [-180,-90) [-90,0) [0,90) [90,180) [180,270) [270,360)
        [-180,  0)        3           4           1         2      3       4        1         2
        [  0, 180]        1           2           3         4      1       2        3         4

    Output quadrant -> output phi strip (theta covers the full available range):
        Q1 -> phi [  0,  90)
        Q2 -> phi [ 90, 180)
        Q3 -> phi [-180, -90)
        Q4 -> phi [ -90,   0)

    Output phi order left-to-right: Q3, Q4, Q1, Q2.

    Transformation logic:
    - Points at (+theta, phi) stay at (+theta, wrap(phi))
    - Points at (-theta, phi) move to (+theta, wrap(phi + 180))
    - Standard [inclusive, exclusive) intervals handle boundary capture.
    """
    # Initial cleanup
    da = da.sortby([theta_dim, phi_dim])
    theta_src = da[theta_dim].values
    phi_src = da[phi_dim].values

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

            # Filter coordinates using standard [lo, hi) for phi
            # Theta is treated as [lo, hi] to capture the 180.0 pole
            eps = 1e-9
            t_mask = (theta_src >= t_lo - eps) & (theta_src <= t_hi + eps)
            p_mask = (phi_src >= p_lo - eps) & (phi_src < p_hi - eps)

            if t_mask.any() and p_mask.any():
                block = da.isel({theta_dim: t_mask, phi_dim: p_mask})
                p_vals = block[phi_dim].values

                if t_lo < 0:
                    # Physical Rotation: (-theta, phi) -> (+theta, phi + 180)
                    block = block.assign_coords({theta_dim: np.abs(block[theta_dim].values)})
                    block = block * -1.0  # Wipes out the 180-degree step change
                    new_phi = p_vals + 180.0
                else:
                    new_phi = p_vals

                # Wrap to canonical [-180, 180)
                new_phi = (new_phi + 180.0) % 360.0 - 180.0
                block = block.assign_coords({phi_dim: new_phi})
                sub_blocks.append(block)

        if sub_blocks:
            # Combine fragments that landed in this quadrant along phi
            q_combined = (
                xr.concat(sub_blocks, dim=phi_dim).drop_duplicates("phi").drop_duplicates("theta")
            )

            # Resolve duplicate coordinates (overlaps at reflection or boundary points)
            if len(np.unique(q_combined[theta_dim])) < len(q_combined[theta_dim]):
                q_combined = q_combined.groupby(theta_dim).mean(dim=theta_dim)

            if len(np.unique(q_combined[phi_dim])) < len(q_combined[phi_dim]):
                q_combined = q_combined.groupby(phi_dim).mean(dim=phi_dim)

            quadrant_blocks[q] = q_combined.sortby([theta_dim, phi_dim])

    if not quadrant_blocks:
        raise ValueError("No data found matching expected antenna pattern ranges.")

    #  Final alignment and assembly
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
