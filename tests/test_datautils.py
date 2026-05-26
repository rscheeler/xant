"""Pytest suite for utils/datautils.py"""

import numpy as np
import pytest
import xarray as xr

from xant.utils.datautils import remap_antenna_pattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOL = 0.01  # degree tolerance for coordinate checks


def _da(theta, phi, data=None, extra_dims=None):
    """Build a 2-D (or higher) DataArray.  data defaults to ones."""
    if data is None:
        data = np.ones((len(theta), len(phi)))
    da = xr.DataArray(data, coords={"theta": theta, "phi": phi}, dims=["theta", "phi"])
    if extra_dims:
        for name, vals in extra_dims.items():
            da = da.expand_dims({name: vals})
    return da


def _assert_coords(result, theta_lo, theta_hi, phi_lo, phi_hi):
    assert abs(result.theta.values[0] - theta_lo) < TOL, (
        f"theta start: got {result.theta.values[0]}, expected {theta_lo}"
    )
    assert abs(result.theta.values[-1] - theta_hi) < TOL, (
        f"theta end:   got {result.theta.values[-1]}, expected {theta_hi}"
    )
    assert abs(result.phi.values[0] - phi_lo) < TOL, (
        f"phi start:   got {result.phi.values[0]}, expected {phi_lo}"
    )
    assert abs(result.phi.values[-1] - phi_hi) < TOL, (
        f"phi end:     got {result.phi.values[-1]}, expected {phi_hi}"
    )


# ---------------------------------------------------------------------------
# remap_antenna_pattern
# ---------------------------------------------------------------------------


class TestRemapOutputCoordinates:
    """Output theta and phi are in canonical range."""

    def test_output_theta_min_is_zero(self):
        da = _da(np.arange(-180, 180.1, 10.0), np.arange(-90, 90.1, 10.0))
        r = remap_antenna_pattern(da)
        assert r.theta.min().item() >= 0.0

    def test_output_theta_max_is_180(self):
        da = _da(np.arange(-180, 180.1, 10.0), np.arange(-90, 90.1, 10.0))
        r = remap_antenna_pattern(da)
        assert r.theta.max().item() <= 180.0

    def test_output_phi_min_is_minus180(self):
        da = _da(np.arange(-180, 180.1, 10.0), np.arange(-90, 90.1, 10.0))
        r = remap_antenna_pattern(da)
        assert abs(r.phi.min().item() - (-180.0)) < TOL

    def test_output_phi_never_reaches_180(self):
        """Phi is half-open [-180, 180)."""
        da = _da(np.arange(0, 180.1, 0.5), np.arange(0, 360.0, 0.5))
        r = remap_antenna_pattern(da)
        assert r.phi.max().item() < 180.0


class TestRemapFullCoverage:
    """Full-range inputs produce complete, NaN-free output."""

    def test_full_range_no_nans(self):
        da = _da(np.arange(-180, 180.1, 10.0), np.arange(-90, 90.1, 10.0))
        r = remap_antenna_pattern(da)
        assert np.isnan(r.values).sum() == 0

    def test_user_exact_sampling(self):
        """theta[-180,180] step 0.5, phi[-90,90] step 0.5."""
        theta = np.arange(-180.0, 180.5, 0.5)
        phi = np.arange(-90.0, 90.5, 0.5)
        r = remap_antenna_pattern(_da(theta, phi, np.random.rand(len(theta), len(phi))))
        assert np.isnan(r.values).sum() == 0
        _assert_coords(r, 0.0, 180.0, -180.0, 179.5)

    def test_full_phi_pos_theta(self):
        """phi[0,360), theta[0,180] — all four quadrants from positive theta."""
        theta = np.arange(0.0, 180.5, 0.5)
        phi = np.arange(0.0, 360.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi, np.random.rand(len(theta), len(phi))))
        assert np.isnan(r.values).sum() == 0
        _assert_coords(r, 0.0, 180.0, -180.0, 179.5)

    def test_full_phi_neg_theta(self):
        """phi[-360,0), theta[-180,0) — all four quadrants from negative theta."""
        theta = np.arange(-180.0, 0.0, 0.5)
        phi = np.arange(-360.0, 0.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi, np.random.rand(len(theta), len(phi))))
        assert np.isnan(r.values).sum() == 0
        _assert_coords(r, 0.5, 180.0, -180.0, 179.5)


class TestRemapSingleQuadrant:
    """Single-quadrant inputs remap to the correct output strip."""

    def test_q1_pos_theta_phi_range(self):
        """Q1: theta[0,180], phi[0,90) -> output phi [0, 90)."""
        theta = np.arange(0.0, 180.5, 0.5)
        phi = np.arange(0.0, 90.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi))
        non_nan_phi = r.phi.values[~np.isnan(r).all("theta").values]
        assert abs(non_nan_phi[0] - 0.0) < TOL
        assert abs(non_nan_phi[-1] - 89.5) < TOL

    def test_single_quadrant_pos_theta_no_nans(self):
        """Single-quadrant pos-theta input should produce no NaNs."""
        theta = np.arange(0.0, 180.5, 0.5)
        phi = np.arange(0.0, 90.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi))
        assert np.isnan(r.values).sum() == 0, f"{np.isnan(r.values).sum()} NaNs found"

    def test_q1_neg_theta_flip(self):
        """Q1 via neg-theta: theta[-180,0), phi[-180,-90) -> output phi [0,90)."""
        theta = np.arange(-180.0, 0.0, 0.5)
        phi = np.arange(-180.0, -90.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi))
        assert np.isnan(r.values).sum() == 0
        _assert_coords(r, 0.5, 180.0, 0.0, 89.5)

    def test_q3_pos_theta(self):
        """Q3: theta[0,180], phi[-180,-90) -> output phi [-180,-90)."""
        theta = np.arange(0.0, 180.5, 0.5)
        phi = np.arange(-180.0, -90.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi))
        non_nan_phi = r.phi.values[~np.isnan(r).all("theta").values]
        assert abs(non_nan_phi[0] - (-180.0)) < TOL
        assert abs(non_nan_phi[-1] - (-90.5)) < TOL

    def test_q4_pos_theta(self):
        """Q4: theta[0,180], phi[-90,0) -> output phi [-90, 0)."""
        theta = np.arange(0.0, 180.5, 0.5)
        phi = np.arange(-90.0, 0.0, 0.5)
        r = remap_antenna_pattern(_da(theta, phi))
        non_nan_phi = r.phi.values[~np.isnan(r).all("theta").values]
        assert abs(non_nan_phi[0] - (-90.0)) < TOL
        assert abs(non_nan_phi[-1] - (-0.5)) < TOL


class TestRemapExtraDimensions:
    """Extra dimensions (freq, pol, etc.) pass through unchanged."""

    def test_extra_dims_preserved(self):
        freqs = [1e9, 2e9]
        pols = ["H", "V"]
        theta = np.arange(-180, 180.1, 10.0)
        phi = np.arange(-90, 90.0, 10.0)
        data = np.random.rand(len(freqs), len(pols), len(theta), len(phi))
        da = xr.DataArray(
            data,
            coords={"freq": freqs, "pol": pols, "theta": theta, "phi": phi},
            dims=["freq", "pol", "theta", "phi"],
        )
        r = remap_antenna_pattern(da)
        assert "freq" in r.dims
        assert "pol" in r.dims
        assert list(r.freq.values) == freqs
        assert list(r.pol.values) == pols

    def test_extra_dims_no_nans(self):
        freqs = [1e9]
        pols = ["H"]
        theta = np.arange(-180, 180.1, 10.0)
        phi = np.arange(-90, 90.0, 10.0)
        data = np.ones((len(freqs), len(pols), len(theta), len(phi)))
        da = xr.DataArray(
            data,
            coords={"freq": freqs, "pol": pols, "theta": theta, "phi": phi},
            dims=["freq", "pol", "theta", "phi"],
        )
        r = remap_antenna_pattern(da)
        assert np.isnan(r.values).sum() == 0

    def test_extra_dims_phi_range(self):
        freqs = [1e9]
        pols = ["H"]
        theta = np.arange(-180, 180.1, 10.0)
        phi = np.arange(-90, 90.0, 10.0)
        data = np.ones((len(freqs), len(pols), len(theta), len(phi)))
        da = xr.DataArray(
            data,
            coords={"freq": freqs, "pol": pols, "theta": theta, "phi": phi},
            dims=["freq", "pol", "theta", "phi"],
        )
        r = remap_antenna_pattern(da)
        assert np.isclose(r.phi.min(), -180.0)
        assert np.isclose(r.phi.max(), 170.0)


class TestRemapPhiStepSizes:
    """Different phi step sizes all produce correct output."""

    @pytest.mark.parametrize("step", [1.0, 2.0, 5.0, 10.0])
    def test_step_sizes(self, step):
        theta = np.arange(-180, 180.1, step)
        phi = np.arange(-90, 90.0, step)
        r = remap_antenna_pattern(_da(theta, phi))
        assert r.theta.min().item() >= 0.0
        assert abs(r.phi.min().item() - (-180.0)) < TOL


class TestRemapErrors:
    def test_empty_raises(self):
        da = xr.DataArray(
            np.ones((0, 0)),
            coords={"theta": [], "phi": []},
            dims=["theta", "phi"],
        )
        with pytest.raises(Exception):
            remap_antenna_pattern(da)
