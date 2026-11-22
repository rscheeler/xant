"""Regression tests for defects found in review (all offline)."""

import warnings

import numpy as np
import pytest
import xarray as xr
from hics import GLOBAL_CS

from xant import Antenna, ureg
from xant.antenna.common import Cosine, Dipole, Isotropic, TE10Aperture
from xant.antenna.phasedarray import AntennaArray, steer_phase_centers
from xant.propagation.itm import itm

TH = np.linspace(0, 180, 181) * ureg.degree
PH = np.arange(-180, 180, 5) * ureg.degree


def _peak_dbi(ant, theta=TH, phi=PH):
    g = ant.request_data(theta=theta, phi=phi, coordinate_frame="phitheta").sel(
        polarization="apolar"
    )
    return 20 * np.log10(float(np.abs(g).max().data.magnitude))


@pytest.mark.parametrize(
    ("length_wl", "d0_dbi"), [(0.1, 1.77), (0.5, 2.15), (1.0, 3.82), (1.25, 5.16)]
)
def test_dipole_peak_gain_is_directivity(length_wl, d0_dbi):
    """Peak |E|^2 must equal the Balanis directivity for every length, not just the half-wave dipole."""
    f = np.array([299.792458e6]) * ureg.Hz  # 1 m wavelength
    assert _peak_dbi(Dipole(l=length_wl * ureg.m, frequency=f)) == pytest.approx(d0_dbi, abs=0.03)


def test_cosine_has_no_back_lobe():
    f = np.array([1e9]) * ureg.Hz
    for n in (1, 2, 3, 4):
        g = Cosine(f, n=n).request_data(theta=TH, phi=PH, coordinate_frame="phitheta")
        back = np.abs(g.sel(polarization="apolar").sel(theta=np.pi, method="nearest")).max()
        assert float(back.data.magnitude) == pytest.approx(0.0, abs=1e-12)


def test_te10_aperture_efficiency():
    f = np.array([299.792458e6]) * ureg.Hz
    ant = TE10Aperture(3 * ureg.m, 2 * ureg.m, f)
    expect = 10 * np.log10(8 / np.pi**2 * 4 * np.pi * 6)
    assert _peak_dbi(ant, np.linspace(0, 90, 91) * ureg.degree) == pytest.approx(expect, abs=0.05)


@pytest.mark.parametrize("steer", [0.0, 30.0, 50.0])
def test_beamwidth_of_steered_array(steer):
    f = np.array([3e9]) * ureg.Hz
    lam = (ureg.speed_of_light / f).to("m")
    arr = AntennaArray.rectangular(
        Isotropic(frequency=f), nx=16, dx=0.5 * lam, ny=1, dy=0.5 * lam, cs_reference=GLOBAL_CS
    )
    steer_phase_centers(arr, frequency=f[0], theta=steer * ureg.degree, phi=0 * ureg.degree)
    bw = arr.total.beamwidth(dim="theta", coordinate_frame="phitheta", phi=0 * ureg.degree)
    expect = np.rad2deg(0.886 / (16 * 0.5)) / np.cos(np.deg2rad(steer))
    assert bw.to("degree").magnitude == pytest.approx(expect, rel=0.02)


def _data_antenna(freqs, pols=("theta", "phi")):
    th = np.deg2rad(np.linspace(0, 180, 37))
    ph = np.deg2rad(np.arange(-180, 180, 10.0))
    fr = np.array(freqs, dtype=float)
    d = np.zeros((len(pols), fr.size, th.size, ph.size), dtype=complex)
    d[0] = (fr / 1e9)[:, None, None]  # amplitude linear in frequency
    return Antenna(
        xr.DataArray(
            d,
            dims=("polarization", "frequency", "theta", "phi"),
            coords=dict(polarization=list(pols), frequency=fr, theta=th, phi=ph),
            attrs=dict(coordinate_frame="phitheta"),
        ),
    )


def _amp(ant, f):
    g = ant.request_data(
        theta=np.array([30.0]) * ureg.degree,
        phi=np.array([0.0]) * ureg.degree,
        frequency=np.array(f),
    )
    return np.abs(g.sel(polarization="apolar").values).ravel()


def test_interpolation_on_nonuniform_and_descending_grids():
    # five uniform points: the spline reproduces a linear trend away from the ends
    np.testing.assert_allclose(
        _amp(_data_antenna([1e9, 2e9, 3e9, 4e9, 5e9]), [2.4e9, 3.6e9]), [2.4, 3.6], rtol=0.02
    )
    # non-uniform grid used to snap to the nearest sample (2.0 and 3.0 here)
    got = _amp(_data_antenna([1e9, 2e9, 3e9, 5e9, 8e9]), [2.4e9, 3.6e9])
    assert abs(got[0] - 2.0) > 0.2
    assert abs(got[1] - 3.0) > 0.2
    # a descending axis must give the same answer as the ascending one
    np.testing.assert_allclose(
        _amp(_data_antenna([5e9, 4e9, 3e9, 2e9, 1e9]), [2.4e9, 3.6e9]),
        _amp(_data_antenna([1e9, 2e9, 3e9, 4e9, 5e9]), [2.4e9, 3.6e9]),
        rtol=1e-9,
    )


def test_apolar_measured_pattern_loads():
    ant = _data_antenna([1e9, 2e9], pols=("apolar",))
    assert _amp(ant, [1e9])[0] == pytest.approx(1.0, rel=1e-6)


def _profile(elev, d_km, htx, hrx):
    elev = np.asarray(elev, dtype=float)
    da = xr.DataArray(
        elev * ureg.m, dims="distance", coords=dict(distance=np.linspace(0, d_km * 1e3, elev.size))
    )
    da.attrs = dict(
        txagl=htx, rxagl=hrx, txamsl=elev[0] + htx, rxamsl=elev[-1] + hrx, distance_km=d_km
    )
    return da


def test_itm_horizon_distance_index():
    flat = np.zeros(301)
    flat[100] = 200.0  # single obstacle 10 km along a 30 km path
    pfl, _ = itm.take_magnitudes(
        (_profile(flat, 30, 10.0, 10.0), ureg.m), (900 * ureg.MHz, ureg.MHz)
    )
    pfl.attrs["distance_m"] = 30e3
    _, d_hzn = itm.find_horizons(pfl, 8.5e6)
    np.testing.assert_allclose(d_hzn, [10e3, 20e3])


def test_itm_los_rescaling_branch_and_integer_heights():
    """Line-of-sight path whose smooth-earth horizons fall short of the path used to raise TypeError."""
    x = np.linspace(0, 1, 301)
    basin = 900 * (2 * x - 1) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = itm.itm_p2p(_profile(basin, 200, 2, 2), 5, 301, 900 * ureg.MHz, 1, 15, 0.005, mdvar=3)
    loss = float(res.squeeze())
    assert np.isfinite(loss)
    assert loss > float(res.fspl.data.magnitude) - 1e-6


def test_troposcatter_returns_h0():
    out = itm.troposcatter_loss(
        600e3,
        np.array([0.0, 0.0]),
        np.array([40e3, 40e3]),
        np.array([30.0, 30.0]),
        8.5e6,
        301.0,
        900.0,
        -0.0094,
        -1,
    )
    assert len(out) == 2
    assert out[1] >= 0
