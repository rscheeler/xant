"""Offline tests for the satellite / 3GPP propagation models (no terrain download needed)."""

import numpy as np
import pytest
import xarray as xr
from hics import HCS

from xant import ureg
from xant.antenna.common import Isotropic
from xant.propagation import propagators
from xant.propagation import satcom as S
from xant.propagation.itm.p2108 import p2108_2, p2108_3
from xant.propagation.rflink import calculate_spatial_link


# ---------------------------------------------------------------- kernels: TR 38.811
def test_slant_range():
    # overhead: slant range is the altitude; horizon-grazing: sqrt(h^2 + 2 h R)
    assert S.slant_range(90.0, 600e3) == pytest.approx(600e3)
    assert S.slant_range(0.0, 600e3) == pytest.approx(np.sqrt(600e3**2 + 2 * 600e3 * S.R_EARTH_M))


def test_ntn_tables():
    # Table 6.6.1-1
    np.testing.assert_allclose(
        S.ntn_los_probability([10, 50, 90], "dense_urban"), [0.282, 0.537, 0.981]
    )
    np.testing.assert_allclose(S.ntn_los_probability([10, 90], "rural"), [0.782, 0.998])
    # Tables 6.6.2-x, S-band below 6 GHz and Ka-band above
    assert [float(v) for v in S.ntn_shadow_clutter(2.0, 10.0, "dense_urban")] == [3.5, 15.5, 34.3]
    assert [float(v) for v in S.ntn_shadow_clutter(20.0, 30.0, "suburban")] == [1.9, 11.2, 21.9]
    assert [float(v) for v in S.ntn_shadow_clutter(30.0, 90.0, "urban")] == [4.0, 6.0, 32.9]
    # nearest reference angle by default, interpolation on request
    assert float(S.ntn_shadow_clutter(2.0, 34.0, "urban")[2]) == 29.0
    assert float(S.ntn_shadow_clutter(2.0, 35.0, "urban", interpolate=True)[2]) == pytest.approx(
        28.35
    )
    # below the horizon
    assert np.isnan(S.ntn_los_probability(-5.0, "urban"))


def test_ntn_excess_loss_percentiles():
    # LOS median is zero, NLOS median is the clutter loss
    assert float(S.ntn_excess_loss(2.0, 30.0, "urban", True, 50)) == pytest.approx(0.0, abs=1e-9)
    assert float(S.ntn_excess_loss(2.0, 30.0, "urban", False, 50)) == pytest.approx(29.0)
    # 90 % of locations: + 1.2816 sigma
    assert float(S.ntn_excess_loss(2.0, 30.0, "urban", False, 90)) == pytest.approx(
        29.0 + 1.2816 * 6, abs=1e-3
    )
    # mixture percentile: the mixture CDF evaluated at the returned loss must give the percentile back
    from scipy.special import ndtr

    for q in (10, 50, 90, 99):
        x = float(S.ntn_excess_loss(2.0, 30.0, "urban", "mixture", q))
        cdf = 0.493 * ndtr(x / 4.0) + (1 - 0.493) * ndtr((x - 29.0) / 6.0)
        assert cdf == pytest.approx(q / 100, abs=1e-9)
    # and it sits between the pure LOS and pure NLOS answers
    assert 0.0 < float(S.ntn_excess_loss(2.0, 30.0, "urban", "mixture", 50)) < 29.0


def test_ntn_scintillation():
    assert float(S.ntn_tropo_scintillation(20.0, 10.0)) == 1.08
    assert float(S.ntn_tropo_scintillation(2.0, 10.0)) == 0.0
    assert float(S.ntn_iono_scintillation(4.0)) == pytest.approx(1.1 / np.sqrt(2))
    assert float(S.ntn_iono_scintillation(20.0)) == 0.0


# ---------------------------------------------------------------- kernels: ITU-R
def test_p681_roadside():
    # 1.5 GHz, 45 deg: M = 3.7775, N = 14.825
    np.testing.assert_allclose(
        S.p681_roadside_tree_shadowing(1.5, 45.0, [1, 10, 20]),
        [14.825, 14.825 - 3.7775 * np.log(10), 14.825 - 3.7775 * np.log(20)],
        rtol=1e-9,
    )
    # 20-80 % branch is continuous at 20 % and reaches zero at 80 %
    a20 = S.p681_roadside_tree_shadowing(1.5, 45.0, 20.0)
    assert S.p681_roadside_tree_shadowing(1.5, 45.0, 20.0001) == pytest.approx(a20, rel=1e-4)
    assert S.p681_roadside_tree_shadowing(1.5, 45.0, 80.0) == pytest.approx(0.0, abs=1e-12)
    # more fade at higher frequency and lower elevation; under 20 deg uses the 20 deg value
    assert S.p681_roadside_tree_shadowing(10.0, 45.0, 10) > S.p681_roadside_tree_shadowing(
        1.5, 45.0, 10
    )
    assert S.p681_roadside_tree_shadowing(1.5, 10.0, 10) == S.p681_roadside_tree_shadowing(
        1.5, 20.0, 10
    )
    # Table 1 at 80 deg, zero at zenith, NaN where the Recommendation gives nothing
    assert S.p681_roadside_tree_shadowing(1.6, 80.0, 5.0) == pytest.approx(2.0)
    assert S.p681_roadside_tree_shadowing(2.6, 90.0, 5.0) == pytest.approx(0.0)
    assert np.isnan(S.p681_roadside_tree_shadowing(10.0, 70.0, 10.0))
    assert np.isnan(S.p681_roadside_tree_shadowing(30.0, 45.0, 10.0))


def test_p2108():
    # section 3.2: median unchanged, and loss *not exceeded* must grow with the percentage of locations
    assert p2108_2(3 * ureg.GHz, 0.5 * ureg.km, 0.5).magnitude == pytest.approx(
        26.6281195, abs=1e-6
    )
    lo, hi = p2108_2(3 * ureg.GHz, 0.5 * ureg.km, np.array([0.1, 0.9])).magnitude
    assert hi > lo
    # distance is capped at 2 km, array inputs work
    np.testing.assert_allclose(
        p2108_2(3 * ureg.GHz, np.array([2.0, 5.0]) * ureg.km, 0.5).magnitude[0],
        p2108_2(3 * ureg.GHz, np.array([2.0, 5.0]) * ureg.km, 0.5).magnitude[1],
    )
    # section 3.3: zero at zenith for the median, falls with elevation, grows with percentage
    loss = p2108_3(30 * ureg.GHz, np.array([0.0, 10.0, 45.0, 90.0]) * ureg.degree, 0.5).magnitude
    assert loss[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(loss) < 0)
    # hand evaluation of eq. (7) at 30 GHz, 45 deg, 50 %
    k1 = 93 * 30**0.175
    expect = (-k1 * np.log(0.5) / np.tan(0.05 * 0.5 + np.pi / 4)) ** 0.25 - 1
    assert loss[2] == pytest.approx(expect)
    assert p2108_3(30 * ureg.GHz, 10 * ureg.degree, 0.9) > p2108_3(
        30 * ureg.GHz, 10 * ureg.degree, 0.5
    )


# ---------------------------------------------------------------- kernels: TR 38.901
def test_tr38901_path_loss():
    # hand-evaluated values of Table 7.4.1-1
    d3 = np.hypot(100.0, 25 - 1.5)
    assert S.tr38901_path_loss("uma", 100.0, 3.5) == pytest.approx(
        28 + 22 * np.log10(d3) + 20 * np.log10(3.5)
    )
    d3 = np.hypot(100.0, 10 - 1.5)
    assert S.tr38901_path_loss("umi", 100.0, 28.0, los=False) == pytest.approx(
        35.3 * np.log10(d3) + 22.4 + 21.3 * np.log10(28.0),
    )
    assert S.tr38901_path_loss("rma", 1000.0, 0.7) == pytest.approx(93.38, abs=0.01)
    d3 = np.hypot(10.0, 3 - 1.5)
    assert S.tr38901_path_loss("inh", 10.0, 60.0) == pytest.approx(
        32.4 + 17.3 * np.log10(d3) + 20 * np.log10(60)
    )
    # NLOS never below LOS; LOS probability
    d = np.logspace(1, 3.5, 30)
    for sc in ("rma", "uma", "umi"):
        assert np.all(
            S.tr38901_path_loss(sc, d, 6.0, los=False) >= S.tr38901_path_loss(sc, d, 6.0) - 1e-9
        )
    assert S.tr38901_los_probability("uma", 100.0) == pytest.approx(0.18 + np.exp(-100 / 63) * 0.82)
    assert S.tr38901_los_probability("umi", 10.0) == 1.0
    assert S.tr38901_los_probability("rma", 1010.0) == pytest.approx(np.exp(-1))


# ---------------------------------------------------------------- propagators
@pytest.fixture(scope="module")
def sat_link():
    f = np.array([2e9, 20e9]) * ureg.Hz
    sat = HCS.from_crs((40.0 * ureg.degree, -105.0 * ureg.degree, 600e3 * ureg.m), hagl=False)
    n = 5
    lat = xr.DataArray(np.linspace(40, 55, n) * ureg.degree, dims="pt")
    lon = xr.DataArray(np.full(n, -105.0) * ureg.degree, dims="pt")
    h = xr.DataArray(np.full(n, 1600.0) * ureg.m, dims="pt")
    gs = HCS.from_crs([lat, lon, h], hagl=False)
    return f, sat, gs


def test_link_geometry(sat_link):
    f, sat, gs = sat_link
    for a, b in ((sat, gs), (gs, sat)):  # which end is "tx" must not matter
        geo = S.link_geometry(a, b)
        el = geo["elevation"].values
        assert el[0] == pytest.approx(90.0, abs=1e-3)
        assert np.all(np.diff(el) < 0)
        assert geo["h"].values[0] == pytest.approx(1600.0, abs=1e-3)
    # consistent with the closed-form slant range (spherical earth, so only approximately)
    d = propagators.fspl(txcs=sat, rxcs=gs, frequency=xr.DataArray(f, dims="frequency")).distance
    np.testing.assert_allclose(
        S.slant_range(el, 600e3 - 1600.0, S.R_EARTH_M + 1600.0),
        d.data.to("m").magnitude,
        rtol=5e-3,
    )


def test_ntn_propagator_in_link(sat_link):
    f, sat, gs = sat_link
    tx, rx = Isotropic(frequency=f, hcs=sat), Isotropic(frequency=f, hcs=gs)
    ref, _, _ = calculate_spatial_link(tx, power=1 * ureg.watt, rx=rx, propagation="fspl")
    res, _, _ = calculate_spatial_link(
        tx,
        power=1 * ureg.watt,
        rx=rx,
        propagation="ntn_tr38811",
        scenario="urban",
        los=False,
        location=50,
    )

    def db(da):
        out = da.copy()
        out.data = out.data.to("dB").magnitude
        return out.transpose("pt", "frequency")

    extra = (db(res.prop_loss) - db(ref.prop_loss)).values
    el = res.prop_loss.elevation.values
    for i, fi in enumerate((2.0, 20.0)):
        np.testing.assert_allclose(extra[:, i], S.ntn_shadow_clutter(fi, el, "urban")[2], atol=1e-6)

    # received power drops by exactly the excess loss
    def dbm(da):
        out = da.copy()
        out.data = out.data.to("dBm").magnitude
        return out.transpose("pt", "frequency")

    np.testing.assert_allclose((dbm(ref.rx_power) - dbm(res.rx_power)).values, extra, atol=1e-6)


def test_other_propagators_run(sat_link):
    f, sat, gs = sat_link
    fda = xr.DataArray(f, dims="frequency", coords=dict(frequency=f))
    lms = propagators.lms_p681(txcs=sat, rxcs=gs, frequency=fda, percent_exceeded=10)
    assert lms.data.units == ureg.dB
    assert "tree_shadowing" in lms.coords
    terr = propagators.tr38901(txcs=sat, rxcs=gs, frequency=fda, scenario="uma", los="mean")
    assert np.all(np.isfinite(terr.data.magnitude))
    add = propagators.ntn_tr38811(
        txcs=sat, rxcs=gs, frequency=fda, additional_loss=3 * ureg.dB, los=True
    )
    base = propagators.ntn_tr38811(txcs=sat, rxcs=gs, frequency=fda, los=True)
    np.testing.assert_allclose(add.data.magnitude - base.data.magnitude, 3.0, atol=1e-9)


def test_itur_atmosphere(sat_link):
    pytest.importorskip("itur")
    f, sat, gs = sat_link
    fda = xr.DataArray(f, dims="frequency", coords=dict(frequency=f))
    # gases: zenith / sin(elevation), more at 20 GHz than at 2 GHz
    g = S.atmosphere_gas_standard([2.0, 20.0], 90.0)
    assert 0 < g[0] < g[1] < 1.0
    assert S.atmosphere_gas_standard(20.0, 30.0) == pytest.approx(2 * g[1])
    out = propagators.earth_space_itur(txcs=sat, rxcs=gs, frequency=fda, p=1.0)
    atm = out.atmosphere.transpose("frequency", "pt").values
    assert np.all(atm > 0)
    assert np.all(atm[1] > atm[0])  # 20 GHz worse than 2 GHz
    assert np.all(np.diff(atm[1]) > 0)  # worse at lower elevation
    for k in ("gas", "cloud", "rain", "scintillation"):
        assert k in out.coords


def test_mask_los_satellite():
    """A GEO satellite at ~44 degrees elevation is visible; the small-height horizon shortcut used to mask it."""
    f = xr.DataArray(np.array([2e9]) * ureg.Hz, dims="frequency")
    gs = HCS.from_crs((40.0 * ureg.degree, -105.0 * ureg.degree, 1600 * ureg.m), hagl=False)
    geo = HCS.from_crs((0.0 * ureg.degree, -105.0 * ureg.degree, 35786e3 * ureg.m), hagl=False)
    hidden = HCS.from_crs((10.0 * ureg.degree, -105.0 * ureg.degree, 550e3 * ureg.m), hagl=False)
    pl = propagators.fspl(txcs=geo, rxcs=gs, frequency=f)
    assert np.all(np.isfinite(np.asarray(propagators.mask_los(pl, geo, gs).values, dtype=float)))
    pl = propagators.fspl(txcs=hidden, rxcs=gs, frequency=f)
    assert np.all(np.isnan(np.asarray(propagators.mask_los(pl, hidden, gs).values, dtype=float)))


class _FakeAtmosphere:
    """Constant-attenuation backend: lets the plumbing be tested without the itur package."""

    def gas_standard(self, f_ghz, elevation_deg):
        f, e = np.broadcast_arrays(np.asarray(f_ghz, float), np.asarray(elevation_deg, float))
        return np.where(e > 0, 0.1 * f / np.sin(np.deg2rad(np.where(e > 0, e, 90.0))), np.nan)

    def slant_path(self, lat_deg, lon_deg, f_ghz, elevation_deg, p, antenna_diameter_m):
        el = np.asarray(elevation_deg, float)
        base = np.where(el > 0, f_ghz / p, np.nan)
        return dict(gas=base, cloud=base, rain=base, scintillation=base, total=4 * base)


def test_atmosphere_backend_is_pluggable(sat_link):
    f, sat, gs = sat_link
    fda = xr.DataArray(f, dims="frequency", coords=dict(frequency=f))
    default = S.get_atmosphere_backend()
    try:
        S.set_atmosphere_backend(_FakeAtmosphere())
        assert S.atmosphere_gas_standard(10.0, 90.0) == pytest.approx(1.0)
        out = propagators.earth_space_itur(txcs=sat, rxcs=gs, frequency=fda, p=1.0)
        fg = np.asarray(f.to("GHz").magnitude)
        atm = out.atmosphere.transpose("frequency", "pt").values
        np.testing.assert_allclose(atm, 4 * fg[:, None] * np.ones_like(atm))
        ntn = propagators.ntn_tr38811(txcs=sat, rxcs=gs, frequency=fda, los=True, atmosphere=True)
        assert np.all(np.isfinite(ntn.gas.values))
        with pytest.raises(TypeError):
            S.set_atmosphere_backend(object())
    finally:
        S.set_atmosphere_backend(None)
    assert isinstance(S.get_atmosphere_backend(), S.ItuRpyBackend)
    assert not isinstance(default, _FakeAtmosphere)


def test_atmosphere_default_backend_errors_clearly_without_itur(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "itur":
            raise ImportError("no itur")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="pip install"):
        S.atmosphere_gas_standard(10.0, 30.0)
