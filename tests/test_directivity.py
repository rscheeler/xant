import numpy as np
import pytest
from hics import HCS
from scipy.spatial.transform import Rotation

from xant import ureg
from xant.antenna import directivity as dr
from xant.antenna.common import Dipole
from xant.antenna.phasedarray import AntennaArray, steer_phase_centers

FS = np.array([2, 3]) * ureg.GHz
LAM = 1 / FS * ureg.speed_of_light
TH = np.arange(0.5, 180, 1.0) * ureg.degree
PH = np.arange(-179, 180, 2.0) * ureg.degree


@pytest.fixture
def mount():
    return HCS((0, 0, 0) * ureg.m, Rotation.from_euler("ZYZ", angles=[45, 90, 0], degrees=True))


def make_array(mount, n=4, spacing=0.5, fs=FS):
    lam = (1 / fs * ureg.speed_of_light)[0]
    return AntennaArray.rectangular(
        Dipole(5 * ureg.cm, fs), n, lam * spacing, n, lam * spacing, cs_reference=mount
    )


def steer(arr, t, p=0.0, f=3 * ureg.GHz):
    steer_phase_centers(arr, frequency=f, phi=[p] * ureg.degree, theta=[t] * ureg.degree)


def brute_trp(arr, fs=FS[1:]):
    """Independent reference: equispaced-grid integration of the actual array pattern."""
    return arr._raw_total.trp(theta=TH, phi=PH, frequency=fs).data.magnitude.squeeze()


def peak_u(ant, mount, t, p, half=15.0, step=0.25):
    th = np.arange(max(t - half, 0.0), t + half + 1e-9, step) * ureg.degree
    ph = np.arange(p - half, p + half + 1e-9, step) * ureg.degree
    d = ant.request_data(theta=th, phi=ph, frequency=3 * ureg.GHz, hcs=mount).sel(
        polarization="apolar"
    )
    v = d.data
    return (np.abs(np.asarray(v.magnitude if hasattr(v, "magnitude") else v)) ** 2).max()


def patch_max(fn, t, p, half=15.0, step=0.25):
    th = np.arange(max(t - half, 0.0), t + half + 1e-9, step) * ureg.degree
    ph = np.arange(p - half, p + half + 1e-9, step) * ureg.degree
    return float(fn(theta=th, phi=ph, frequency=3 * ureg.GHz).values.max())


# ---- kernels -------------------------------------------------------------------------------------
def test_quadrature_exact():
    g = dr.gl_sphere(40)
    th = g.theta[:, None] * np.ones(g.phi.size)
    assert np.isclose(g.weights.sum(), 4 * np.pi)
    assert np.isclose((g.weights * np.cos(th) ** 2).sum(), 4 * np.pi / 3)


def test_isotropic_gram_matches_sinc():
    lam = 0.1
    k = 2 * np.pi / lam
    xs, ys = np.meshgrid(np.arange(8), np.arange(8))
    pos = np.stack([xs.ravel() * lam / 2, ys.ravel() * lam / 2, 0 * xs.ravel()], 1)
    d = np.linalg.norm(pos[None] - pos[:, None], axis=-1)
    grid = dr.gl_sphere(dr.lmax_for(k * np.linalg.norm(pos.max(0) - pos.min(0)), 0))
    g = dr.gram_identical(pos, k, np.ones(grid.shape), grid)
    assert np.abs(g - 4 * np.pi * np.sinc(k * d / np.pi)).max() / (4 * np.pi) < 1e-7


# ---- TRP vs brute force -------------------------------------------------------------------------
@pytest.mark.parametrize("spacing", [0.5, 1.0])  # 1.0 lambda: grating lobes enter visible space
def test_array_trp_matches_far_field_integration(mount, spacing):
    arr = make_array(mount, spacing=spacing)
    steer(arr, 35, 30)
    assert np.allclose(arr.trp().values.squeeze(), brute_trp(arr, FS), rtol=2e-3)


def test_gram_cached_across_resteer(mount):
    arr = make_array(mount)
    arr.trp()
    n = len(arr._gram_cache)
    steer(arr, 20, 10)
    arr.trp()
    assert len(arr._gram_cache) == n  # steering a flat array never rebuilds G


def test_cache_is_bounded(mount):
    arr = make_array(mount, 2)
    for i in range(80):
        arr._cache_put(("x", i), i)
    assert len(arr._gram_cache) <= 64


# ---- hierarchical --------------------------------------------------------------------------------
def make_hier(mount, sub_ref=None, lossless=True):
    sub = AntennaArray.rectangular(
        Dipole(5 * ureg.cm, FS[1:]), 2, LAM[1] / 2, 2, LAM[1] / 2, cs_reference=sub_ref or mount
    )
    return AntennaArray.rectangular(sub, 2, 2.5 * LAM[1], 2, 2.5 * LAM[1], cs_reference=mount)


@pytest.mark.parametrize("sub_ref", [None, "origin"])  # same frame / rotated sub-frame
def test_hierarchical_resteer_matches_brute_force(mount, sub_ref):
    arr = make_hier(mount, HCS((0, 0, 0) * ureg.m) if sub_ref else None)
    for p, t in (
        (40, 30),
        (120, 10),
        (-60, 50),
    ):  # same object re-steered: G must follow sub-array weights
        steer(arr, t, p)
        assert np.allclose(arr.trp().values.squeeze(), brute_trp(arr), rtol=2e-3)


def test_three_level_hierarchy(mount):
    subsub = AntennaArray.rectangular(
        Dipole(5 * ureg.cm, FS[1:]), 2, LAM[1] / 2, 2, LAM[1] / 2, cs_reference=mount
    )
    sub = AntennaArray.rectangular(subsub, 2, 1.5 * LAM[1], 2, 1.5 * LAM[1], cs_reference=mount)
    arr = AntennaArray.rectangular(sub, 2, 3.5 * LAM[1], 2, 3.5 * LAM[1], cs_reference=mount)
    for p, t in ((40, 30), (-120, 20)):
        steer(arr, t, p)
        assert np.allclose(arr.trp().values.squeeze(), brute_trp(arr), rtol=3e-3)


def test_hierarchical_flat_model_is_weight_independent(mount):
    arr = make_hier(mount)
    steer(arr, 30, 40)
    assert arr._models(arr.excitation).__next__()[0] == "flat"
    arr.trp()
    n = len(arr._gram_cache)
    steer(arr, 10, 120)  # re-steers the sub-arrays too, yet G is untouched
    arr.trp()
    assert len(arr._gram_cache) == n


def test_hierarchical_identical_model_when_flat_unavailable(mount):
    a, b = make_hier(mount), make_hier(mount)
    for x in (a, b):
        steer(x, 30, 40)
    b._flat_leaf_model = lambda top: None
    assert b._models(b.excitation).__next__()[0] == "identical"
    assert np.allclose(a.trp().values, b.trp().values, rtol=1e-6)
    steer(b, 10, 120)  # identical model must follow sub-array weights
    assert np.allclose(b.trp().values.squeeze(), brute_trp(b), rtol=2e-3)
    steer(
        b, 25, -70
    )  # ...without re-requesting the sub-array fields: they are cached once, weights applied after
    b.trp()
    assert sum(k[0] == "ports" for k in b._gram_cache if isinstance(k, tuple)) == 1


def test_wrong_flat_model_is_rejected_by_self_check(mount):
    a, b = make_hier(mount), make_hier(mount)
    for x in (a, b):
        steer(x, 30, 40)
    good = b._flat_leaf_model

    def bad(top):  # plausible but wrong: sub-array offsets dropped
        leaf, pos, w = good(top)
        return (
            leaf,
            pos + np.tile([0.0, 0.01, 0.0], (pos.shape[0], 1)) * np.arange(pos.shape[0])[:, None],
            w,
        )

    b._flat_leaf_model = bad
    assert np.allclose(
        a.trp().values, b.trp().values, rtol=1e-6
    )  # fell through to a model that is right


# ---- self-checking fallbacks ---------------------------------------------------------------------
def test_fallback_to_per_port_when_model_check_fails(mount):
    a, b = make_array(mount, 3), make_array(mount, 3)
    steer(a, 30)
    steer(b, 30)
    b._model_matches = lambda *a, **k: False
    assert np.allclose(a.trp().values, b.trp().values, rtol=1e-6)


def test_fallback_to_quadrature_when_gram_fails(mount):
    a, b = make_array(mount, 3), make_array(mount, 3)
    steer(a, 30)
    steer(b, 30)

    def boom(*a, **k):
        raise NotImplementedError("nope")

    b._model_matches = lambda *a, **k: False
    b._dense_gram = boom
    assert np.allclose(a.trp().values.squeeze(), b.trp().values.squeeze(), rtol=2e-3)


# ---- gain / scan loss ----------------------------------------------------------------------------
@pytest.mark.parametrize("spacing", [0.5, 1.0])
def test_gain_at_steer_matches_brute_force(mount, spacing):
    arr = make_array(mount, 4, spacing, FS[1:])
    for t in (0, 40, 60):
        steer(arr, t)
        g = float(arr.directivity(frequency=3 * ureg.GHz).values.squeeze())
        tot = arr._raw_total.request_data(
            theta=[t] * ureg.degree, phi=[0] * ureg.degree, frequency=3 * ureg.GHz, hcs=mount
        ).sel(polarization="apolar")
        v = tot.data
        u = float(np.abs(np.asarray(v.magnitude if hasattr(v, "magnitude") else v)).squeeze() ** 2)
        assert np.isclose(g, 4 * np.pi * u / brute_trp(arr), rtol=3e-3)


def test_grating_lobes_reduce_scanned_gain(mount):
    """At 1 lambda spacing the scanned beam loses far more gain than at lambda/2 (power goes to the grating lobe)."""
    loss = {}
    for spacing in (0.5, 1.0):
        arr = make_array(mount, 4, spacing, FS[1:])
        steer(arr, 0)
        g0 = float(arr.directivity(frequency=3 * ureg.GHz).values.squeeze())
        steer(arr, 60)
        g60 = float(arr.directivity(frequency=3 * ureg.GHz).values.squeeze())
        loss[spacing] = 10 * np.log10(g0 / g60)
    assert loss[1.0] > loss[0.5] + 3.0


def test_normalized_total_follows_scan_and_resteering(mount):
    arr = make_array(mount, 4, 1.0, FS[1:])
    arr.normalize_total = "directivity"
    for t in (0, 40, 60):
        steer(arr, t)
        assert np.isclose(
            peak_u(arr.total, mount, t, 0), patch_max(arr.directivity, t, 0), rtol=1e-3
        )


def test_gain_mode_keeps_element_efficiency(mount):
    eta_el = 0.5
    lossy = Dipole(5 * ureg.cm, FS[1:]) * np.sqrt(eta_el)
    base_trp = float(
        Dipole(5 * ureg.cm, FS[1:]).trp(frequency=3 * ureg.GHz).data.magnitude.squeeze()
    )
    eta = eta_el * base_trp / (4 * np.pi)
    arr = AntennaArray.rectangular(lossy, 4, LAM[1] / 2, 4, LAM[1] / 2, cs_reference=mount)
    steer(arr, 40)
    d = float(arr.directivity(frequency=3 * ureg.GHz).values.squeeze())
    g = float(arr.gain(frequency=3 * ureg.GHz).values.squeeze())
    assert np.isclose(g / d, eta, rtol=2e-3)
    assert np.isclose(
        arr._radiated_element_power(3 * ureg.GHz).values.squeeze(), 4 * np.pi * eta, rtol=2e-3
    )
    arr.normalize_total = "directivity"
    pk_d = peak_u(arr.total, mount, 40, 0)
    arr.normalize_total = "gain"
    assert np.isclose(peak_u(arr.total, mount, 40, 0) / pk_d, eta, rtol=2e-3)


def test_gain_mode_hierarchical_lossy(mount):
    lossy = Dipole(5 * ureg.cm, FS[1:]) * np.sqrt(0.5)
    eta = float(lossy.trp(frequency=3 * ureg.GHz).data.magnitude.squeeze()) / (4 * np.pi)
    for sub_ref in (mount, HCS((0, 0, 0) * ureg.m)):
        sub = AntennaArray.rectangular(lossy, 2, LAM[1] / 2, 2, LAM[1] / 2, cs_reference=sub_ref)
        arr = AntennaArray.rectangular(sub, 2, 2.5 * LAM[1], 2, 2.5 * LAM[1], cs_reference=mount)
        steer(arr, 30, 40)
        assert np.isclose(
            arr._radiated_element_power(3 * ureg.GHz).values.squeeze(), 4 * np.pi * eta, rtol=2e-3
        )


def test_gain_is_the_default_and_can_be_disabled(mount):
    arr = make_array(mount, 4, 1.0, FS[1:])
    assert arr.normalize_total == "gain"
    steer(arr, 60)
    g = patch_max(arr.gain, 60, 0)
    assert np.isclose(peak_u(arr.total, mount, 60, 0), g, rtol=1e-3)
    arr.normalize_total = False
    assert not np.isclose(peak_u(arr.total, mount, 60, 0), g, rtol=0.1)


def test_request_without_frequency_and_copy_keep_setting(mount):
    arr = make_array(mount, 3)
    dat = arr.total.request_data(
        theta=np.array([10.0, 40.0]) * ureg.degree, phi=np.array([0.0, 90.0]) * ureg.degree
    )
    assert "frequency" in dat.dims and dat.sizes["frequency"] == FS.size
    assert arr.copy().normalize_total == "gain"
    arr.normalize_total = "directivity"
    assert arr.copy().normalize_total == "directivity"


def test_per_port_element_and_out_of_band_frequency():
    from xant.antenna.common import DipoleAboveGround

    fs = np.linspace(1400, 1600, 5) * ureg.MHz
    hcs = HCS((0, 0, 0) * ureg.m)
    el_hcs = HCS(
        (0, 0, 0) * ureg.m,
        rotation=Rotation.from_euler("ZYZ", [45, 0, 0], degrees=True),
        reference=hcs,
    )
    lam = 1.2 * ureg.speed_of_light / fs[0]
    element = DipoleAboveGround(lam / 2, lam / 4, fs, hcs=el_hcs)
    cs = [
        HCS(
            (0, 0, 0.1 * n) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [45, -90, 0], degrees=True),
            reference=hcs,
        )
        for n in range(2)
    ]
    arr = AntennaArray(element, cs)
    assert arr._identical_element() is None  # per-port element -> per-port path
    f = 1.5 * ureg.GHz
    th = np.arange(0.5, 180, 2.0) * ureg.degree
    ph = np.arange(-179, 180, 4.0) * ureg.degree
    ref = arr._raw_total.trp(theta=th, phi=ph, frequency=f).data.magnitude.squeeze()
    assert np.isclose(arr.trp(frequency=f).data.magnitude.squeeze(), ref, rtol=2e-3)
    out = arr.total.request_data(
        theta=[36] * ureg.degree, phi=[66] * ureg.degree, frequency=3.7 * ureg.GHz
    )
    assert "frequency" in out.dims


def test_total_d0_and_trp_use_the_exact_array_path(mount):
    """arr.total.d0()/trp() with no grid must not fall back to the coarse legacy rectangle rule."""
    arr = make_array(mount, 6, 1.0, FS[1:])
    for t in (0, 3, 40):
        steer(arr, t)
        exact = float(arr.directivity(frequency=3 * ureg.GHz).values.squeeze())
        assert np.isclose(
            float(arr.total.d0(frequency=3 * ureg.GHz).data.magnitude.squeeze()), exact, rtol=1e-9
        )
        assert np.isclose(
            float(arr.d0(frequency=3 * ureg.GHz).data.magnitude.squeeze()), exact, rtol=1e-9
        )
    # gain-normalised total reports its own TRP (radiated element power); an explicit grid keeps the legacy rule
    eta4pi = float(arr._radiated_element_power(3 * ureg.GHz).values.squeeze())
    assert np.isclose(
        float(arr.total.trp(frequency=3 * ureg.GHz).data.magnitude.squeeze()), eta4pi, rtol=1e-9
    )
    legacy = arr.total.trp(theta=TH, phi=PH, frequency=3 * ureg.GHz)
    assert np.isclose(float(legacy.data.magnitude.squeeze()), eta4pi, rtol=5e-3)


def test_moving_parent_frame_keeps_fast_path(monkeypatch):
    """
    An array attached to a time-varying frame (e.g. a satellite) carries a constant ``time`` axis in every pattern
    request.  That must not defeat the Gram fast path (the quadrature fallback is O(elements * time) memory).
    """
    import pandas as pd
    import xarray as xr
    from hics import GLOBAL_CS

    ts = pd.date_range("2026-01-01", periods=3, freq="10s")
    rots = [Rotation.from_euler("ZYZ", [10 * k, 20 * k, 5 * k], degrees=True) for k in range(3)]
    parent = HCS(
        xr.DataArray(
            np.array([[1e6 * (k + 1), 2e5, 7e6] for k in range(3)]) * ureg.m,
            dims=("time", "position"),
            coords=dict(time=np.asarray(ts), position=["x", "y", "z"]),
        ),
        rotation=xr.DataArray(rots, dims=("time",), coords=dict(time=np.asarray(ts))),
        reference=GLOBAL_CS,
    )
    rot = Rotation.from_euler("ZYZ", angles=[45, 90, 0], degrees=True)
    lam = (1 / FS[1:] * ureg.speed_of_light)[0]

    def build(ref):
        el = Dipole(
            5 * ureg.cm, FS[1:], hcs=ref
        )  # element attached to the array frame, as in the satellite example
        return AntennaArray.rectangular(el, 4, lam, 4, lam, cs_reference=ref)

    moving = build(HCS((0, 0, 0) * ureg.m, rot, reference=parent))
    static = build(HCS((0, 0, 0) * ureg.m, rot))

    def no_quadrature(*a, **k):
        raise AssertionError("fell back to quadrature")

    monkeypatch.setattr(type(moving), "_trp_quadrature", no_quadrature)
    for a in (moving, static):
        steer(a, 20, 30)
    f = 3 * ureg.GHz
    assert np.allclose(
        moving.trp(frequency=f).values.squeeze(),
        static.trp(frequency=f).values.squeeze(),
        rtol=1e-8,
    )
