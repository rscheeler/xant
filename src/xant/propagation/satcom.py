"""
Propagation models for satellite / non-terrestrial links and 3GPP terrestrial scenarios.

Two layers:

* **Kernels** -- plain numpy functions (GHz, degrees, metres, dB in and out; arrays broadcast). No coordinate
  systems, no units, easy to test against the source documents:

  ================================  =================================================================
  ``slant_range``                   TR 38.811 eq. (6.6-3) slant range from elevation and altitude
  ``ntn_los_probability``           TR 38.811 Table 6.6.1-1
  ``ntn_shadow_clutter``            TR 38.811 Tables 6.6.2-1/2/3 (sigma_SF, clutter loss)
  ``ntn_excess_loss``               shadow fading + clutter as a location percentile (LOS / NLOS / mixture)
  ``ntn_tropo_scintillation``       TR 38.811 Table 6.6.6.2.1-1 (99 %, 20 GHz, Toulouse)
  ``ntn_iono_scintillation``        TR 38.811 section 6.6.6.1 low-latitude shortcut (below 6 GHz)
  ``p681_roadside_tree_shadowing``  ITU-R P.681 section 4.1.1 (0.8-20 GHz land-mobile-satellite)
  ``p681_building_blockage``        ITU-R P.681 section 4.2 percentage blockage by roadside buildings
  ``tr38901_path_loss``             TR 38.901 Table 7.4.1-1 (RMa, UMa, UMi, InH; 0.5-100 GHz)
  ``tr38901_los_probability``       TR 38.901 Table 7.4.2-1
  ``atmosphere_slant_path``         gases (P.676), cloud (P.840), rain (P.618/P.838), scintillation (P.618)
  ``atmosphere_gas_standard``       standard-atmosphere gases (TR 38.811 section 6.6.4)
  ================================  =================================================================

  The two ``atmosphere_*`` functions go through a pluggable backend (``set_atmosphere_backend``). The default
  wraps the optional ``itur`` package (``pip install xant[itur]``) as ``itur_slant_path`` / ``itur_gas_standard``;
  everything else in this module needs only numpy and scipy.

  Earth-space clutter per ITU-R P.2108 section 3.3 is ``xant.propagation.itm.p2108.p2108_3``.

* **Propagators** -- same call signature as ``propagators.fspl`` (``txcs``, ``rxcs``, ``frequency``,
  ``additional_loss``), so they plug into ``calculate_spatial_link(..., propagation="ntn_tr38811", ...)``:
  ``ntn_tr38811``, ``lms_p681``, ``earth_space_itur`` and ``tr38901``. Each returns free-space loss plus the
  model's excess loss in dB, with the elevation angle and the individual terms attached as coordinates.

All percentile arguments are percentages. "Loss not exceeded for ``location`` % of locations" is used throughout,
so a larger percentage is a more conservative (larger) loss.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity
from scipy.special import ndtr, ndtri
from xrench.units import ureg

R_EARTH_M = 6371e3  # TR 38.811 uses 6371 km
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3

# --------------------------------------------------------------------------------------
# TR 38.811 tables
# --------------------------------------------------------------------------------------
_NTN_ELEV = np.arange(10.0, 100.0, 10.0)

# Table 6.6.1-1: LOS probability [%] at 10, 20, ..., 90 degrees
_NTN_LOS_PROB = {
    "dense_urban": np.array([28.2, 33.1, 39.8, 46.8, 53.7, 61.2, 73.8, 82.0, 98.1]),
    "urban": np.array([24.6, 38.6, 49.3, 61.3, 72.6, 80.5, 91.9, 96.8, 99.2]),
    "suburban": np.array([78.2, 86.9, 91.9, 92.9, 93.5, 94.0, 94.9, 95.2, 99.8]),
}
_NTN_LOS_PROB["rural"] = _NTN_LOS_PROB["suburban"]

# Tables 6.6.2-1/2/3. Columns: S-band [sigma_SF LOS, sigma_SF NLOS, CL], Ka-band [sigma_SF LOS, sigma_SF NLOS, CL]
_NTN_SFCL = {
    "dense_urban": np.array(
        [
            [3.5, 15.5, 34.3, 2.9, 17.1, 44.3],
            [3.4, 13.9, 30.9, 2.4, 17.1, 39.9],
            [2.9, 12.4, 29.0, 2.7, 15.6, 37.5],
            [3.0, 11.7, 27.7, 2.4, 14.6, 35.8],
            [3.1, 10.6, 26.8, 2.4, 14.2, 34.6],
            [2.7, 10.5, 26.2, 2.7, 12.6, 33.8],
            [2.5, 10.1, 25.8, 2.6, 12.1, 33.3],
            [2.3, 9.2, 25.5, 2.8, 12.3, 33.0],
            [1.2, 9.2, 25.5, 0.6, 12.3, 32.9],
        ],
    ),
    "urban": np.array(
        [
            [4.0, 6.0, 34.3, 4.0, 6.0, 44.3],
            [4.0, 6.0, 30.9, 4.0, 6.0, 39.9],
            [4.0, 6.0, 29.0, 4.0, 6.0, 37.5],
            [4.0, 6.0, 27.7, 4.0, 6.0, 35.8],
            [4.0, 6.0, 26.8, 4.0, 6.0, 34.6],
            [4.0, 6.0, 26.2, 4.0, 6.0, 33.8],
            [4.0, 6.0, 25.8, 4.0, 6.0, 33.3],
            [4.0, 6.0, 25.5, 4.0, 6.0, 33.0],
            [4.0, 6.0, 25.5, 4.0, 6.0, 32.9],
        ],
    ),
    "suburban": np.array(
        [
            [1.79, 8.93, 19.52, 1.9, 10.7, 29.5],
            [1.14, 9.08, 18.17, 1.6, 10.0, 24.6],
            [1.14, 8.78, 18.42, 1.9, 11.2, 21.9],
            [0.92, 10.25, 18.28, 2.3, 11.6, 20.0],
            [1.42, 10.56, 18.63, 2.7, 11.8, 18.7],
            [1.56, 10.74, 17.68, 3.1, 10.8, 17.8],
            [0.85, 10.17, 16.50, 3.0, 10.8, 17.2],
            [0.72, 11.52, 16.30, 3.6, 10.8, 16.9],
            [0.72, 11.52, 16.30, 0.4, 10.8, 16.8],
        ],
    ),
}
_NTN_SFCL["rural"] = _NTN_SFCL["suburban"]

# Table 6.6.6.2.1-1: tropospheric scintillation fade [dB], 99 %, 20 GHz, Toulouse
_NTN_TROPO_SCINT = np.array([1.08, 0.48, 0.30, 0.22, 0.17, 0.13, 0.12, 0.12, 0.12])

NTN_SCENARIOS = tuple(_NTN_LOS_PROB)


def _scenario(scenario: str) -> str:
    key = scenario.lower().replace("-", "_").replace(" ", "_")
    if key not in _NTN_LOS_PROB:
        raise ValueError(f"Unknown NTN scenario '{scenario}'. Choose from {NTN_SCENARIOS}.")
    return key


def _elev_lookup(table: np.ndarray, elevation_deg, interpolate: bool):
    """
    Look a 9-row elevation table up. TR 38.811 tabulates reference angles 10..90 degrees; by default the nearest
    reference angle is used (as the TR and ns-3 do). ``interpolate=True`` interpolates linearly instead. Elevations
    under 10 degrees use the 10 degree row; elevations at or under 0 return NaN.
    """
    elev = np.asarray(elevation_deg, dtype=float)
    e = np.clip(elev, _NTN_ELEV[0], _NTN_ELEV[-1])
    if interpolate:
        out = (
            np.interp(e, _NTN_ELEV, table)
            if table.ndim == 1
            else np.stack(
                [np.interp(e, _NTN_ELEV, table[:, j]) for j in range(table.shape[1])], axis=-1
            )
        )
    else:
        idx = np.clip(np.rint(np.nan_to_num(e, nan=10.0) / 10.0).astype(int) - 1, 0, 8)
        out = table[idx]
    below = ~(elev > 0)
    if np.any(below):
        out = np.array(out, dtype=float)
        out[below] = np.nan
    return out


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def slant_range(elevation_deg, altitude_m, r_earth_m: float = R_EARTH_M):
    """
    Slant range [m] to a platform at ``altitude_m`` seen at ``elevation_deg`` from the ground.

    TR 38.811 eq. (6.6-3): ``d = sqrt(R^2 sin^2(a) + h^2 + 2 h R) - R sin(a)``.
    """
    s = np.sin(np.deg2rad(np.asarray(elevation_deg, dtype=float)))
    h = np.asarray(altitude_m, dtype=float)
    return np.sqrt(r_earth_m**2 * s**2 + h**2 + 2 * h * r_earth_m) - r_earth_m * s


def free_space_loss_db(distance_m, f_ghz):
    """Free-space path loss [dB], TR 38.811 eq. (6.6-2): ``32.45 + 20 log10(f_GHz) + 20 log10(d_m)``."""
    return (
        32.45
        + 20 * np.log10(np.asarray(f_ghz, dtype=float))
        + 20 * np.log10(np.asarray(distance_m, dtype=float))
    )


# --------------------------------------------------------------------------------------
# TR 38.811 section 6.6
# --------------------------------------------------------------------------------------
def ntn_los_probability(elevation_deg, scenario: str = "suburban", interpolate: bool = False):
    """LOS probability (0-1) from TR 38.811 Table 6.6.1-1."""
    return _elev_lookup(_NTN_LOS_PROB[_scenario(scenario)], elevation_deg, interpolate) / 100.0


def ntn_shadow_clutter(
    f_ghz,
    elevation_deg,
    scenario: str = "suburban",
    interpolate: bool = False,
    band_split_ghz: float = 6.0,
):
    """
    Shadow-fading standard deviations and clutter loss from TR 38.811 Tables 6.6.2-1/2/3.

    Returns ``(sigma_los, sigma_nlos, clutter_loss)`` in dB, broadcast over ``f_ghz`` and ``elevation_deg``. The TR
    tabulates two bands only: the S-band columns are used below ``band_split_ghz`` and the Ka-band columns at or
    above it. The default of 6 GHz follows the TR's below / above 6 GHz split; ns-3 switches at 13 GHz. Results for
    frequencies far from 2 GHz and 20-30 GHz are an extrapolation of the TR either way.
    """
    f, e = np.broadcast_arrays(
        np.asarray(f_ghz, dtype=float), np.asarray(elevation_deg, dtype=float)
    )
    rows = _elev_lookup(_NTN_SFCL[_scenario(scenario)], e, interpolate)
    ka = f >= band_split_ghz
    sig_los = np.where(ka, rows[..., 3], rows[..., 0])
    sig_nlos = np.where(ka, rows[..., 4], rows[..., 1])
    clutter = np.where(ka, rows[..., 5], rows[..., 2])
    return sig_los, sig_nlos, clutter


def _mixture_quantile(q, p_los, sig_los, sig_nlos, clutter):
    """
    Quantile ``q`` (0-1) of ``X = SF_los`` with probability ``p_los`` else ``CL + SF_nlos``, by bisection.

    This is the loss not exceeded at a fraction ``q`` of locations when the LOS / NLOS state is itself random.
    """
    q, p_los, sig_los, sig_nlos, clutter = np.broadcast_arrays(q, p_los, sig_los, sig_nlos, clutter)
    sl = np.maximum(sig_los, 1e-9)
    sn = np.maximum(sig_nlos, 1e-9)
    lo = np.minimum(-8 * sl, clutter - 8 * sn)
    hi = np.maximum(8 * sl, clutter + 8 * sn)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = p_los * ndtr(mid / sl) + (1 - p_los) * ndtr((mid - clutter) / sn)
        below = cdf < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def ntn_excess_loss(
    f_ghz,
    elevation_deg,
    scenario: str = "suburban",
    los: bool | str = "mixture",
    location: float = 50.0,
    interpolate: bool = False,
    band_split_ghz: float = 6.0,
):
    """
    Shadow fading plus clutter loss of TR 38.811 section 6.6.2 as a deterministic location percentile [dB].

    ``PL_b = FSPL + SF + CL`` with ``SF ~ N(0, sigma_SF^2)`` and ``CL = 0`` in LOS. This returns ``SF + CL`` not
    exceeded at ``location`` % of locations.

    Parameters
    ----------
    los : True, False or "mixture"
        ``True`` / ``"los"``: LOS state. ``False`` / ``"nlos"``: NLOS state. ``"mixture"`` (default): the LOS state
        is random with the probability of Table 6.6.1-1, and the percentile is taken over the combined
        distribution -- the number to use for a coverage percentage when you do not know the state.
    location : float
        Percentage of locations (0-100) at which the loss is not exceeded. 50 gives the median.
    """
    f, e = np.broadcast_arrays(
        np.asarray(f_ghz, dtype=float), np.asarray(elevation_deg, dtype=float)
    )
    sig_los, sig_nlos, clutter = ntn_shadow_clutter(f, e, scenario, interpolate, band_split_ghz)
    q = np.clip(np.asarray(location, dtype=float) / 100.0, 1e-9, 1 - 1e-9)
    z = ndtri(q)
    if los is True or (isinstance(los, str) and los.lower() == "los"):
        return sig_los * z
    if los is False or (isinstance(los, str) and los.lower() == "nlos"):
        return clutter + sig_nlos * z
    if isinstance(los, str) and los.lower() == "mixture":
        p_los = ntn_los_probability(e, scenario, interpolate)
        return _mixture_quantile(q, p_los, sig_los, sig_nlos, clutter)
    raise ValueError("los must be True, False, 'los', 'nlos' or 'mixture'")


def ntn_tropo_scintillation(f_ghz, elevation_deg, interpolate: bool = False):
    """
    Tropospheric scintillation fade [dB] of TR 38.811 Table 6.6.6.2.1-1.

    The table is a single reference case (99 % of the time, 20 GHz, Toulouse) that the TR applies above 6 GHz
    without frequency scaling; below 6 GHz this returns 0. For a site- and frequency-specific value use
    ``itur_slant_path`` (ITU-R P.618 section 2.4.1).
    """
    f, e = np.broadcast_arrays(
        np.asarray(f_ghz, dtype=float), np.asarray(elevation_deg, dtype=float)
    )
    return np.where(f > 6.0, _elev_lookup(_NTN_TROPO_SCINT, e, interpolate), 0.0)


def ntn_iono_scintillation(f_ghz, pfluc_4ghz_db: float = 1.1):
    """
    Ionospheric scintillation loss [dB] below 6 GHz, TR 38.811 section 6.6.6.1 low-latitude shortcut:
    ``P_fluc(4 GHz) * (f / 4)^-1.5 / sqrt(2)``; 0 at and above 6 GHz.

    The TR applies this within +/-20 degrees of the magnetic equator only (nothing at mid latitudes, ITU-R P.531
    at high latitudes). ``pfluc_4ghz_db = 1.1`` is the value ns-3's NTN model uses for the TR's reference case;
    check it against the TR figure for your own S4 assumption before relying on it.
    """
    f = np.asarray(f_ghz, dtype=float)
    return np.where(f < 6.0, pfluc_4ghz_db * (f / 4.0) ** -1.5 / np.sqrt(2.0), 0.0)


# --------------------------------------------------------------------------------------
# ITU-R P.681 (land mobile satellite)
# --------------------------------------------------------------------------------------
_P681_T1_P = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 30.0])
_P681_T1 = {
    1.6: np.array([4.1, 2.0, 1.5, 1.4, 1.3, 1.2]),
    2.6: np.array([9.0, 5.2, 3.8, 3.2, 2.8, 2.5]),
}


def _p681_60(f, theta, p):
    """P.681 eqs. (1)-(5) for 7 <= theta <= 60 degrees (theta below 20 uses the 20 degree value)."""
    th = np.clip(theta, 20.0, 60.0)
    m = 3.44 + 0.0975 * th - 0.002 * th**2
    n = -0.443 * th + 34.76
    a_l = -m * np.log(np.minimum(p, 20.0)) + n  # 1.5 GHz, 1 % <= p <= 20 %
    a_20 = a_l * np.exp(1.5 * (1 / np.sqrt(1.5) - 1 / np.sqrt(f)))  # frequency scaling
    return np.where(p > 20.0, a_20 * np.log(80.0 / p) / np.log(4.0), a_20)


def p681_roadside_tree_shadowing(f_ghz, elevation_deg, p):
    """
    Fade [dB] exceeded for ``p`` % of the distance travelled, from shadowing by roadside trees.

    ITU-R P.681 section 4.1.1 (extended empirical roadside shadowing model). Valid for 0.8-20 GHz, elevation
    7-60 degrees and 1 <= p <= 80 %, for roads where 55-75 % of the path is optically shadowed. Section 4.1.1.1
    extends the model above 60 degrees only for 1.6-2.6 GHz and p <= 30 % (linear interpolation to the Table 1
    values at 80 degrees, then to zero at 90 degrees). Anything outside those ranges returns NaN rather than an
    extrapolation.

    Note the convention: this is a fade *exceeded* for p % (small p = deep, rare fades), the opposite sense to the
    "not exceeded" percentages used by the P.2108 and TR 38.811 functions.
    """
    f, th, p = np.broadcast_arrays(
        np.asarray(f_ghz, dtype=float),
        np.asarray(elevation_deg, dtype=float),
        np.asarray(p, dtype=float),
    )
    out = np.full(f.shape, np.nan)
    ok = (f >= 0.8) & (f <= 20.0) & (p >= 1.0) & (p <= 80.0) & (th >= 7.0) & (th <= 90.0)
    low = ok & (th <= 60.0)
    out[low] = _p681_60(f[low], th[low], p[low])

    high = ok & (th > 60.0) & (f >= 1.6) & (f <= 2.6) & (p <= 30.0)
    if np.any(high):
        fh, thh, ph = f[high], th[high], p[high]
        a60 = _p681_60(fh, np.full_like(thh, 60.0), ph)
        lp = np.log(ph)
        a80_16 = np.interp(lp, np.log(_P681_T1_P), _P681_T1[1.6])
        a80_26 = np.interp(lp, np.log(_P681_T1_P), _P681_T1[2.6])
        a80 = a80_16 + (a80_26 - a80_16) * (fh - 1.6) / (2.6 - 1.6)
        out[high] = np.where(
            thh <= 80.0,
            a60 + (a80 - a60) * (thh - 60.0) / 20.0,
            a80 * (90.0 - thh) / 10.0,
        )
    return out


def p681_building_blockage(
    f_ghz,
    elevation_deg,
    azimuth_deg=90.0,
    h_building_m: float = 15.0,
    h_mobile_m: float = 1.5,
    d_front_m: float = 17.5,
    fresnel_clearance: float = 0.7,
):
    """
    Percentage probability [%] of blockage by roadside buildings, ITU-R P.681 section 4.2 eq. (8).

    ``azimuth_deg`` is the direction to the satellite relative to the street (90 = broadside to the buildings),
    ``h_building_m`` the most common building height, ``d_front_m`` the distance from the mobile to the building
    front and ``fresnel_clearance`` the fraction of the first Fresnel zone that must be clear. The defaults are
    the Recommendation's worked example. Building heights are Rayleigh distributed about ``h_building_m``, so the
    result is the probability that a building is taller than the ray height less the required clearance.
    """
    f = np.asarray(f_ghz, dtype=float)
    th = np.deg2rad(np.clip(np.asarray(elevation_deg, dtype=float), 1e-3, 90.0))
    ph = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    lam = 0.299792458 / f
    sin_ph = np.maximum(np.abs(np.sin(ph)), 1e-9)
    # h1: height of the ray above ground where it crosses the building line; dr: slant distance to that point
    h1 = h_mobile_m + d_front_m * np.tan(th) / sin_ph
    dr = d_front_m / (sin_ph * np.maximum(np.cos(th), 1e-9))
    h2 = fresnel_clearance * np.sqrt(lam * dr)
    clear = np.maximum(h1 - h2, 0.0)
    return 100.0 * np.exp(-(clear**2) / (2 * h_building_m**2))


# --------------------------------------------------------------------------------------
# TR 38.901 (terrestrial, 0.5-100 GHz)
# --------------------------------------------------------------------------------------
TR38901_SCENARIOS = ("rma", "uma", "umi", "inh")
_TR38901_SIGMA = {
    ("rma", True): 4.0,  # 6 dB beyond the breakpoint
    ("rma", False): 8.0,
    ("uma", True): 4.0,
    ("uma", False): 6.0,
    ("umi", True): 4.0,
    ("umi", False): 7.82,
    ("inh", True): 3.0,
    ("inh", False): 8.03,
}


def tr38901_los_probability(scenario: str, d2d_m, h_ut_m: float = 1.5):
    """LOS probability (0-1), TR 38.901 Table 7.4.2-1 (outdoor UT; InH is the mixed-office case)."""
    sc = scenario.lower()
    d = np.maximum(np.asarray(d2d_m, dtype=float), 1e-9)
    if sc == "rma":
        return np.where(d <= 10, 1.0, np.exp(-(d - 10) / 1000))
    if sc == "umi":
        return np.where(d <= 18, 1.0, 18 / d + np.exp(-d / 36) * (1 - 18 / d))
    if sc == "uma":
        h = np.clip(np.asarray(h_ut_m, dtype=float), 1.5, 23.0)
        c = np.where(h <= 13, 0.0, ((h - 13) / 10) ** 1.5)
        pr = (18 / d + np.exp(-d / 63) * (1 - 18 / d)) * (
            1 + c * 1.25 * (d / 100) ** 3 * np.exp(-d / 150)
        )
        return np.where(d <= 18, 1.0, pr)
    if sc == "inh":
        return np.where(
            d <= 1.2,
            1.0,
            np.where(d < 6.5, np.exp(-(d - 1.2) / 4.7), np.exp(-(d - 6.5) / 32.6) * 0.32),
        )
    raise ValueError(f"Unknown TR 38.901 scenario '{scenario}'. Choose from {TR38901_SCENARIOS}.")


def tr38901_path_loss(
    scenario: str,
    d2d_m,
    f_ghz,
    h_bs_m: float | None = None,
    h_ut_m: float = 1.5,
    los: bool = True,
    h_building_m: float = 5.0,
    street_width_m: float = 20.0,
    h_e_m: float = 1.0,
):
    """
    Median path loss [dB] of TR 38.901 Table 7.4.1-1 (free-space loss is already inside these formulas).

    ``scenario`` is ``"rma"``, ``"uma"``, ``"umi"`` (street canyon) or ``"inh"`` (indoor office). ``d2d_m`` is the
    horizontal distance; default base-station heights are 35 / 25 / 10 / 3 m. Applicability: 0.5-100 GHz (RMa to
    30 GHz), d2D from 10 m to 5 km (RMa 10 km, InH 1-150 m). Use ``tr38901_shadow_sigma`` for the shadow-fading
    standard deviation.
    """
    sc = scenario.lower()
    if sc not in TR38901_SCENARIOS:
        raise ValueError(
            f"Unknown TR 38.901 scenario '{scenario}'. Choose from {TR38901_SCENARIOS}."
        )
    if h_bs_m is None:
        h_bs_m = {"rma": 35.0, "uma": 25.0, "umi": 10.0, "inh": 3.0}[sc]
    d2d, fc = np.broadcast_arrays(np.asarray(d2d_m, dtype=float), np.asarray(f_ghz, dtype=float))
    d3d = np.sqrt(d2d**2 + (h_bs_m - h_ut_m) ** 2)
    c = 299792458.0
    lg = np.log10

    if sc == "rma":
        h = h_building_m
        d_bp = 2 * np.pi * h_bs_m * h_ut_m * fc * 1e9 / c

        def pl1(d):
            return (
                20 * lg(40 * np.pi * d * fc / 3)
                + min(0.03 * h**1.72, 10) * lg(d)
                - min(0.044 * h**1.72, 14.77)
                + 0.002 * lg(h) * d
            )

        pl_los = np.where(d2d <= d_bp, pl1(d3d), pl1(d_bp) + 40 * lg(d3d / d_bp))
        if los:
            return pl_los
        pl_n = (
            161.04
            - 7.1 * lg(street_width_m)
            + 7.5 * lg(h)
            - (24.37 - 3.7 * (h / h_bs_m) ** 2) * lg(h_bs_m)
            + (43.42 - 3.1 * lg(h_bs_m)) * (lg(d3d) - 3)
            + 20 * lg(fc)
            - (3.2 * lg(11.75 * h_ut_m) ** 2 - 4.97)
        )
        return np.maximum(pl_los, pl_n)

    if sc == "inh":
        pl_los = 32.4 + 17.3 * lg(d3d) + 20 * lg(fc)
        if los:
            return pl_los
        return np.maximum(pl_los, 38.3 * lg(d3d) + 17.30 + 24.9 * lg(fc))

    # UMa / UMi share the breakpoint construction (effective heights reduced by the environment height h_E)
    d_bp = 4 * (h_bs_m - h_e_m) * (h_ut_m - h_e_m) * fc * 1e9 / c
    dh2 = (h_bs_m - h_ut_m) ** 2
    if sc == "uma":
        pl_los = np.where(
            d2d <= d_bp,
            28.0 + 22 * lg(d3d) + 20 * lg(fc),
            28.0 + 40 * lg(d3d) + 20 * lg(fc) - 9 * lg(d_bp**2 + dh2),
        )
        if los:
            return pl_los
        return np.maximum(pl_los, 13.54 + 39.08 * lg(d3d) + 20 * lg(fc) - 0.6 * (h_ut_m - 1.5))
    pl_los = np.where(
        d2d <= d_bp,
        32.4 + 21 * lg(d3d) + 20 * lg(fc),
        32.4 + 40 * lg(d3d) + 20 * lg(fc) - 9.5 * lg(d_bp**2 + dh2),
    )
    if los:
        return pl_los
    return np.maximum(pl_los, 35.3 * lg(d3d) + 22.4 + 21.3 * lg(fc) - 0.3 * (h_ut_m - 1.5))


def tr38901_shadow_sigma(scenario: str, los: bool = True) -> float:
    """Shadow-fading standard deviation [dB], TR 38.901 Table 7.4.1-1 (RMa LOS: 4 dB, 6 dB past the breakpoint)."""
    return _TR38901_SIGMA[(scenario.lower(), bool(los))]


# --------------------------------------------------------------------------------------
# ITU-R slant-path atmosphere: pluggable backend (default: the optional ``itur`` package)
# --------------------------------------------------------------------------------------
@runtime_checkable
class AtmosphereBackend(Protocol):
    """
    What the propagators need from an atmospheric-attenuation model. Implement these two methods and pass the
    object to :func:`set_atmosphere_backend` to replace ITU-Rpy (for example with native P.676 / P.618 code, or with
    values from a site survey).

    ``gas_standard(f_ghz, elevation_deg)``
        Gaseous attenuation [dB] through the standard atmosphere (1013.25 hPa, 288.15 K, 7.5 g/m^3) for arrays that
        broadcast together; NaN where elevation <= 0.
    ``slant_path(lat_deg, lon_deg, f_ghz, elevation_deg, p, antenna_diameter_m)``
        Dict of arrays [dB] with keys ``gas``, ``cloud``, ``rain``, ``scintillation`` and ``total``, exceeded for
        ``p`` % of an average year; ``f_ghz`` and ``p`` are scalars; NaN where elevation <= 0.
    """

    def gas_standard(self, f_ghz, elevation_deg): ...

    def slant_path(
        self, lat_deg, lon_deg, f_ghz, elevation_deg, p: float, antenna_diameter_m: float
    ) -> dict: ...


def _import_itur():
    try:
        import itur
    except ImportError as e:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "Atmospheric terms use the ITU-Rpy package (P.676 gases, P.840 cloud, P.618 / P.838 rain, "
            "P.618 scintillation). Install it with `pip install itur`.",
        ) from e
    return itur


def itur_gas_standard(f_ghz, elevation_deg):
    """
    Gaseous attenuation [dB] on a slant path through the standard atmosphere (ITU-R P.676, 1013.25 hPa, 288.15 K,
    7.5 g/m^3), which is what TR 38.811 section 6.6.4 prescribes: zenith attenuation divided by sin(elevation).
    """
    itur = _import_itur()
    f, e = np.broadcast_arrays(
        np.asarray(f_ghz, dtype=float), np.asarray(elevation_deg, dtype=float)
    )
    out = np.full(f.shape, np.nan)
    fu, inv = np.unique(f, return_inverse=True)
    zen = np.array(
        [
            float(
                np.asarray(
                    itur.gaseous_attenuation_slant_path(fi, 90.0, 7.5, 1013.25, 288.15).value
                )
            )
            for fi in fu
        ],
    )
    ok = e > 0
    out[ok] = zen[inv.reshape(f.shape)][ok] / np.sin(np.deg2rad(e[ok]))
    return out


def itur_slant_path(
    lat_deg, lon_deg, f_ghz, elevation_deg, p: float = 1.0, antenna_diameter_m: float = 1.0
):
    """
    Site-specific slant-path attenuation exceeded for ``p`` % of an average year, from the ITU-R digital maps.

    Returns a dict of arrays [dB]: ``gas`` (P.676), ``cloud`` (P.840), ``rain`` (P.618 / P.838 / P.837 / P.839),
    ``scintillation`` (P.618) and ``total`` (combined per P.618 section 2.5). ``lat_deg``, ``lon_deg`` and
    ``elevation_deg`` broadcast together; ``f_ghz`` is a scalar. Points with elevation <= 0 return NaN. The ITU-R
    method is intended for elevations above about 5 degrees and frequencies to 55 GHz.
    """
    itur = _import_itur()
    lat, lon, el = np.broadcast_arrays(
        np.asarray(lat_deg, dtype=float),
        np.asarray(lon_deg, dtype=float),
        np.asarray(elevation_deg, dtype=float),
    )
    shape = lat.shape
    lat, lon, el = lat.ravel(), lon.ravel(), el.ravel()
    keys = ("gas", "cloud", "rain", "scintillation", "total")
    out = {k: np.full(lat.shape, np.nan) for k in keys}
    ok = el > 0
    if np.any(ok):
        res = itur.atmospheric_attenuation_slant_path(
            lat[ok],
            lon[ok],
            float(f_ghz),
            el[ok],
            float(p),
            float(antenna_diameter_m),
            return_contributions=True,
        )
        for k, v in zip(keys, res, strict=True):
            out[k][ok] = np.atleast_1d(np.asarray(v.value, dtype=float))
    return {k: v.reshape(shape) for k, v in out.items()}


class ItuRpyBackend:
    """Default :class:`AtmosphereBackend`, built on the ``itur`` package (``pip install xant[itur]``)."""

    gas_standard = staticmethod(itur_gas_standard)
    slant_path = staticmethod(itur_slant_path)


_ATMOSPHERE_BACKEND: AtmosphereBackend = ItuRpyBackend()


def set_atmosphere_backend(backend: AtmosphereBackend | None) -> None:
    """Use ``backend`` for every atmospheric term; ``None`` restores the default ITU-Rpy backend."""
    global _ATMOSPHERE_BACKEND
    if backend is None:
        backend = ItuRpyBackend()
    if not isinstance(backend, AtmosphereBackend):
        raise TypeError("backend must provide gas_standard(f_ghz, elevation_deg) and slant_path(...)")
    _ATMOSPHERE_BACKEND = backend


def get_atmosphere_backend() -> AtmosphereBackend:
    """The backend currently used for atmospheric terms."""
    return _ATMOSPHERE_BACKEND


def atmosphere_gas_standard(f_ghz, elevation_deg):
    """Standard-atmosphere gaseous attenuation [dB] from the active backend (see :class:`AtmosphereBackend`)."""
    return _ATMOSPHERE_BACKEND.gas_standard(f_ghz, elevation_deg)


def atmosphere_slant_path(
    lat_deg, lon_deg, f_ghz, elevation_deg, p: float = 1.0, antenna_diameter_m: float = 1.0
):
    """Site-specific slant-path attenuation terms [dB] from the active backend (see :class:`AtmosphereBackend`)."""
    return _ATMOSPHERE_BACKEND.slant_path(
        lat_deg, lon_deg, f_ghz, elevation_deg, float(p), float(antenna_diameter_m)
    )


# --------------------------------------------------------------------------------------
# Propagators (HCS in, DataArray of loss in dB out)
# --------------------------------------------------------------------------------------
def _mag(da, unit: str) -> xr.DataArray:
    """DataArray of plain floats in ``unit``."""
    if isinstance(da, Quantity):
        return xr.DataArray(np.asarray(da.to(unit).magnitude, dtype=float))
    if isinstance(da, xr.DataArray):
        out = da.copy()
        if isinstance(out.data, Quantity):
            out.data = np.asarray(out.data.to(unit).magnitude, dtype=float)
        return out
    return xr.DataArray(np.asarray(da, dtype=float))


def _strip_coords(da: xr.DataArray) -> xr.DataArray:
    """Drop non-index coordinates so geometry helpers don't drag conflicting scalar coords along."""
    return da.drop_vars([c for c in da.coords if c not in da.dims])


def _ecef(lat_rad, lon_rad, h_m):
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * np.sin(lat_rad) ** 2)
    x = (n + h_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + h_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - _WGS84_E2) + h_m) * np.sin(lat_rad)
    return x, y, z


def link_geometry(txcs, rxcs) -> dict:
    """
    Elevation angle of the higher terminal seen from the lower one, from the two coordinate systems.

    Returns a dict of plain-float DataArrays: ``elevation`` [deg] above the local (WGS-84) horizontal at the lower
    terminal, ``lat`` / ``lon`` [deg] and ``h`` [m] of the lower ("ground") terminal and ``h_platform`` [m].
    Which end is the ground terminal is decided from the mean altitudes.
    """
    tl = [
        _strip_coords(_mag(v, u)) for v, u in zip(txcs.llh, ("radian", "radian", "m"), strict=True)
    ]
    rl = [
        _strip_coords(_mag(v, u)) for v, u in zip(rxcs.llh, ("radian", "radian", "m"), strict=True)
    ]
    g, s = (tl, rl) if float(tl[2].mean()) <= float(rl[2].mean()) else (rl, tl)
    gx, gy, gz = _ecef(*g)
    sx, sy, sz = _ecef(*s)
    dx, dy, dz = sx - gx, sy - gy, sz - gz
    rng = np.sqrt(dx**2 + dy**2 + dz**2)
    # local "up" (ellipsoid normal) at the ground terminal
    ux, uy, uz = np.cos(g[0]) * np.cos(g[1]), np.cos(g[0]) * np.sin(g[1]), np.sin(g[0])
    sin_el = (dx * ux + dy * uy + dz * uz) / rng
    elevation = np.rad2deg(np.arcsin(sin_el.clip(-1, 1)))
    ones = xr.ones_like(elevation)
    return dict(
        elevation=elevation,
        lat=np.rad2deg(g[0]) * ones,
        lon=np.rad2deg(g[1]) * ones,
        h=g[2] * ones,
        h_platform=s[2] * ones,
    )


def _fspl_db(txcs, rxcs, frequency) -> xr.DataArray:
    """Free-space loss from ``propagators.fspl`` as a plain-float DataArray in dB (keeps the distance coord)."""
    from . import propagators

    fs = propagators.fspl(txcs=txcs, rxcs=rxcs, frequency=frequency)
    out = fs.copy()
    out.data = np.asarray(fs.data.to("dB").magnitude, dtype=float)
    return out


def _finish(fs_db: xr.DataArray, terms: dict, additional_loss, coords: dict) -> xr.DataArray:
    """Sum free-space and excess terms, apply ``additional_loss`` and attach the terms as coordinates."""
    total = fs_db
    for v in terms.values():
        total = total + v
    if additional_loss is not None:
        add = additional_loss
        if isinstance(add, xr.DataArray):
            add = add.copy()
            if isinstance(add.data, Quantity):
                add.data = np.asarray(add.data.to("dB").magnitude, dtype=float)
        elif isinstance(add, Quantity):
            add = (
                float(add.to("dB").magnitude)
                if add.ndim == 0
                else np.asarray(add.to("dB").magnitude)
            )
        total = total + add
    if "distance" in fs_db.coords and "distance" not in total.coords:
        total = total.assign_coords(distance=fs_db.coords["distance"])
    extra = {"fspl": fs_db.drop_vars("distance", errors="ignore"), **terms, **coords}
    for k, v in extra.items():
        v = v if isinstance(v, xr.DataArray) else xr.DataArray(v)
        if set(v.dims) <= set(total.dims):
            total = total.assign_coords({k: v.drop_vars([c for c in v.coords if c not in v.dims])})
    total.data = np.asarray(total.data, dtype=float) * ureg.dB
    return total


def _f_ghz(frequency) -> xr.DataArray:
    return _mag(frequency, "GHz")


def ntn_tr38811(
    txcs=None,
    rxcs=None,
    frequency=None,
    additional_loss=None,
    scenario: str = "suburban",
    los: bool | str = "mixture",
    location: float = 50.0,
    atmosphere: bool = False,
    scintillation: bool = False,
    ionospheric_scintillation: bool = False,
    building_entry_loss: Quantity | float = 0.0,
    interpolate: bool = False,
    band_split_ghz: float = 6.0,
    **kwargs,
) -> xr.DataArray:
    """
    Satellite / HAPS to ground path loss per 3GPP TR 38.811 section 6.6: ``PL = PL_b + PL_g + PL_s + PL_e``.

    * ``PL_b`` -- free-space loss + shadow fading + clutter loss (Tables 6.6.2-x) for ``scenario``
      (``"dense_urban"``, ``"urban"``, ``"suburban"``, ``"rural"``), evaluated at ``location`` % of locations for
      the LOS state ``los`` (see ``ntn_excess_loss``).
    * ``PL_g`` -- gases, when ``atmosphere=True`` (ITU-R P.676 standard atmosphere / sin(elevation); needs
      ``itur``). The TR only requires it above 10 GHz or under 10 degrees elevation.
    * ``PL_s`` -- ``scintillation=True`` adds the tropospheric table value above 6 GHz;
      ``ionospheric_scintillation=True`` adds the low-latitude ionospheric term below 6 GHz.
    * ``PL_e`` -- ``building_entry_loss`` in dB (see ITU-R P.2109), 0 for outdoor terminals.

    The elevation angle is taken at whichever terminal is lower. Points below the horizon return NaN, and
    elevations under 10 degrees use the 10 degree table row (the TR tabulates 10-90 degrees only).
    """
    geo = link_geometry(txcs, rxcs)
    fs = _fspl_db(txcs, rxcs, frequency)
    f = _f_ghz(frequency)
    el = geo["elevation"]

    def _apply(fn, *a, **kw):
        return xr.apply_ufunc(lambda ff, ee: fn(ff, ee, *a, **kw), f, el)

    terms = {
        "shadow_clutter": _apply(
            ntn_excess_loss,
            scenario,
            los,
            location,
            interpolate,
            band_split_ghz,
        ),
    }
    if atmosphere:
        terms["gas"] = _apply(atmosphere_gas_standard)
    if scintillation:
        terms["tropo_scintillation"] = _apply(ntn_tropo_scintillation, interpolate)
    if ionospheric_scintillation:
        terms["iono_scintillation"] = xr.apply_ufunc(ntn_iono_scintillation, f) + 0 * el
    bel = building_entry_loss
    bel = float(bel.to("dB").magnitude) if isinstance(bel, Quantity) else float(bel)
    if bel:
        terms["building_entry"] = xr.zeros_like(el) + bel
    coords = dict(
        elevation=el,
        los_probability=xr.apply_ufunc(lambda e: ntn_los_probability(e, scenario, interpolate), el),
    )
    return _finish(fs, terms, additional_loss, coords)


def lms_p681(
    txcs=None,
    rxcs=None,
    frequency=None,
    additional_loss=None,
    percent_exceeded: float = 10.0,
    **kwargs,
) -> xr.DataArray:
    """
    Land-mobile-satellite path loss: free space plus the ITU-R P.681 section 4.1.1 roadside tree-shadowing fade
    exceeded for ``percent_exceeded`` % of the distance travelled (0.8-20 GHz, elevation 7-60 degrees; see
    ``p681_roadside_tree_shadowing`` for the limits). Outside the model's range the result is NaN.
    """
    geo = link_geometry(txcs, rxcs)
    fs = _fspl_db(txcs, rxcs, frequency)
    fade = xr.apply_ufunc(
        lambda ff, ee: p681_roadside_tree_shadowing(ff, ee, percent_exceeded),
        _f_ghz(frequency),
        geo["elevation"],
    )
    return _finish(fs, {"tree_shadowing": fade}, additional_loss, dict(elevation=geo["elevation"]))


def earth_space_itur(
    txcs=None,
    rxcs=None,
    frequency=None,
    additional_loss=None,
    p: float = 1.0,
    antenna_diameter: Quantity | float = 1.0,
    clutter_location: float | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Fixed earth-space path loss: free space plus the site-specific ITU-R slant-path attenuation (gases, cloud,
    rain, scintillation) exceeded for ``p`` % of an average year at the ground terminal's latitude / longitude.

    ``antenna_diameter`` (metres or a length Quantity) is the earth-station dish used for scintillation averaging.
    ``clutter_location`` (percent of locations) additionally applies the ITU-R P.2108 section 3.3 earth-space
    clutter loss for a terminal inside urban / suburban clutter; that model is only defined for 10-100 GHz and
    returns NaN outside it. Uses the active atmosphere backend (default: the ``itur`` package, see
    :func:`set_atmosphere_backend`).
    """
    geo = link_geometry(txcs, rxcs)
    fs = _fspl_db(txcs, rxcs, frequency)
    f = _f_ghz(frequency)
    el = geo["elevation"]
    dia = antenna_diameter
    dia = float(dia.to("m").magnitude) if isinstance(dia, Quantity) else float(dia)

    names = ("gas", "cloud", "rain", "scintillation")
    per_f = {k: [] for k in (*names, "total")}
    fvals = np.atleast_1d(f.values)
    for fi in fvals:
        res = atmosphere_slant_path(geo["lat"].values, geo["lon"].values, fi, el.values, p, dia)
        for k in per_f:
            per_f[k].append(xr.DataArray(res[k], dims=el.dims, coords=el.coords))

    def _stack(lst):
        if f.ndim == 0:
            return lst[0]
        return xr.concat(lst, dim=f.dims[0]).assign_coords({f.dims[0]: f.coords[f.dims[0]]})

    comps = {k: _stack(v) for k, v in per_f.items()}
    terms = {"atmosphere": comps["total"]}
    if clutter_location is not None:
        from .itm.p2108 import p2108_3

        def _clutter(ff, ee):
            ff, ee = np.broadcast_arrays(np.asarray(ff, dtype=float), np.asarray(ee, dtype=float))
            out = np.full(ff.shape, np.nan)
            ok = (ff >= 10.0) & (ff <= 100.0) & (ee > 0)
            if np.any(ok):
                out[ok] = p2108_3(
                    ff[ok] * ureg.GHz, ee[ok] * ureg.degree, clutter_location / 100.0
                ).magnitude
            return out

        if np.any((fvals < 10.0) | (fvals > 100.0)):
            logger.warning(
                "ITU-R P.2108 section 3.3 clutter is only defined for 10-100 GHz; NaN outside it."
            )
        terms["clutter"] = xr.apply_ufunc(_clutter, f, el)
    coords = dict(elevation=el, **{k: comps[k] for k in names})
    return _finish(fs, terms, additional_loss, coords)


def tr38901(
    txcs=None,
    rxcs=None,
    frequency=None,
    additional_loss=None,
    scenario: str = "uma",
    los: bool | str = True,
    location: float = 50.0,
    h_bs: Quantity | float | None = None,
    h_ut: Quantity | float = 1.5,
    **kwargs,
) -> xr.DataArray:
    """
    Terrestrial path loss per 3GPP TR 38.901 Table 7.4.1-1 (``"rma"``, ``"uma"``, ``"umi"``, ``"inh"``).

    The 3D distance comes from the two coordinate systems; ``h_bs`` and ``h_ut`` are antenna heights above ground
    (metres or length Quantities; ``h_bs`` defaults to the scenario's reference height) and set the 2D distance.
    ``los`` is ``True``, ``False`` or ``"mean"`` (LOS-probability-weighted mean of the LOS and NLOS medians in dB).
    ``location`` adds the shadow-fading margin not exceeded at that percentage of locations (``"mean"`` uses the
    NLOS sigma). Free-space loss is already part of these formulas, so the ``fspl`` coordinate is informative only.
    """
    sc = scenario.lower()
    h_ut_m = float(h_ut.to("m").magnitude) if isinstance(h_ut, Quantity) else float(h_ut)
    if h_bs is None:
        h_bs_m = {"rma": 35.0, "uma": 25.0, "umi": 10.0, "inh": 3.0}[sc]
    else:
        h_bs_m = float(h_bs.to("m").magnitude) if isinstance(h_bs, Quantity) else float(h_bs)
    fs = _fspl_db(txcs, rxcs, frequency)
    f = _f_ghz(frequency)
    d3d = _strip_coords(_mag(fs.coords["distance"], "m"))
    d2d = np.sqrt((d3d**2 - (h_bs_m - h_ut_m) ** 2).clip(min=1.0))
    z = float(ndtri(np.clip(location / 100.0, 1e-9, 1 - 1e-9)))

    def _pl(is_los):
        return xr.apply_ufunc(
            lambda dd, ff: tr38901_path_loss(sc, dd, ff, h_bs_m, h_ut_m, los=is_los),
            d2d,
            f,
        )

    p_los = xr.apply_ufunc(lambda dd: tr38901_los_probability(sc, dd, h_ut_m), d2d)
    if isinstance(los, str) and los.lower() == "mean":
        pl = p_los * _pl(True) + (1 - p_los) * _pl(False)
        sigma = tr38901_shadow_sigma(sc, False)
    elif los is True or (isinstance(los, str) and los.lower() == "los"):
        pl, sigma = _pl(True), tr38901_shadow_sigma(sc, True)
    elif los is False or (isinstance(los, str) and los.lower() == "nlos"):
        pl, sigma = _pl(False), tr38901_shadow_sigma(sc, False)
    else:
        raise ValueError("los must be True, False, 'los', 'nlos' or 'mean'")
    pl = pl + sigma * z
    # Express as free space + excess so the bookkeeping matches the other propagators
    return _finish(fs, {"excess": pl - fs}, additional_loss, dict(los_probability=p_los))
