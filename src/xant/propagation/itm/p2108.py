"""
Implementation of ITU-R P.2108 Prediction of clutter loss


References:
https://github.com/NTIA/p2108
https://www.itu.int/rec/R-REC-P.2108
"""

import numpy as np
from scipy.special import ndtri
from xrench.units import ureg


def _qinv(p):
    """Inverse complementary normal distribution function, Q^-1(p), for 0 < p < 1."""
    return -ndtri(p)


@ureg.wraps(ureg.dB, (ureg.GHz, ureg.km, None))
def p2108_2(f_ghz: float, d_km: float, p: float) -> float:
    """
    ITU-R P.2108-1 Section 3.2

    Statistical clutter loss model for terrestrial paths (0.5-67 GHz, d >= 0.25 km). Returns the loss *not
    exceeded* for a fraction ``p`` of locations (0 < p < 1), so the loss increases with ``p``. Inputs may be arrays.
    """
    f_ghz = np.asarray(f_ghz, dtype=float)
    p = np.asarray(p, dtype=float)
    # Maximum loss is for 2km (eq. 6)
    d_km = np.minimum(np.asarray(d_km, dtype=float), 2.0)
    # eq. (4a)
    Ll = -2 * np.log10(10 ** (-5 * np.log10(f_ghz) - 12.5) + 10 ** (-16.5))
    # eq. (4b)
    sigma_l = 4
    # eq. (5a)
    Ls = 32.98 + 23.9 * np.log10(d_km) + 3 * np.log10(f_ghz)
    # eq. (5b)
    sigma_s = 6
    # eq. (3b)
    sigma_cb = np.sqrt(
        (sigma_l**2 * 10 ** (-0.2 * Ll) + sigma_s**2 * 10 ** (-0.2 * Ls))
        / (10 ** (-0.2 * Ll) + 10 ** (-0.2 * Ls)),
    )
    # eq. (3a): the Q^-1 term is the inverse *complementary* normal, which is negative above the median
    return -5 * np.log10(10 ** (-0.2 * Ll) + 10 ** (-0.2 * Ls)) - sigma_cb * _qinv(p)


@ureg.wraps(ureg.dB, (ureg.GHz, ureg.degree, None))
def p2108_3(f_ghz: float, elevation_deg: float, p: float) -> float:
    """
    ITU-R P.2108-1 Section 3.3, eq. (7)

    Earth-space and aeronautical statistical clutter loss for a terminal within urban / suburban clutter
    (10-100 GHz, elevation 0-90 degrees). Returns the loss *not exceeded* for a fraction ``p`` of locations
    (0 < p < 1). Inputs may be arrays.
    """
    f_ghz = np.asarray(f_ghz, dtype=float)
    theta = np.clip(np.asarray(elevation_deg, dtype=float), 0.0, 90.0)
    p = np.asarray(p, dtype=float)
    K1 = 93 * f_ghz**0.175
    A1 = 0.05
    angle = A1 * (1 - theta / 90.0) + np.pi * theta / 180.0
    base = -K1 * np.log(1 - p) / np.tan(angle)
    return base ** (0.5 * (90.0 - theta) / 90.0) - 1 - 0.6 * _qinv(p)
