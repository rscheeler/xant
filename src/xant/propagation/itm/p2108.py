"""
Implementation of ITU-R P.2108 Prediction of clutter loss


References:
https://github.com/NTIA/p2108
https://www.itu.int/rec/R-REC-P.2108
"""

import numpy as np
from scipy.stats import Normal, norm

from ... import ureg


@ureg.wraps(ureg.dB, (ureg.GHz, ureg.km, None))
def p2108_2(f_ghz: float, d_km: float, p: float) -> float:
    """
    ITU-R P.2108-1 Section 3.2

    Statistical clutter loss model for terrestrial paths
    """
    # Maximum loss is for 2km (eq. 6)
    d_km = min(d_km, 2)
    Qinv = norm.ppf(p, loc=0, scale=1)
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
    # eq. (3a)
    return -5 * np.log10(10 ** (-0.2 * Ll) + 10 ** (-0.2 * Ls)) - sigma_cb * Qinv
