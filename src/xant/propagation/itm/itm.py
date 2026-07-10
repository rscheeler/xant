"""
This module is a python implementation of the NTIA Irregular Terrain Model (ITM) (Longley-Rice)
Note this implementation follows version 1.3 which is an update to the 1.2.2 FORTRAN source

Model valid from 20 MHz- 20GHz

Module focused on point-to-point propagation.

The original ITM algorithm is a work of the US federal government (NTIA/ITS)
and is in the public domain under 17 U.S.C. § 105.
This Python implementation is original work by Rob Scheeler and is
licensed under the MIT License.

References:
    https://its.ntia.gov/research-topics/radio-propagation-software/itm/itm.aspx
    https://github.com/NTIA/itm
"""

import warnings

import numpy as np
import xarray as xr
from pint import Quantity
from xrench.units import ureg

from .utils import take_magnitudes

## Constants
GAMMA_A = 157e-9  # gamma_a is the curvature of the actual earth, approximately 1 / 6370 km
a_0__meter = 6370e3  # Earth radius in meters
a_9000__meter = 9000e3

# Electrical Ground Constants
# NTIA Technical Report ERL 79-ITS 67
# Prediction of Tropospheric Radio Transmission Loss Over Irregular Terrain: A Computer Method - 1968
# https://www.its.bldrdoc.gov/publications/details.aspx?pub=2784
# NTIA REPORT 82-100
# A Guide to the Use of the  ITS Irregular Terrain Model in the Area Prediction Mode
# https://www.ntia.doc.gov/files/ntia/publications/ntia_82-100_20121129145031_555510.pdf
_GND_EPS = dict(poor=4, average=15, good=25, fresh_water=81, sea_water=81)
_GND_SGM = dict(poor=0.001, average=0.005, good=0.02, fresh_water=0.01, sea_water=5.0)

DEFAULT_PARAMS = dict()
ITM_POLARIZATION = dict(horizontal=0, vertical=1)

modes = dict(SINGLE_MESSAGE_MODE=0, ACCIDENTAL_MODE=1, MOBILE_MODE=2, BROADCAST_MODE=3)


def itm_p2p(
    pfl: xr.DataArray,
    climate,
    N0,
    frequency: Quantity,
    pol,
    epsilon,
    sigma,
    mdvar=13,
    time=[50],
    location=[50],
    situation=[50],
):
    """"""

    # Convert Quantities
    pfl, frequency = take_magnitudes((pfl, ureg.m), (frequency, ureg.MHz))

    # Validate inputs
    validate(
        pfl.txagl,
        pfl.rxagl,
        climate,
        time,
        location,
        situation,
        N0,
        frequency,
        pol,
        epsilon,
        sigma,
        mdvar,
    )

    # Distance in kilometers and distance meters
    dkm = pfl.distance_km
    pfl.attrs = {**pfl.attrs, **dict(distance_m=dkm * 1000.0)}
    # Number of points in the pfl
    npts = pfl.distance.size

    # Switch from percentages to ratios
    time = np.array(time) / 100.0
    location = np.array(location) / 100.0
    situation = np.array(situation) / 100.0

    # Compute the average path height, ignoring the first and last 10%
    p10 = int(0.1 * npts)
    hsys = pfl[p10 : (npts - p10)].mean().item()

    zg, gamma_e, Ns = initialize_p2p(frequency, hsys, N0, pol, epsilon, sigma)

    theta_hzn, d_hzn, he, delta_h, theta_hzn_out = pfl_extraction(pfl, gamma_e)

    # Reference attenuation, in dB
    A_ref__db, propmode = longley_rice(
        theta_hzn,
        frequency,
        zg,
        d_hzn,
        he,
        gamma_e,
        Ns,
        delta_h,
        np.array([pfl.txagl, pfl.rxagl]),
        pfl.distance_m,
        "p2p",
    )
    # Free space path loss
    A_fs__db = free_space_pl(pfl.distance_m, frequency)

    # Iterate over variability
    A__db = []
    time_mg, location_mg, situation_mg = np.meshgrid(time, location, situation, indexing="ij")
    for t, l, s in zip(time_mg.ravel(), location_mg.ravel(), situation_mg.ravel()):
        A__db.append(
            variability(t, l, s, he, delta_h, frequency, pfl.distance_m, A_ref__db, climate, mdvar)
            + A_fs__db,
        )

    # Capture necessary attributes
    coords = dict(
        fspl=A_fs__db * ureg.dB,
        ref=A_ref__db * ureg.dB,
        dominant_prop=propmode,
        tx_angle=(theta_hzn_out[0] * ureg.radian).to("degree"),
        rx_angle=(theta_hzn_out[1] * ureg.radian).to("degree"),
        sigma=sigma,
        epsilon=epsilon,
        ipol=pol,
        N0=N0,
        Ns=Ns,
        climate=climate,
        tx_d_hzn=d_hzn[0] * ureg.m,
        rx_d_hzn=d_hzn[1] * ureg.m,
        tx_he=he[0] * ureg.m,
        rx_he=he[1] * ureg.m,
        delta_h=delta_h * ureg.m,
    )
    # Format output in DataArray
    output = xr.DataArray(
        np.array(A__db).reshape(time_mg.shape),
        dims=("time", "location", "situation"),
        coords={**dict(time=time, location=location, situation=situation), **coords},
    )

    return output


def validate(
    htx,
    hrx,
    climate,
    time,
    location,
    situation,
    N0,
    frequency,
    pol,
    epsilon,
    sigma,
    mdvar,
):
    """
    Perform input parameter validation.
    """
    if not 0.5 < htx < 3000.0:
        raise ValueError(f"TX terminal height is out of range: {htx}")
    if not 1.0 < htx < 1000.0:
        warnings.warn(f"TX terminal height near limits: {htx}")

    if not 0.5 < hrx < 3000.0:
        raise ValueError(f"RX terminal height is out of range: {hrx}")
    if not 1.0 < hrx < 1000.0:
        warnings.warn(f"RX terminal height near limits: {hrx}")

    if climate not in range(1, 8):
        raise ValueError(f"Invalid value for radio climate: {climate}")

    if not 250 < N0 < 400:
        raise ValueError(f"Refractivity is out of range: {N0}")

    if not 20.0 < frequency < 20000.0:
        raise ValueError(f"Frequency is out of range: {frequency}")

    if pol not in range(2):
        raise ValueError(f"Invalid value for polarization: {pol}")

    if epsilon < 1:
        raise ValueError(f"Epsilon is out of range: {epsilon}")

    if sigma <= 0:
        raise ValueError(f"Sigma is out of range: {sigma}")

    if (
        (mdvar < 0)
        or (mdvar > 3 and mdvar < 10)
        or (mdvar > 13 and mdvar < 20)
        or (mdvar > 23 and mdvar < 30)
        or (mdvar > 33)
    ):
        raise ValueError("Invalid value for mode of variability")

    if not all([0 <= s <= 100 for s in situation]):
        raise ValueError("Situation percentage out of range.")
    if not all([0 <= t <= 100 for t in time]):
        raise ValueError("Time percentage out of range.")
    if not all([0 <= l <= 100 for l in location]):
        raise ValueError("Location percentage out of range.")


def initialize_p2p(frequency, h_sys, N0, pol, epsilon, sigma):
    """
    Initialize parameters for point-to-point mode.
    """
    # Scale the refractivity based on the elevation above mean sea level
    Ns = N0 * np.exp(-h_sys / 9460.0)  # [TN101, Eq 4.3]

    # gamma_e is the curvature of the effective earth
    gamma_e = GAMMA_A * (1.0 - 0.04665 * np.exp(Ns / 179.3))  # [TN101, Eq 4.4], reworked

    # complex relative permittivity
    epsr = complex(epsilon, 18000 * sigma / frequency)

    # Ground impedance (horizontal polarization)
    zg = np.sqrt(epsr - 1.0)

    # Adjust for vertical polarization
    if pol == 1:
        zg = zg / epsr

    return zg, gamma_e, Ns


def find_horizons(pfl, a_e__meter):

    # Compute radials (ignore radius of earth since it cancels out in the later math)
    z_tx__meter = pfl.txamsl
    z_rx__meter = pfl.rxamsl
    d__meter = pfl.distance_m

    # Set the terminal horizon angles as if the terminals are line-of-sight
    # [TN101, Eq 6.15]
    theta_hzn = [
        (z_rx__meter - z_tx__meter) / d__meter - d__meter / (2 * a_e__meter),
        -(z_rx__meter - z_tx__meter) / d__meter - d__meter / (2 * a_e__meter),
    ]

    # Distances from TX and RX terminals
    d_tx__meter = pfl.distance
    d_rx__meter = d__meter - pfl.distance

    # Terrain angles
    theta_tx = (pfl[1:-1] - z_tx__meter) / d_tx__meter[1:-1] - d_tx__meter[1:-1] / (2 * a_e__meter)
    theta_rx = -(z_rx__meter - pfl[1:-1]) / d_rx__meter[1:-1] - d_rx__meter[1:-1] / (2 * a_e__meter)

    # Initialize horizon distances
    d_hzn = [d__meter, d__meter]
    if theta_tx.max() > theta_hzn[0]:
        theta_hzn[0] = theta_tx.max().item()
        d_hzn[0] = d_tx__meter.isel(distance=theta_tx.argmax()).item()
    if theta_rx.max() > theta_hzn[1]:
        theta_hzn[1] = theta_rx.max().item()
        d_hzn[1] = d_rx__meter.isel(distance=theta_rx.argmax()).item()

    return np.array(theta_hzn), np.array(d_hzn)


def terrain_variability(pfl, d_start, d_end):
    """"""
    # This approach does not match TODO: comeback to this

    # Get pfl from start and end
    pfl_downsel = pfl.loc[d_start:d_end]
    # Perform linear fit
    fitparams = pfl_downsel.polyfit("distance", 1).polyfit_coefficients
    # Create line
    fitline = pfl_downsel.distance * fitparams[0].item() + fitparams[1].item()
    # Get difference
    diff = pfl_downsel - fitline
    # Find 90% and 10% quantiles
    delta_h = diff.quantile(0.9) - diff.quantile(0.10)
    delta_h /= 1.0 - 0.8 * np.exp(-(d_end - d_start) / 50e3)

    return delta_h.item()


def pfl_extraction(pfl, gamma_e):
    """
    Terrain parameter extraction
    """
    # Effective earth radius
    ae = 1 / gamma_e

    # Get horizon angles and distances
    theta_hzn, d_hzn = find_horizons(pfl, ae)
    # Capture for output
    theta_hzn_out = theta_hzn.copy()

    # Total distance
    dmeter = pfl.distance_m
    # NTIA Comments below
    # "In our own work we have sometimes said that consideration of terrain elevations should begin at a point about 15 times the tower height"
    #     - [Hufford, 1982] Page 25
    d_start = min(
        15.0 * pfl.txagl,
        0.1 * d_hzn[0],
    )  # take lesser: 10% of horizon distance or 15x terminal height
    d_end = dmeter - min(
        15.0 * pfl.rxagl,
        0.1 * d_hzn[1],
    )  # << ditto, but measured from the far end of the link >>

    delta_h = terrain_variability(pfl, d_start, d_end)

    # Initialized he
    he = np.array([pfl.txagl, pfl.rxagl])

    if d_hzn.sum() > 1.5 * dmeter:
        # The combined horizon distance is at least 50% larger than the total path distance
        # -> so we are well within the line-of-sight range

        # Perform linear fit
        fit_tx, fit_rx = fit_lstsq(pfl, d_start, d_end)
        he += np.array([max(pfl[0].item() - fit_tx, 0.0), max(pfl[-1].item() - fit_rx, 0.0)])
        d_hzn = np.sqrt(2.0 * he * ae) * np.exp(
            -0.07 * np.sqrt(delta_h / np.max((he, np.array([5.0] * 2)), axis=0)),
        )
        combined_horizons = d_hzn.sum()
        if combined_horizons <= dmeter:
            q = (dmeter / combined_horizons) ** 2
            he *= q
            d_hzn = np.sqrt(2.0 * he * ae) * np.exp(
                -0.07 * np.sqrt(*delta_h / np.max((he, np.array([5.0] * 2)), axis=0)),
            )

        q = np.sqrt(2 * he * ae)
        theta_hzn = (0.65 * delta_h * (q / d_hzn - 1.0) - 2.0 * he) / q

    else:
        # Perform linear fit
        fit_tx, _ = fit_lstsq(pfl, d_start, 0.9 * d_hzn[0])

        # Perform linear fit
        _, fit_rx = fit_lstsq(pfl, dmeter - 0.9 * d_hzn[1], d_end)

        he += np.array([max(pfl[0].item() - fit_tx, 0.0), max(pfl[-1].item() - fit_rx, 0.0)])

    return theta_hzn, d_hzn, he, delta_h, theta_hzn_out


def fit_lstsq(da, start, end):
    """
    Least-squares fit for DataArray.
    """
    # Downselect data
    downsel = da.loc[start:end]
    # Get coefficients
    fitparams = downsel.polyfit("distance", 1).polyfit_coefficients
    # Find y1 and y2
    y1 = fitparams[0].item() * downsel.distance[0].item() + fitparams[1].item()
    y2 = fitparams[0].item() * downsel.distance[-1].item() + fitparams[1].item()

    return y1, y2


def longley_rice(
    theta_hzn,
    f__mhz,
    Z_g,
    d_hzn__meter,
    h_e__meter,
    gamma_e,
    N_s,
    delta_h__meter,
    h__meter,
    d__meter,
    mode,
):
    """
    Compute the reference attenuation using the Longley-Rice method
    """
    d_x__meter = 0

    # Effective earth radius
    a_e__meter = 1 / gamma_e

    # Terrestrial smooth earth horizon distance approximation
    d_hzn_s__meter = np.sqrt(2.0 * h_e__meter * a_e__meter)

    # Maximum line-of-sight distance for smooth earth
    d_sML__meter = d_hzn_s__meter.sum()

    # Maximum line-of-sight distance for actual path
    d_ML__meter = d_hzn__meter.sum()

    # Angular distance of line-of-sight region
    theta_los = -max(theta_hzn.sum(), -d_ML__meter / a_e__meter)

    # Check validity of small angle approximation
    if abs(theta_hzn[0]) > 200e-3:
        warnings.warn("TX horizon angle is large - small angle approximations could break down")
    if abs(theta_hzn[1]) > 200e-3:
        warnings.warn("RX horizon angle is large - small angel approximations could break down")

    # Checks that the actual horizon distance can't be less than 1/10 of the smooth earth horizon distance
    if d_hzn__meter[0] < 0.1 * d_hzn_s__meter[0]:
        warnings.warn("TX horizon distance is less than 1/10 of the smooth earth horizon distance")
    if d_hzn__meter[1] < 0.1 * d_hzn_s__meter[1]:
        warnings.warn("RX horizon distance is less than 1/10 of the smooth earth horizon distance")

    # Checks that the actual horizon distance can't be greater than 3 times the smooth earth horizon distance
    if d_hzn__meter[0] > 3.0 * d_hzn_s__meter[0]:
        warnings.warn(
            "TX horizon distance is greater than 3 times the smooth earth horizon distance",
        )
    if d_hzn__meter[1] > 3.0 * d_hzn_s__meter[1]:
        warnings.warn(
            "RX horizon distance is greater than 3 times the smooth earth horizon distance",
        )

    # Check the surface refractivity
    if N_s < 150:
        raise ValueError("Internally computed surface refractivity value is too small")

    if N_s > 400:
        raise ValueError("Internally computed surface refractivity value is too large")

    if N_s < 250:
        warnings.warn(
            "Internally computed surface refractivity value is small - care must be taken with result",
        )

    # Check effective earth size
    if not 4000000 <= a_e__meter <= 13333333:
        raise ValueError("Internally computed effective earth radius is invalid")

    # Check ground impedance
    if Z_g.real <= abs(Z_g.imag):
        raise ValueError(
            "The imaginary portion of the complex impedance is larger than the real portion",
        )

    # Select two distances far in the diffraction region
    d_3__meter = max(d_sML__meter, d_ML__meter + 5.0 * (a_e__meter**2 / f__mhz) ** (1 / 3.0))
    d_4__meter = d_3__meter + 10.0 * (a_e__meter**2 / f__mhz) ** (1 / 3.0)

    # Compute the diffraction loss at the two distances
    A_3__db = diffraction_loss(
        d_3__meter,
        d_hzn__meter,
        h_e__meter,
        Z_g,
        a_e__meter,
        delta_h__meter,
        h__meter,
        mode,
        theta_los,
        d_sML__meter,
        f__mhz,
    )
    A_4__db = diffraction_loss(
        d_4__meter,
        d_hzn__meter,
        h_e__meter,
        Z_g,
        a_e__meter,
        delta_h__meter,
        h__meter,
        mode,
        theta_los,
        d_sML__meter,
        f__mhz,
    )

    # Compute the slope and intercept of the diffraction line
    M_d = (A_4__db - A_3__db) / (d_4__meter - d_3__meter)
    A_d0__db = A_3__db - M_d * d_3__meter

    d_min__meter = abs(h_e__meter[0] - h_e__meter[1]) / 200e-3

    if d__meter < d_min__meter:
        warnings.warn("Path distance is near its lower limit")
    if d__meter < 1e3:
        warnings.warn("Path distance is small - care must be taken with result")
    if d__meter > 1000e3:
        warnings.warn("Path distance is near its upper limit")
    if d__meter > 2000e3:
        warnings.warn("Path distance is large - care must be taken with result")

    # if the path distance is less than the maximum smooth earth line of sight distance...
    if d__meter < d_sML__meter:
        # Compute the diffraction loss at the maximum smooth earth line of sight distance
        A_sML__db = d_sML__meter * M_d + A_d0__db

        # [ERL 79-ITS 67, Eqn 3.16a], in meters instead of km and with MIN() part below
        d_0__meter = 0.04 * f__mhz * np.prod(h_e__meter)

        if A_d0__db >= 0.0:
            d_0__meter = min(d_0__meter, 0.5 * d_ML__meter)
            # other part of [ERL 79-ITS 67, Eqn 3.16a]
            d_1__meter = d_0__meter + 0.25 * (d_ML__meter - d_0__meter)
            # [ERL 79-ITS 67, Eqn 3.16d]

        else:
            d_1__meter = max(-A_d0__db / M_d, 0.25 * d_ML__meter)

        A_1__db = line_of_sight_loss(
            d_1__meter,
            h_e__meter,
            Z_g,
            delta_h__meter,
            M_d,
            A_d0__db,
            d_sML__meter,
            f__mhz,
        )

        flag = False
        kHat_1__db_per_meter = 0
        kHat_2__db_per_meter = 0

        if d_0__meter < d_1__meter:
            A_0__db = line_of_sight_loss(
                d_0__meter,
                h_e__meter,
                Z_g,
                delta_h__meter,
                M_d,
                A_d0__db,
                d_sML__meter,
                f__mhz,
            )

            q = np.log(d_sML__meter / d_0__meter)

            # [ERL 79 - ITS 67, Eqn 3.20]
            kHat_2__db_per_meter = max(
                0.0,
                (
                    (d_sML__meter - d_0__meter) * (A_1__db - A_0__db)
                    - (d_1__meter - d_0__meter) * (A_sML__db - A_0__db)
                )
                / (
                    (d_sML__meter - d_0__meter) * np.log(d_1__meter / d_0__meter)
                    - (d_1__meter - d_0__meter) * q
                ),
            )

            flag = A_d0__db > 0.0 or kHat_2__db_per_meter > 0.0
            if flag:
                # [ERL 79-ITS 67, Eqn 3.21]
                kHat_1__db_per_meter = (A_sML__db - A_0__db - kHat_2__db_per_meter * q) / (
                    d_sML__meter - d_0__meter
                )

                if kHat_1__db_per_meter < 0.0:
                    kHat_1__db_per_meter = 0.0
                    kHat_2__db_per_meter = max(A_sML__db - A_0__db, 0.0) / q

                    if kHat_2__db_per_meter == 0.0:
                        kHat_1__db_per_meter = M_d
        if not flag:
            kHat_1__db_per_meter = max(A_sML__db - A_1__db, 0.0) / (d_sML__meter - d_1__meter)
            kHat_2__db_per_meter = 0.0

            if kHat_1__db_per_meter == 0.0:
                kHat_1__db_per_meter = M_d

        A_o__db = (
            A_sML__db
            - kHat_1__db_per_meter * d_sML__meter
            - kHat_2__db_per_meter * np.log(d_sML__meter)
        )

        # [ERL 79 - ITS 67, Eqn 3.19]
        A_ref__db = (
            A_o__db + kHat_1__db_per_meter * d__meter + kHat_2__db_per_meter * np.log(d__meter)
        )

    # this is a trans-horizon path
    else:
        # select to points far into the troposcatter region
        d_5__meter = d_ML__meter + 200e3
        d_6__meter = d_ML__meter + 400e3

        # Compute the troposcatter loss at the two distances
        h0 = -1
        A_6__db = troposcatter_loss(
            d_6__meter,
            theta_hzn,
            d_hzn__meter,
            h_e__meter,
            a_e__meter,
            N_s,
            f__mhz,
            theta_los,
            h0,
        )
        A_5__db = troposcatter_loss(
            d_5__meter,
            theta_hzn,
            d_hzn__meter,
            h_e__meter,
            a_e__meter,
            N_s,
            f__mhz,
            theta_los,
            h0,
        )

        # if we got a reasonable prediction value back...
        if A_5__db < 1000.0:
            # Compute the slope of the troposcatter line
            M_s = (A_6__db - A_5__db) / 200e3

            # Find the diffraction-troposcatter transition distance
            d_x__meter = max(
                max(
                    d_sML__meter,
                    d_ML__meter + 1.088 * (a_e__meter**2 / f__mhz) ** (1.0 / 3.0) * np.log(f__mhz),
                ),
                (A_5__db - A_d0__db - M_s * d_5__meter) / (M_d - M_s),
            )
            # Find the diffraction-troposcatter transition distance
            A_s0__db = (M_d - M_s) * d_x__meter + A_d0__db
        else:
            # troposcatter gives no real results - so use diffraction line parameters for tropo line
            M_s = M_d
            A_s0__db = A_d0__db
            d_x__meter = 10e6

        if d__meter > d_x__meter:
            A_ref__db = M_s * d__meter + A_s0__db

        else:
            A_ref__db = M_d * d__meter + A_d0__db

    # set mode of propagation
    delta__meter = d__meter - d_ML__meter
    if int(delta__meter) < 0:
        propmode = "line-of-sight"
    elif d__meter <= d_sML__meter or d__meter <= d_x__meter:
        propmode = (
            "diffraction single horizon" if int(delta__meter) == 0 else "diffraction double horizon"
        )

    else:
        propmode = (
            "troposcatter single horizon"
            if int(
                delta__meter,
            )
            == 0
            else "troposcatter double horizon"
        )

    # Don't allow a negative loss
    A_ref__db = max(A_ref__db, 0.0)

    return A_ref__db, propmode


def diffraction_loss(
    d__meter,
    d_hzn__meter,
    h_e__meter,
    Z_g,
    a_e__meter,
    delta_h__meter,
    h__meter,
    mode,
    theta_los,
    d_sML__meter,
    f__mhz,
):
    """"""
    A_k__db = knife_edge_diffraction(d__meter, f__mhz, a_e__meter, theta_los, d_hzn__meter)

    A_se__db = smooth_earth_diffraction(
        d__meter,
        f__mhz,
        a_e__meter,
        theta_los,
        d_hzn__meter,
        h_e__meter,
        Z_g,
    )

    # Terrain clutter

    # Terrain roughness term, using d_sML__meter, per [ERL 79-ITS 67, page 3-13]
    delta_h_dsML__meter = terrain_roughness(d_sML__meter, delta_h__meter)

    sigma_h_d__meter = sigma_h_function(delta_h_dsML__meter)

    # Clutter factor
    # [ERL 79-ITS 67, Eqn 3.38c]
    q = np.prod(h__meter)
    A_fo__db = min(15.0, 5 * np.log10(1.0 + 1e-5 * q * f__mhz * sigma_h_d__meter))

    # Combined diffraction losses
    # compute the weighting factor in the following calculations

    delta_h_d__meter = terrain_roughness(d__meter, delta_h__meter)

    qk = np.prod(h_e__meter) - q

    # For low antennas with known path parameters, C ~= 10 [ERL 79-ITS 67, page 3-8]
    if mode.lower() == "p2p":
        q += 10.0

    term1 = np.sqrt(1.0 + qk / q)  # square root term in [ERL 79-ITS 67, Eqn 2.23]

    d_ML__meter = d_hzn__meter.sum()  # Maximum line-of-sight distance for actual path
    q = (term1 + (-theta_los * a_e__meter + d_ML__meter) / d__meter) * min(
        delta_h_d__meter * f__mhz / 47.7,
        6283.2,
    )

    # weighting factor [ERL 17-ITS 67, Eqn 3.23]
    w = 25.1 / (25.1 + np.sqrt(q))

    A_d__db = w * A_se__db + (1.0 - w) * A_k__db + A_fo__db

    return A_d__db


def knife_edge_diffraction(d__meter, f__mhz, a_e__meter, theta_los, d_hzn__meter):
    """"""

    d_ML__meter = d_hzn__meter.sum()  # Maximum line-of-sight distance for actual path
    theta_nlos = (
        d__meter / a_e__meter - theta_los
    )  # Angular distance of diffraction region [Algorithm, Eqn 4.12]

    d_nlos__meter = d__meter - d_ML__meter  # Diffraction distance, in meters

    # 1 / (4 pi) = 0.0795775
    # [TN101, Eqn I.7]
    # TODO: do this exact and maybe not just with f_mhz
    v = (
        1
        / (4 * np.pi)
        * (f__mhz / 47.7)
        * theta_nlos**2
        * d_hzn__meter
        * d_nlos__meter
        / (d_nlos__meter + d_hzn__meter)
    )

    A_k__db = fresnel_integral(v).sum()  # [TN101, Eqn I.1]

    return A_k__db


def fresnel_integral(v2):
    """
    Approximate to ideal knife edge diffraction loss
    """
    res = np.ones_like(v2)
    for i, vi in enumerate(v2):
        if vi < 5.76:
            res[i] = (
                6.02 + 9.11 * np.sqrt(vi) - 1.27 * vi
            )  # [TN101v2, Eqn III.24b] and [ERL 79-ITS 67, Eqn 3.27a & 3.27b]
        else:
            res[i] = 12.953 + 10 * np.log10(
                vi,
            )  # [TN101v2, Eqn III.24c] and [ERL 79-ITS 67, Eqn 3.27a & 3.27b]

    return res


def smooth_earth_diffraction(
    d__meter,
    f__mhz,
    a_e__meter,
    theta_los,
    d_hzn__meter,
    h_e__meter,
    Z_g,
):
    """"""
    theta_nlos = d__meter / a_e__meter - theta_los  # [Algorithm, Eqn 4.12]
    d_ML__meter = d_hzn__meter.sum()  # Maximum line-of-sight distance for actual path

    # compute 3 radii
    # which is a_e__meter when theta_los = d_ML__meter / a_e__meter
    # Compute the radius of the effective earth for terminal j using[Volger 1964, Eqn 3] re - arranged
    a__meter = np.hstack(
        (
            (d__meter - d_ML__meter) / (d__meter / a_e__meter - theta_los),
            0.5 * d_hzn__meter**2 / h_e__meter,
        ),
    )  # which is a_e__meter when theta_los = d_ML__meter / a_e__meter

    d__km = (
        np.hstack((a__meter[0] * theta_nlos, d_hzn__meter)) / 1000.0
    )  # angular distance of the "diffraction path"

    # C_0 is the ratio of the 4/3 earth to effective earth (technically Vogler 1964 ratio is 4/3 to effective earth k value), all raised to the (1/3) power.
    # C_0 = (4 / 3k) ^ (1 / 3) [Vogler 1964, Eqn 2]
    C_0 = ((4.0 / 3.0) * a_0__meter / a__meter) ** (1 / 3.0)

    # Vogler 1964, Eqn 6a / 7a]
    K = 0.017778 * C_0 * f__mhz ** (-1 / 3.0) / abs(Z_g)

    # compute B_0 for each radius
    # [Vogler 1964, Fig 4]
    B_0 = 1.607 - K

    # compute x__km for each radius [Vogler 1964, Eqn 2]
    x__km = B_0 * C_0**2 * f__mhz ** (1 / 3.0) * d__km
    x__km[0] += x__km[1:].sum()

    # compute height gain functions
    F_x__db = np.array([height_function(x__km[i], K[i]) for i in range(1, 3)])

    # compute distance function
    G_x__db = 0.05751 * x__km[0] - 10.0 * np.log10(
        x__km[0],
    )  # [TN101, Eqn 8.4] & [Volger 1964, Eqn 13]

    return G_x__db - F_x__db.sum() - 20  # [Algorithm, Eqn 4.20] & [Volger 1964]


def height_function(x__km, K):
    """
    Height Function, F(x, K) for smooth earth diffraction
    """
    if x__km < 200.0:
        w = -np.log(K)

        if K < 1e-5 or x__km * w**3 > 5495.0:
            result = -117.0

            if x__km > 1.0:
                result += 17.372 * np.log(x__km)

        else:
            result = 2.5e-5 * x__km**2 / K - 8.686 * w - 15.0

    else:
        result = 0.05751 * x__km - 4.343 * np.log(x__km)

        if x__km < 2000:
            w = 0.0134 * x__km * np.exp(-0.005 * x__km)
            result = (1.0 - w) * result + w * (17.372 * np.log(x__km) - 117.0)

    return result


def terrain_roughness(d__meter, delta_h__meter):
    """
    Compute delta_h_d
    """
    # [ERL 79 - ITS 67, Eqn 3],  with distance in meters instead of kilometers
    return delta_h__meter * (1.0 - 0.8 * np.exp(-d__meter / 50e3))


def sigma_h_function(delta_h__meter):
    """
    Compute sigma h function
    """
    # RMS deviation of terrain and terrain clutter within the limits of the first Fresnel zone in the dominant reflecting plane"
    #  [ERL 79-ITS 67, Eqn 3.6a]
    return 0.78 * delta_h__meter * np.exp(-0.5 * delta_h__meter**0.25)


def line_of_sight_loss(d__meter, h_e__meter, Z_g, delta_h__meter, M_d, A_d0, d_sML__meter, f__mhz):
    """
    Compute the loss in the line-of-sight region
    """
    delta_h_d__meter = terrain_roughness(d__meter, delta_h__meter)

    sigma_h_d__meter = sigma_h_function(delta_h_d__meter)

    # wavenumber, k
    # TODO: Make exact
    wn = f__mhz / 47.7

    # [Algorithm, Eqn 4.46]
    sin_psi = h_e__meter.sum() / np.sqrt(d__meter**2 + h_e__meter.sum() ** 2)

    # [Algorithm, Eqn 4.47]
    R_e = (sin_psi - Z_g) / (sin_psi + Z_g) * np.exp(-min(10.0, wn * sigma_h_d__meter * sin_psi))

    # q = Magnitude of R_e', [Algorithm, Eqn 4.48]
    q = R_e.real**2 + R_e.imag**2
    if q < 0.25 or q < sin_psi:
        R_e *= np.sqrt(sin_psi / q)

    # phase difference between rays, [Algorithm, Eqn 4.49]
    delta_phi = wn * 2.0 * np.prod(h_e__meter) / d__meter

    # [Algorithm, Eqn 4.50]
    if delta_phi > np.pi / 2.0:
        delta_phi = np.pi - (np.pi / 2.0) ** 2 / delta_phi

    # Two-ray attenuation
    rr = complex(np.cos(delta_phi), -np.sin(delta_phi)) + R_e
    A_t__db = -10 * np.log10(rr.real**2 + rr.imag**2)

    # Extended diffraction attenuation
    A_d__db = M_d * d__meter + A_d0

    # weighting factor
    w = 1 / (1 + f__mhz * delta_h__meter / max(10e3, d_sML__meter))

    A_los__db = w * A_t__db + (1 - w) * A_d__db

    return A_los__db


def free_space_pl(d__meter, f__mhz):
    """
    Free space path loss basic equation.
    TODO: exact
    """
    return 32.45 + 20.0 * np.log10(f__mhz) + 20.0 * np.log10(d__meter / 1000.0)


def troposcatter_loss(
    d__meter,
    theta_hzn,
    d_hzn__meter,
    h_e__meter,
    a_e__meter,
    N_s,
    f__mhz,
    theta_los,
    h0,
):
    """
    Troposcatter loss
    """
    # wavenumber, k
    wn = f__mhz / 47.7

    # short-circuit calculations if already greater than 15 dB
    if h0 > 15.0:
        H_0 = h0
    else:
        ad = d_hzn__meter[0] - d_hzn__meter[1]
        rr = h_e__meter[1] / h_e__meter[0]
        # ensure correct frame of reference
        if ad < 0.0:
            ad = -ad
            rr = 1.0 / rr

        theta = theta_hzn[0] + theta_hzn[1] + d__meter / a_e__meter  # angular distance, in radians

        # [TN101, Eqn 9.4a]
        r_1 = 2.0 * wn * theta * h_e__meter[0]
        r_2 = 2.0 * wn * theta * h_e__meter[1]

        if r_1 < 0.2 and r_2 < 0.2:
            return 1001  # "If both r_1 and r_2 are less than 0.2 the function A_scat is not defined (or is infinite)" [Algorithm, page 11]

        s = (d__meter - ad) / (d__meter + ad)  # asymmetry parameter

        # "In all of this, we truncate the values of s and q at 0.1 and 10" [Algorithm, page 16]
        q = min(max(0.1, rr / s), 10.0)  # TN101, Eqn 9.5
        s = max(0.1, s)  # TN101, Eqn 9.5

        h_0__meter = (
            (d__meter - ad) * (d__meter + ad) * theta * 0.25 / d__meter
        )  # height of cross-over, [Algorithm, 4.66] [TN101v1, 9.3b]

        Z_0__meter = 1.7556e3  # Scale height, [Algorithm, 4.67]
        Z_1__meter = 8.0e3  # [Algorithm, 4.67]
        eta_s = (h_0__meter / Z_0__meter) * (
            1.0
            + (0.031 - N_s * 2.32e-3 + N_s**2 * 5.67e-6)
            * np.exp(-(min(1.7, h_0__meter / Z_1__meter) ** 6))
        )  # Scattering efficiency factor, eta_s [TN101 Eqn 9.3a]

        H_00 = (
            h0_function(r_1, eta_s) + h0_function(r_2, eta_s)
        ) / 2  # First term in TN101v1, Eqn 9.5
        Delta_H_0 = min(H_00, 6.0 * (0.6 - np.log10(max(eta_s, 1.0))) * np.log10(s) * np.log10(q))

        H_0 = H_00 + Delta_H_0  # TN101, Eqn 9.5
        H_0 = max(H_0, 0.0)  # "If Delta_H_0 would make H_0 negative, use H_0 = 0" [TN101v1, p9.4]

        if eta_s < 1.0:  # if <=1, interpolate with the special case of eta_s = 0
            H_0 = eta_s * H_0 + (1.0 - eta_s) * 10 * np.log10(
                ((1.0 + np.sqrt(2) / r_1) * (1.0 + np.sqrt(2) / r_2)) ** 2
                * (r_1 + r_2)
                / (r_1 + r_2 + 2 * np.sqrt(2)),
            )

        # "If, at d_5, calculations show that H_0 will exceed 15 dB, they are replaced by the value it has at d_6" [Algorithm, page 12]
        if H_0 > 15.0 and h0 >= 0.0:
            H_0 = h0
    h0 = H_0

    th = d__meter / a_e__meter - theta_los

    D_0__meter = 40e3  # [Algorithm, 6.8]

    H__meter = 47.7  # [Algorithm, 4.63]
    return (
        ff_function(th * d__meter)
        + 10 * np.log10(wn * H__meter * th**4)
        - 0.1 * (N_s - 301.0) * np.exp(-th * d__meter / D_0__meter)
        + H_0
    )
    # [Algorithm, 4.63]


def h0_function(r, eta_s):
    """
    Troposcatter frequency gain function, H_0(), from [TN101v1, Ch 9.2]
    """
    eta_s = min(max(eta_s, 1), 5)
    # range 1 <= eta_s <= 5

    i = int(eta_s)
    # integer part of eta_s
    q = eta_s - i
    # decimal part of eta_s

    result = h0_curve(i - 1, r)

    if q != 0.0:  # interpolate with next curve, if needed
        result = (1.0 - q) * result + q * h0_curve(i, r)

    return result


def h0_curve(j, r):
    """
    Curve fit helper function to approximate H_0()
    """
    # values from [Algorithm, 6.13]
    a = [25.0, 80.0, 177.0, 395.0, 705.0]
    b = [24.0, 45.0, 68.0, 80.0, 105.0]

    return 10 * np.log10(
        1 + a[j] * (1.0 / r) ** 4 + b[j] * (1.0 / r) ** 2,
    )  # related to TN101v2, Eqn III.49, but from [Algorithm, 6.13]


def ff_function(td):
    """
    The attenuation function, F(th * d)
    """
    # constants from [Algorithm, 6.9]
    a = [133.4, 104.6, 71.8]
    b = [0.332e-3, 0.212e-3, 0.157e-3]
    c = [-10, -2.5, 5]

    # select the set of values to use
    if td <= 10e3:  # <= 10 km
        i = 0
    elif td <= 70e3:  # 10 km to 70 km
        i = 1
    else:  # > 70 km
        i = 2

    F_0 = a[i] + b[i] * td + c[i] * np.log10(td)  # [Algorithm, 6.9]

    return F_0


def variability(
    time,
    location,
    situation,
    h_e__meter,
    delta_h__meter,
    f__mhz,
    d__meter,
    A_ref__db,
    climate,
    mdvar,
):
    """
    Compute the variability loss

    References:
    ----------
    https://github.com/NTIA/itm/blob/master/src/Variability.cpp
    """
    # Asymptotic values from TN101, Fig 10.13
    # -> approximate to TN101v2 Eqn III.69 & III.70
    # -> to describe the curves for each climate
    all_year = [
        [-9.67, -0.62, 1.26, -9.21, -0.62, -0.39, 3.15],
        [12.7, 9.19, 15.5, 9.05, 9.19, 2.86, 857.9],
        [144.9e3, 228.9e3, 262.6e3, 84.1e3, 228.9e3, 141.7e3, 2222.0e3],
        [190.3e3, 205.2e3, 185.2e3, 101.1e3, 205.2e3, 315.9e3, 164.8e3],
        [133.8e3, 143.6e3, 99.8e3, 98.6e3, 143.6e3, 167.4e3, 116.3e3],
    ]

    bsm1 = [2.13, 2.66, 6.11, 1.98, 2.68, 6.86, 8.51]
    bsm2 = [159.5, 7.67, 6.65, 13.11, 7.16, 10.38, 169.8]
    xsm1 = [762.2e3, 100.4e3, 138.2e3, 139.1e3, 93.7e3, 187.8e3, 609.8e3]
    xsm2 = [123.6e3, 172.5e3, 242.2e3, 132.7e3, 186.8e3, 169.6e3, 119.9e3]
    xsm3 = [94.5e3, 136.4e3, 178.6e3, 193.5e3, 133.5e3, 108.9e3, 106.6e3]

    bsp1 = [2.11, 6.87, 10.08, 3.68, 4.75, 8.58, 8.43]
    bsp2 = [102.3, 15.53, 9.60, 159.3, 8.12, 13.97, 8.19]
    xsp1 = [636.9e3, 138.7e3, 165.3e3, 464.4e3, 93.2e3, 216.0e3, 136.2e3]
    xsp2 = [134.8e3, 143.7e3, 225.7e3, 93.1e3, 135.9e3, 152.0e3, 188.5e3]
    xsp3 = [95.6e3, 98.6e3, 129.7e3, 94.2e3, 113.4e3, 122.7e3, 122.9e3]

    C_D = [1.224, 0.801, 1.380, 1.000, 1.224, 1.518, 1.518]  # [Algorithm, Table 5.1], C_d
    z_D = [1.282, 2.161, 1.282, 20.0, 1.282, 1.282, 1.282]  # [Algorithm, Table 5.1], z_d

    bfm1 = [1.0, 1.0, 1.0, 1.0, 0.92, 1.0, 1.0]
    bfm2 = [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0]
    bfm3 = [0.0, 0.0, 0.0, 0.0, 1.77, 0.0, 0.0]

    bfp1 = [1.0, 0.93, 1.0, 0.93, 0.93, 1.0, 1.0]
    bfp2 = [0.0, 0.31, 0.0, 0.19, 0.31, 0.0, 0.0]
    bfp3 = [0.0, 2.00, 0.0, 1.79, 2.00, 0.0, 0.0]

    z_T = inverse_complimentary_cdf(time)
    z_L = inverse_complimentary_cdf(location)
    z_S = inverse_complimentary_cdf(situation)

    climate -= 1  # 0-based indexes

    wn = f__mhz / 47.7

    # compute the effective distance
    d_ex__meter = (
        np.sqrt(2 * a_9000__meter * h_e__meter[0])
        + np.sqrt(2 * a_9000__meter * h_e__meter[1])
        + (575.7e12 / wn) ** (1 / 3.0)
    )  # [Algorithm, Eqn 5.3]

    if d__meter < d_ex__meter:
        d_e__meter = 130e3 * d__meter / d_ex__meter
    else:
        d_e__meter = 130e3 + d__meter - d_ex__meter

    # -------------------------------------
    # situation variability calcs

    # if mdvar >= 20, then "Direct situation variability is to be eliminated as it should when
    #                       considering interference problems.  Note that there may still be a
    #                       small residual situation variability" [Hufford, 1982]
    plus20 = mdvar >= 20
    if plus20:
        mdvar -= 20
        sigma_S = 0.0
    else:
        D__meter = 100e3  # Scale distance, D = 100 km
        sigma_S = 5.0 + 3.0 * np.exp(-d_e__meter / D__meter)  # [Algorithm, Eqn 5.10]

    #
    # -------------------------------------

    plus10 = mdvar >= 10
    if plus10:
        mdvar -= 10

    V_med__db = curve(
        all_year[0][climate],
        all_year[1][climate],
        all_year[2][climate],
        all_year[3][climate],
        all_year[4][climate],
        d_e__meter,
    )

    if mdvar == modes["SINGLE_MESSAGE_MODE"]:
        z_T = z_S
        z_L = z_S

    elif mdvar == modes["ACCIDENTAL_MODE"]:
        z_L = z_S
    elif mdvar == modes["MOBILE_MODE"]:
        z_L = z_T
    # else using Broadcast Mode (no additional operations)

    if abs(z_T) > 3.10 or abs(z_L) > 3.10 or abs(z_S) > 3.10:
        warnings.warn(
            "One of the provided variabilities is located far in the tail of its distribution",
        )

    # -------------------------------------
    # location variability calcs

    if plus10:
        sigma_L = 0.0
    else:
        delta_h_d__meter = terrain_roughness(d__meter, delta_h__meter)
        sigma_L = (
            10.0 * wn * delta_h_d__meter / (wn * delta_h_d__meter + 13.0)
        )  # Context of [Algorithm, Eqn 5.9]

    Y_L = sigma_L * z_L

    #
    # -------------------------------------

    # -------------------------------------
    # time variability calcs
    q = np.log(0.133 * wn)
    g_minus = bfm1[climate] + bfm2[climate] / (pow(bfm3[climate] * q, 2) + 1.0)
    g_plus = bfp1[climate] + bfp2[climate] / (pow(bfp3[climate] * q, 2) + 1.0)

    sigma_T_minus = (
        curve(bsm1[climate], bsm2[climate], xsm1[climate], xsm2[climate], xsm3[climate], d_e__meter)
        * g_minus
    )
    sigma_T_plus = (
        curve(bsp1[climate], bsp2[climate], xsp1[climate], xsp2[climate], xsp3[climate], d_e__meter)
        * g_plus
    )

    sigma_TD = C_D[climate] * sigma_T_plus
    tgtd = (sigma_T_plus - sigma_TD) * z_D[climate]

    if z_T < 0.0:
        sigma_T = sigma_T_minus
    elif z_T <= z_D[climate]:
        sigma_T = sigma_T_plus
    else:
        sigma_T = sigma_TD + tgtd / z_T
    Y_T = sigma_T * z_T

    #
    # -------------------------------------

    Y_S_temp = (
        pow(sigma_S, 2) + pow(Y_T, 2) / (7.8 + pow(z_S, 2)) + pow(Y_L, 2) / (24.0 + pow(z_S, 2))
    )  # Part of[Algorithm, Eqn 5.11]
    if mdvar == modes["SINGLE_MESSAGE_MODE"]:
        Y_R = 0.0
        Y_S = np.sqrt(pow(sigma_T, 2) + pow(sigma_L, 2) + Y_S_temp) * z_S

    elif mdvar == modes["ACCIDENTAL_MODE"]:
        Y_R = Y_T
        Y_S = np.sqrt(pow(sigma_L, 2) + Y_S_temp) * z_S

    elif mdvar == modes["MOBILE_MODE"]:
        Y_R = np.sqrt(pow(sigma_T, 2) + pow(sigma_L, 2)) * z_T
        Y_S = np.sqrt(Y_S_temp) * z_S

    else:  # BROADCAST_MODE
        Y_R = Y_T + Y_L
        Y_S = np.sqrt(Y_S_temp) * z_S

    result = A_ref__db - V_med__db - Y_R - Y_S

    # [Algorithm, Eqn 52]
    if result < 0.0:
        result = result * (29.0 - result) / (29.0 - 10.0 * result)

    return result


def inverse_complimentary_cdf(q):
    """
    This function computes the inverse complementary cumulative distribution function approximation as described in
    Formula 26.2.23 in Abramowitz & Stegun. This approximation has an error of  abs(epsilon(p)) < 4.5e-4

    References:
    ----------
    https://github.com/NTIA/itm/blob/master/src/InverseComplementaryCumulativeDistributionFunction.cpp

    """
    C_0 = 2.515516
    C_1 = 0.802853
    C_2 = 0.010328
    D_1 = 1.432788
    D_2 = 0.189269
    D_3 = 0.001308

    x = q
    if q > 0.5:
        x = 1.0 - x

    T_x = np.sqrt(-2.0 * np.log(x))

    zeta_x = ((C_2 * T_x + C_1) * T_x + C_0) / (((D_3 * T_x + D_2) * T_x + D_1) * T_x + 1.0)

    Q_q = T_x - zeta_x

    if q > 0.5:
        Q_q = -Q_q

    return Q_q


def curve(c1, c2, x1, x2, x3, d_e__meter):
    """
    Curve helper function for TN101v2 Eqn III.69 & III.70

    References:
    ----------
    https://github.com/NTIA/itm/blob/master/src/Variability.cpp
    """
    return (
        (c1 + c2 / (1.0 + pow((d_e__meter - x2) / x3, 2)))
        * (pow(d_e__meter / x1, 2))
        / (1.0 + (pow(d_e__meter / x1, 2)))
    )
