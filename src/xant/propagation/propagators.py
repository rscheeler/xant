from typing import Union

import numpy as np
import xarray as xr
from hics import HCS
from hics.geo.dem import geocent2llh, llh2geocent
from hics.geo.geoutils import get_surface_profile
from hics.geo.transforms import geo_mid
from loguru import logger
from pint import Quantity

from .. import ureg
from .itm import itm_p2p as _itmrflink_p2p
from .itm.utils import _GND_EPS, _GND_SGM, CLIMATE_ZONES, ITM_POLARIZATION, REFRACTIVITY


def fspl(
    txcs: HCS = None,
    rxcs: HCS = None,
    frequency: Quantity | xr.DataArray = None,
    additional_loss: Quantity | xr.DataArray = None,
    **kwargs,
):
    """
    Returns the free-space path loss. Distance is calculated between the two points - not a geodesic length.

    Parameters
    ----------
    txcs : HCS
        Coordinate system of the transmitter
    rxcs : HCS
        Coordinate system of the receiver
    frequency : Quantity or xr.DataArray of Quantities
        Frequencies to calculate the path loss at
    """
    # Determine Distance from coordinate systems
    distance = txcs.relative_distance(rxcs)

    # Propagation loss as a gain number
    wavelength = 1 / frequency * ureg.speed_of_light

    # Electrical Distance
    electrical_distance = distance / wavelength

    # Only valid in far-field distance >> wavelength
    # Set distance_electrical less than 4*np.pi to 4*np.pi
    spher_ster = 1 / (4 * np.pi)
    mask = electrical_distance < spher_ster
    electrical_distance = xr.where(mask, spher_ster, electrical_distance)

    # Calculate propagation loss
    prop_loss = (4 * np.pi * electrical_distance) ** 2

    # Convert to dB
    prop_loss.data = prop_loss.data.to("dB")

    # Add in additional loss
    if additional_loss is not None:
        # Convert to base units
        prop_loss.data = prop_loss.data.to_base_units()
        if isinstance(additional_loss, xr.DataArray):
            additional_loss.data = additional_loss.data.to_base_units()
        else:
            additional_loss = additional_loss.to_base_units()

        # Add the loss
        prop_loss *= additional_loss

        # Convert to dB
        prop_loss.data = prop_loss.data.to("dB")

    # Add distance coordinate
    prop_loss = prop_loss.assign_coords(distance=distance)

    return prop_loss


def mask_los(prop_loss: xr.DataArray, txcs: HCS, rxcs: HCS):
    """
    Masks the path loss based on approximation for the radar line-of-sight. Note, this assumes the ground height is the
    same for transmit and receive so it is just an approximation. For a more accurate line-of-sight model utilize a
    propagation model that includes terrain.

    Parameters
    ----------
    prop_loss : xr.DataArray
        Propagation loss array
    txcs : HCS
        Coordinate system of the transmitter
    rxcs : HCS
        Coordinate system of the receiver

    References:
    ----------
    https://en.wikipedia.org/wiki/Line-of-sight_propagation
    """
    # Get transmit height above ground by determining altitude of the ground
    txlla = txcs.lla
    gndlla = geocent2llh(
        *llh2geocent(txlla[0], txlla[1], xr.zeros_like(txlla[2]), alt_above_ground=True),
    )
    htx = txlla[2] - gndlla[-1]
    htx = xr.where(htx < 0, 0, htx)

    # Determine observation height above ground - note negative heights are masked
    rxlla = rxcs.lla
    gndlla = geocent2llh(
        *llh2geocent(rxlla[0], rxlla[1], xr.zeros_like(rxlla[2]), alt_above_ground=True),
    )
    ha = rxlla[-1] - gndlla[-1]
    ha = xr.where(ha < 0, 0, ha)

    # Mask line of sight (assumption here is that the ground height of tx and rx is the same
    # This obviously isn't true, but gives a good approximation of the line of sight, for a better estimate using
    # a propagation model that utilizes terrain should be used
    # https://en.wikipedia.org/wiki/Horizon#Objects_above_the_horizon

    # Need to convert to a magnitude so the Quantities work out properly
    htx.data = htx.data.to("m").magnitude
    ha.data = ha.data.to("m").magnitude

    # Compute the horizon distance
    dlos = 3.57 * ureg.km * (np.sqrt(htx) + np.sqrt(ha))

    # Mask the propagation loss if it is greater than the horizon distance
    mask = prop_loss.distance > dlos
    prop_loss = xr.where(mask, np.nan, prop_loss)

    return prop_loss


def itm_rflink(
    txcs: HCS = None,
    rxcs: HCS = None,
    frequency: Quantity | xr.DataArray = None,
    gnd: str = "good",
    additional_loss: Quantity | xr.DataArray = None,
    clutter: bool = True,
    **kwargs,
):
    """
    Returns the path loss from ITM. Note ITM uses geodesic length not the slant range. This may result in slight
    discrepancies for nodes that are close together and offset in height as compared to the fspl method.

    Parameters
    ----------
    txcs : HCS
        Coordinate system of the transmitter
    rxcs : HCS
        Coordinate system of the receiver
    frequency : Quantity or xr.DataArray of Quantities
        Frequencies to calculate the path loss at

    **kwargs
    Any changes to the default parameters
    """
    logger.debug("ITM PROP")
    # Defaults
    if "reliability" not in kwargs and "time" not in kwargs:
        kwargs["time"] = [50]
    elif "reliability" in kwargs:
        kwargs["time"] = kwargs.pop("reliability")
    if "confidence" not in kwargs and "situation" not in kwargs:
        kwargs["situation"] = [50]
    elif "confidence" in kwargs:
        kwargs["situation"] = kwargs.pop("confidence")

    # Determine surface profile
    surface_profile = get_surface_profile(txcs, rxcs)

    # Select clutter or surface
    # if clutter:
    #     surface_profile = surface_profile.lc_profile

    # else:
    #     surface_profile = surface_profile.surface_profile
    surface_profile = surface_profile.surface_profile
    if isinstance(frequency, xr.DataArray):
        # Broadcast surface profile
        surface_profile, tmp_freq = xr.broadcast(surface_profile, frequency)

        if "distance" in surface_profile.dims:
            tmp = surface_profile.isel(distance=0)
            tmp, frequency = xr.broadcast(tmp, frequency)
        else:
            frequency = tmp_freq

    # Loop through profiles if not distance
    if surface_profile.dims != ("distance",):
        sdims = list(surface_profile.dims)
        if "distance" in sdims:
            sdims.pop(sdims.index("distance"))

        # Loop through all profiles
        itm_res = []
        idx_grids = np.meshgrid(*[np.arange(surface_profile[d].size) for d in sdims])
        idx_grids = [idxs.ravel() for idxs in idx_grids]
        for idxs in zip(*idx_grids):
            # Create selection dictionary
            isel_dict = {k: v for k, v in zip(sdims, idxs)}

            surf_prof = surface_profile.isel(isel_dict)
            # Capture coords before item selection
            surf_prof_coords = dict(surf_prof.coords)
            # Remove distance vars
            pop_coords = []
            for k, v in surf_prof_coords.items():
                if "distance" in v.dims:
                    pop_coords.append(k)
            for k in pop_coords:
                surf_prof_coords.pop(k)

            # Select item
            if surf_prof.shape == ():
                surf_prof = surf_prof.item()

            # Get mid point for selecting input data
            lat_mid, lon_mid = geo_mid(
                np.rad2deg(surf_prof.lon[0]),
                np.rad2deg(surf_prof.lat[0]),
                np.rad2deg(surf_prof.lon[-1]),
                np.rad2deg(surf_prof.lat[-1]),
            )

            # Climate selection (1=equatorial,
            # 2=continental subtropical, 3=maritime subtropical,
            # 4=desert, 5=continental temperate,
            # 6=maritime temperate overland,
            # 7=maritime temperate, oversea (5 is the default)
            # Get climate data from mid point
            climate = round(CLIMATE_ZONES.interp(lat=lat_mid, lon=lon_mid).item())

            # Surface refractivity (N-units): also controls effective Earth radius - get from mid point
            N0 = REFRACTIVITY.interp(lat=lat_mid, lon=lon_mid).item()

            # Add frequency user params
            if "frequency" in surface_profile.dims:
                f = frequency.isel(isel_dict).item()

            else:
                f = frequency

            # Call the itm logic function for each polarization and append
            for polk, polv in ITM_POLARIZATION.items():
                # Polarization selection (0=horizontal, 1=vertical)
                res = _itmrflink_p2p(
                    surf_prof,
                    climate,
                    N0,
                    f,
                    polv,
                    _GND_EPS[gnd],
                    _GND_SGM[gnd],
                    mdvar=3,
                    **kwargs,
                )
                attrs = res.attrs
                # Squeeze dimensions (confidence and reliability)
                res = res.squeeze()

                # Add distance coord
                coords = dict(distance=(surf_prof.distance_km * ureg.km).to("m"), polarization=polk)
                # Combine distance, existing coords, and attrs into the coordinates
                coords = {**coords, **surf_prof_coords, **attrs}
                # Remove attrs since they are now capture in coords
                res.attrs = dict()
                # Update coords
                res = res.assign_coords(coords)
                # Expand for dimensions - this is necessary to using xr.combine_by_coords later
                res = res.expand_dims(sdims + ["polarization"])
                # Append to running list
                itm_res.append(res)

    else:
        # Get mid point for selecting input data
        lat_mid, lon_mid = geo_mid(txcs.lla[1], txcs.lla[0], rxcs.lla[1], rxcs.lla[0])

        # Climate selection (1=equatorial,
        # 2=continental subtropical, 3=maritime subtropical,
        # 4=desert, 5=continental temperate,
        # 6=maritime temperate overland,
        # 7=maritime temperate, oversea (5 is the default)
        # Get climate data from mid point
        climate = round(CLIMATE_ZONES.interp(lat=lat_mid, lon=lon_mid).item())

        # Surface refractivity (N-units): also controls effective Earth radius - get from mid point
        N0 = REFRACTIVITY.interp(lat=lat_mid, lon=lon_mid).item()

        # Call the itm logic function for each polarization and append
        itm_res = []
        for polk, polv in ITM_POLARIZATION.items():
            # Polarization selection (0=horizontal, 1=vertical)
            # Call the itm logic function
            res = _itmrflink_p2p(
                surface_profile,
                climate,
                N0,
                frequency,
                polv,
                _GND_EPS[gnd],
                _GND_SGM[gnd],
                mdvar=3,
                **kwargs,
            )
            # user_params, surface_profile.data.magnitude, gnd=gnd)
            attrs = res.attrs
            # Squeeze dimensions (confidence and reliability)
            res = res.squeeze()
            # Add distance coord
            coords = dict(
                distance=(surface_profile.distance_km * ureg.km).to("m"),
                polarization=polk,
            )
            # Combine distance, existing coords, and attrs into the coordinates
            coords = {**coords, **attrs}
            # Remove attrs since they are now capture in coords
            res.attrs = dict()
            # Update coords
            res = res.assign_coords(coords)
            # Expand for dimensions - this is necessary to using xr.combine_by_coords later
            res = res.expand_dims(list(surface_profile.dims) + ["polarization"])
            # Append to running list
            itm_res.append(res)

    # Make into DataArray
    itm_res = xr.combine_by_coords(itm_res)

    # Add decibel units
    itm_res.data = itm_res.data * ureg.dB

    # Add in additional loss
    if additional_loss is not None:
        # Convert to base units
        itm_res.data = itm_res.data.to_base_units()
        if isinstance(additional_loss, xr.DataArray):
            additional_loss.data = additional_loss.data.to_base_units()
        else:
            additional_loss = additional_loss.to_base_units()

        # Add the loss
        itm_res *= additional_loss

        # Convert to dB
        itm_res.data = itm_res.data.to("dB")

    return itm_res
