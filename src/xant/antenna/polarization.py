from typing import Optional

import numpy as np
import xarray as xr
from hics.utils import vector_norm
from loguru import logger
from pint import Quantity
from scipy.spatial.transform import Rotation

from ..utils import conversions
from ..utils.conversions import uvw2phitheta
from ..utils.geometry import apply_rotation

SUPPORTED_POLS = [
    ["apolar"],
    ["theta", "phi"],
    ["x", "y", "z"],
    ["rhcp", "lhcp"],
    ["l3x", "l3y"],
    ["p45", "m45"],
]
SUPPORTED_POLS_SET = [set(p) for p in SUPPORTED_POLS]


def thetaphi2xyz(phi=None, theta=None):
    """
    Generate Jones matrix to convert theta phi polarization in phitheta coordinate frame to cartesian
    x, y, z polarizations.

    .. math::
        E_x &= E_{\\theta} \cos\\theta\cos\phi - E_{\phi} \sin\phi

        E_y &= E_{\\theta} \cos\\theta\sin\phi + E_{\phi} \cos\phi

        E_z &= -E_{\\theta} \sin\\theta
    """
    A = np.array(
        [
            [np.cos(theta) * np.cos(phi), xr.ones_like(theta) * (-np.sin(phi))],
            [np.cos(theta) * np.sin(phi), xr.ones_like(theta) * np.cos(phi)],
            [-np.sin(theta) * xr.ones_like(phi), xr.zeros_like(theta) * xr.zeros_like(phi)],
        ]
    )
    tmp = xr.zeros_like(theta) * xr.zeros_like(phi)
    dims = ["new_polarization", "polarization"] + list(tmp.dims)
    coords = {**dict(new_polarization=["x", "y", "z"], polarization=["theta", "phi"]), **tmp.coords}
    A = xr.DataArray(A, dims=dims, coords=coords)

    return A


def thetaphi2l3xl3y(phi=None, theta=None):
    """
    Returns jones matrix to convert theta, phi polarization in phitheta coordinate frame to Ludwig's III definition
    of co-pol aligned with the x-axis (l3x) and cross-pol aligned with the y-axis (l3y).

    .. math::
        E_{l3x} &= E_{\\theta} \cos\phi - E_{\phi} \sin\phi

        E_{l3y} &= E_{\\theta} \sin\phi + E_{\phi} \cos\phi
    """
    phi_mg = phi * xr.ones_like(theta)
    A = np.array([[np.cos(phi_mg), -np.sin(phi_mg)], [np.sin(phi_mg), np.cos(phi_mg)]])

    dims = ["new_polarization", "polarization"] + list(phi_mg.dims)
    coords = {
        **dict(new_polarization=["l3x", "l3y"], polarization=["theta", "phi"]),
        **phi_mg.coords,
    }
    A = xr.DataArray(A, dims=dims, coords=coords)

    return A


def thetaphi2rhcplhcp(phi=None, theta=None):
    """
    Returns Jones matrix to convert theta, phi polarization to circular RHCP and LHCP.

    .. math::
        E_{RHCP} &= \\frac{E_{\\theta} + j E_{\phi}}{\sqrt{2}}

        E_{LHCP} &= \\frac{E_{\\theta} - j E_{\phi}}{\sqrt{2}}
    """
    one_mg = xr.ones_like(phi) * xr.ones_like(theta)
    A = np.array([[one_mg, 1j * one_mg], [one_mg, -1j * one_mg]]) / np.sqrt(2)
    dims = ["new_polarization", "polarization"] + list(one_mg.dims)
    coords = {
        **dict(new_polarization=["rhcp", "lhcp"], polarization=["theta", "phi"]),
        **one_mg.coords,
    }
    A = xr.DataArray(A, dims=dims, coords=coords)

    return A


def thetaphi2p45m45(phi=None, theta=None):
    """
    Returns Jones matrix to convert theta, phi polarization to slant linear +/- 45.

    .. math::
        E_{+45} &= \\frac{E_{\\theta} + E_{\phi}}{\sqrt{2}}

        E_{-45} &= \\frac{E_{\\theta} - E_{\phi}}{\sqrt{2}}
    """
    one_mg = xr.ones_like(phi) * xr.ones_like(theta)
    A = np.array([[one_mg, one_mg], [one_mg, -1 * one_mg]]) / np.sqrt(2)
    dims = ["new_polarization", "polarization"] + list(one_mg.dims)
    coords = {
        **dict(new_polarization=["p45", "m45"], polarization=["theta", "phi"]),
        **one_mg.coords,
    }
    A = xr.DataArray(A, dims=dims, coords=coords)

    return A


polarization_transforms = [thetaphi2xyz, thetaphi2l3xl3y, thetaphi2rhcplhcp, thetaphi2p45m45]


def project_all_polarizations(data, convert_kwargs: dict | None = None):
    """
    Takes data in and generates all the polarizations by projecting the input basis to other basis
    """
    # Only project if not apolar or if all polarizations are present
    if data.polarization.shape != (1,) and set(data.polarization.values) != set(
        [item for subl in SUPPORTED_POLS for item in subl]
    ):
        # Default convert kwargs
        if convert_kwargs is None:
            convert_kwargs = dict()

        # Store attrs
        attrs = data.attrs

        # Validate polarization basis
        basis = "".join(SUPPORTED_POLS[SUPPORTED_POLS_SET.index(set(data.polarization.values))])

        # Transform spatial data into phitheta coordinate frame as that is what is required
        angles = [data.coords[a] for a in conversions.COORDINATE_DIMS[data.coordinate_frame]]
        # Broadcast angles as this is necessary for some conversions
        angles = xr.broadcast(*angles)
        phi, theta = uvw2phitheta(
            *getattr(conversions, f"{data.coordinate_frame}2uvw")(*angles, **convert_kwargs)
        )

        # All transforms start from thetaphi and go to the new basis
        # Just need to transpose the matrix to get thetaphi
        if basis != "thetaphi":
            # Find polarization transform
            basis2thetaphi = globals()[f"thetaphi2{basis}"]
            A = basis2thetaphi(phi=phi, theta=theta)
            # Transpose names
            A = A.rename(dict(new_polarization="polarization", polarization="new_polarization"))
            # Project basis to thetaphi
            data = (data * A).sum(dim="polarization")
            data = data.rename(dict(new_polarization="polarization"))

        # Map thetaphi to all the polarizations by first building up the transform matrix A
        A = xr.concat(
            [f(phi=phi, theta=theta) for f in polarization_transforms], dim="new_polarization"
        )

        # Get the apolar data, have to remove quantity and add back in
        if isinstance(data.data, Quantity):
            apolar = data.copy()
            apolar.data = apolar.values
            apolar = vector_norm(apolar, "polarization")
            apolar = apolar.assign_coords(dict(polarization="apolar"))
            apolar.data = apolar.data * data.data.units
        else:
            apolar = vector_norm(data, "polarization")
            apolar = apolar.assign_coords(dict(polarization="apolar"))

        # Project data to all polarizations
        newdata = (data * A).sum(dim="polarization")
        newdata = newdata.rename(dict(new_polarization="polarization"))

        # Append base polarization, new_polarization, apolar
        newdata = xr.concat([newdata, data, apolar], dim="polarization")

        # Add back the attributes
        newdata.attrs = attrs
    else:
        newdata = data
    return newdata


def rotate_polarization(data, uvw_request, rprod):
    """
    Performs polarization rotation of data

    References:
    ----------
    [1] T. Milligan, "More applications of Euler rotation angles," in IEEE Antennas and Propagation Magazine,
    vol. 41, no. 4, pp. 78-83, Aug. 1999, doi: 10.1109/74.789738.
    """
    # TODO: check round-trip (may be issue at pole and may need to shift data slightly)
    # Polarization rotation - only perform if data is not apolar and a different coordinate system
    if data.polarization.shape != (1,):
        # First check if theta phi are present and down-select
        if data.polarization.size > 3 and set(["theta", "phi"]).issubset(
            set(data.polarization.values)
        ):
            data = data.sel(polarization=["theta", "phi"])

        # Validate polarization basis
        basis = "".join(SUPPORTED_POLS[SUPPORTED_POLS_SET.index(set(data.polarization.values))])

        # Transform spatial data into phitheta coordinate frame as that is what is required
        # This corresponds to theta and phi in the requested data's coordinate system
        phi, theta = conversions.uvw2phitheta(*uvw_request)

        # All transforms start from thetaphi and go to the new basis
        # Just need to transpose the matrix to get thetaphi
        if basis != "thetaphi":
            # Find polarization transform
            basis2thetaphi = globals()[f"thetaphi2{basis}"]
            A = basis2thetaphi(phi=phi, theta=theta)
            # Transpose names
            A = A.rename(dict(new_polarization="polarization", polarization="new_polarization"))
            # Project basis to thetaphi
            data = (data * A).sum(dim="polarization")
            data = data.rename(dict(new_polarization="polarization"))

        # A in request domain and name the dimension
        A = thetaphi2xyz(phi=phi, theta=theta)
        A = A.rename(dict(new_polarization="cart_polarization"))
        A = A.rename(dict(polarization="new_polarization"))

        # Get cartesian polarization transform
        # Rotate uvw points
        uvw_request_xr = []
        for da, coord in zip(uvw_request, ["x", "y", "z"]):
            da = da.assign_coords(dict(position=coord))
            uvw_request_xr.append(da)
        uvw_request_xr = xr.concat(uvw_request_xr, dim="position")
        # Rotate uvw points
        if isinstance(rprod, Rotation):
            uvw_prime = apply_rotation(rprod, uvw_request_xr)
        else:
            uvw_prime = rprod.apply(uvw_request_xr)
        # # Format back to tuple
        uvw = []
        for coord in ["x", "y", "z"]:
            da = uvw_prime.sel(position=coord)
            da = da.drop_vars("position")
            uvw.append(da)

        # Create xr.DataArrays with original uvws as dims/coords/attrs
        phi_prime, theta_prime = conversions.uvw2phitheta(*uvw)
        Aprime = thetaphi2xyz(phi=phi_prime, theta=theta_prime)

        # Rename dimensions for rotation
        Aprime = Aprime.rename(dict(new_polarization="position"))
        # Rotate
        if isinstance(rprod, Rotation):
            Aprime = apply_rotation(rprod, Aprime, rotation_dim="position", inverse=True)
        else:
            Aprime = rprod.apply(Aprime, inverse=True)
        # Rename back for projection
        Aprime = Aprime.rename(dict(position="cart_polarization"))
        # Project the unit vectors onto the rotated unit vectors (Does the order matter?)
        A = (A * Aprime).sum(dim="cart_polarization")
        # Remap source polarization to the new polarization
        data = (data * A).sum(dim="polarization")
        data = data.rename(dict(new_polarization="polarization"))

    return data
