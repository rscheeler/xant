from typing import Optional

import numpy as np
import xarray as xr
from loguru import logger
from pint import Quantity
from scipy.spatial.transform import Rotation
from xrench.xrutils import apply_rotation, vector_norm

from ..utils import conversions
from ..utils.conversions import uvw2phitheta

SUPPORTED_POLS = [
    ["apolar"],
    ["theta", "phi"],
    ["x", "y", "z"],
    ["rhcp", "lhcp"],
    ["l3x", "l3y"],
    ["az_azel", "el_azel"],
    ["az_elaz", "el_elaz"],
    ["p45", "m45"],
]
SUPPORTED_POLS_SET = [set(p) for p in SUPPORTED_POLS]


def getpolbasis(data: xr.DataArray) -> str:
    """Determine polarization basis of data."""
    return "".join(
        SUPPORTED_POLS[SUPPORTED_POLS_SET.index(set(data.polarization.values))],
    )


def thetaphi2xyz(phi=None, theta=None):
    r"""
    Generate Jones matrix to convert theta phi polarization in phitheta coordinate frame to cartesian
    x, y, z polarizations.

    .. math::
        E_x &= E_{\\theta} \\cos\\theta\\cos\\phi - E_{\\phi} \\sin\\phi

        E_y &= E_{\\theta} \\cos\\theta\\sin\\phi + E_{\\phi} \\cos\\phi

        E_z &= -E_{\\theta} \\sin\\theta
    """
    phm, thm = xr.broadcast(phi, theta)
    A = np.array(
        [
            [np.cos(thm) * np.cos(phm), -np.sin(phm)],
            [np.cos(thm) * np.sin(phm), np.cos(phm)],
            [-np.sin(thm), xr.zeros_like(phm)],
        ],
    )

    dims = ["new_polarization", "polarization"] + list(phm.dims)
    coords = {**dict(new_polarization=["x", "y", "z"], polarization=["theta", "phi"]), **phm.coords}
    A = xr.DataArray(A, dims=dims, coords=coords)

    return A


def xyz2thetaphi(phi=None, theta=None):
    r"""
    Generate Jones matrix to convert theta phi polarization in phitheta coordinate frame to cartesian
    x, y, z polarizations.

    .. math::
        E_{\\theta} &= E_x \\cos\\theta\\cos\\phi + E_y \\cos\\theta\\sin\\phi - E_z \\sin\\theta

        E_{\\phi} &= -E_x \\sin\\phi + E_y \\cos\\phi

    """
    phm, thm = xr.broadcast(phi, theta)
    A = np.array(
        [
            [np.cos(thm) * np.cos(phm), np.cos(thm) * np.sin(phm), -np.sin(thm)],
            [-np.sin(phm), np.cos(phm), xr.zeros_like(phm)],
        ],
    )

    dims = ["new_polarization", "polarization"] + list(phm.dims)
    coords = {**dict(new_polarization=["theta", "phi"], polarization=["x", "y", "z"]), **phm.coords}
    return xr.DataArray(A, dims=dims, coords=coords)


def thetaphi2az_azelel_azel(phi=None, theta=None):
    r"""
    Returns jones matrix to convert theta, phi polarization in phitheta coordinate frame to Ludwig's II definition
    of co-pol aligned with azimuth-axis(az_azel) and cross-pol aligned with the elevation-axis (el_azel) in an
    azimuth over elevation coordinate frame.

    .. math::
        E_{az} &= \\frac{1}{\\cos El}(E_{\\theta} \\cos\\phi - E_{\\phi} \\cos\\theta\\sin\\phi)

        E_{el} &= \\frac{1}{\\cos El}(E_{\\theta} \\cos\theta\\sin\\phi + E_{\\phi} \\cos\\phi)

        \\cos El = \\sqrt{1-\\sin^2\\theta\\sin^2\\phi}
    """
    phm, thm = xr.broadcast(phi, theta)
    cosEl = np.sqrt(1 - np.sin(thm) ** 2 * np.sin(phm) ** 2)
    scale = xr.where(cosEl == 0, 0, 1 / cosEl)
    A = np.array(
        [[np.cos(phm), -np.cos(thm) * np.sin(phm)], [np.cos(thm) * np.sin(phm), np.cos(phm)]],
    )

    dims = ["new_polarization", "polarization"] + list(phm.dims)
    coords = {
        **dict(new_polarization=["az_azel", "el_azel"], polarization=["theta", "phi"]),
        **phm.coords,
    }
    A = xr.DataArray(A, dims=dims, coords=coords)
    A *= scale
    return A


def thetaphi2az_elazel_elaz(phi=None, theta=None):
    r"""
    Returns jones matrix to convert theta, phi polarization in phitheta coordinate frame to Ludwig's II definition
    of co-pol aligned with azimuth-axis(az_elaz) and cross-pol aligned with the elevation-axis (el_elaz) in an
    elevation over azimuth coordinate frame.

    .. math::
        E_{az} &= \\frac{1}{\\cos \\alpha}(E_{\\theta} \\cos\\theta\\cos\\phi - E_{\\phi} \\sin\\phi)

        E_{el} &= \\frac{1}{\\cos \\alpha}(E_{\\theta} \\sin\\phi + E_{\\phi} \\cos\\theta\\cos\\phi)

        \\cos \\alpha = \\sqrt{1-\\sin^2\\theta\\cos^2\\phi}
    """
    phm, thm = xr.broadcast(phi, theta)
    cosalpha = np.sqrt(1 - np.sin(thm) ** 2 * np.cos(phm) ** 2)
    scale = xr.where(cosalpha == 0, 0, 1 / cosalpha)
    A = np.array(
        [[np.cos(thm) * np.cos(phm), -np.sin(phm)], [np.sin(phm), np.cos(thm) * np.cos(phm)]],
    )

    dims = ["new_polarization", "polarization"] + list(phm.dims)
    coords = {
        **dict(new_polarization=["az_elaz", "el_elaz"], polarization=["theta", "phi"]),
        **phm.coords,
    }
    A = xr.DataArray(A, dims=dims, coords=coords)
    A *= scale
    return A


def thetaphi2l3xl3y(phi=None, theta=None):
    r"""
    Returns jones matrix to convert theta, phi polarization in phitheta coordinate frame to Ludwig's III definition
    of co-pol aligned with the x-axis (l3x) and cross-pol aligned with the y-axis (l3y).

    .. math::
        E_{l3x} &= E_{\\theta} \\cos\\phi - E_{\\phi} \\sin\\phi

        E_{l3y} &= E_{\\theta} \\sin\\phi + E_{\\phi} \\cos\\phi
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
    r"""
    Returns Jones matrix to convert theta, phi polarization to circular RHCP and LHCP.

    .. math::
        E_{RHCP} &= \\frac{E_{\\theta} + j E_{\\phi}}{\\sqrt{2}}

        E_{LHCP} &= \\frac{E_{\\theta} - j E_{\\phi}}{\\sqrt{2}}
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
    r"""
    Returns Jones matrix to convert theta, phi polarization to slant linear +/- 45.

    .. math::
        E_{+45} &= \\frac{E_{\\theta} + E_{\\phi}}{\\sqrt{2}}

        E_{-45} &= \\frac{E_{\\theta} - E_{\\phi}}{\\sqrt{2}}
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


polarization_transforms = [
    thetaphi2xyz,
    thetaphi2l3xl3y,
    thetaphi2rhcplhcp,
    thetaphi2p45m45,
    thetaphi2az_azelel_azel,
    thetaphi2az_elazel_elaz,
]


def tothetaphi(
    data: xr.DataArray,
    basis: str,
    phi: xr.DataArray,
    theta: xr.DataArray,
) -> xr.DataArray:
    """Convert data from basis polarization to theta/phi polarization."""
    # 1. Find forward polarization transform (e.g., input: ['theta', 'phi'], new: ['x', 'y', 'z'])
    basis2thetaphi = globals()[f"thetaphi2{basis}"]
    A_forward = basis2thetaphi(phi=phi, theta=theta)

    # 2. Compute pseudo-inverse. We use temporary dimension names for the output core
    # so xarray doesn't accidentally bind the old coordinates to the new axis sizes.
    A_inv = xr.apply_ufunc(
        np.linalg.pinv,
        A_forward,
        input_core_dims=[["new_polarization", "polarization"]],
        output_core_dims=[["target_pol", "source_pol"]],
        vectorize=True,
    )

    # 3. Safely map the correct coordinate string arrays to our temporary dimensions
    A_inv = A_inv.assign_coords(
        {
            "target_pol": A_forward.coords["polarization"].values,  # e.g., ['theta', 'phi']
            "source_pol": A_forward.coords["new_polarization"].values,  # e.g., ['x', 'y', 'z']
        },
    )

    # 4. Rename the temporary dimensions to match your data array for multiplication
    A_inv = A_inv.rename({"source_pol": "polarization", "target_pol": "new_polarization"})

    # 5. Project basis back to thetaphi
    # This automatically matches up 'polarization' on both arrays regardless of where axis order is,
    # sums it out, and renames the remaining 'new_polarization' back to 'polarization'
    return (data * A_inv).sum(dim="polarization").rename(dict(new_polarization="polarization"))


def project_polarizations(
    data,
    convert_kwargs: dict | None = None,
    pol_converters: list[callable] = polarization_transforms,
):
    """Takes data in and generates all the polarizations by projecting the input basis to other basis."""
    # Only project if not apolar or if all polarizations are present
    if data.polarization.shape != (1,) and set(data.polarization.values) != {
        item for subl in SUPPORTED_POLS for item in subl
    }:
        # Default convert kwargs
        if convert_kwargs is None:
            convert_kwargs = {}

        # Store attrs
        attrs = data.attrs

        # Validate polarization basis
        basis = getpolbasis(data)

        # Transform spatial data into phitheta coordinate frame as that is what is required
        angles = [data.coords[a] for a in conversions.COORDINATE_DIMS[data.coordinate_frame]]
        # Broadcast angles as this is necessary for some conversions
        angles = xr.broadcast(*angles)
        # Only convert if needed
        if data.coordinate_frame != "phitheta":
            phi, theta = uvw2phitheta(
                *getattr(conversions, f"{data.coordinate_frame}2uvw")(*angles, **convert_kwargs),
            )
        else:
            phi, theta = angles

        # All transforms start from thetaphi and go to the new basis
        if basis != "thetaphi":
            data = tothetaphi(data, basis, phi, theta)

        # Map thetaphi to all the polarizations by first building up the transform matrix A
        A = xr.concat(
            [f(phi=phi, theta=theta) for f in pol_converters],
            dim="new_polarization",
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
            set(data.polarization.values),
        ):
            data = data.sel(polarization=["theta", "phi"])

        # Validate polarization basis
        basis = getpolbasis(data)

        if basis == "xyz":
            # Since basis is x,y,z just rotate the identity matrix
            A = xr.DataArray(
                np.eye(3),
                dims=["position", "polarization"],
                coords=dict(position=["x", "y", "z"], polarization=["x", "y", "z"]),
            )

            if isinstance(rprod, Rotation):
                A = apply_rotation(rprod, A, rotation_dim="position", inverse=False)
            else:
                A = rprod.apply(A, inverse=False)
            A = A.rename(dict(position="new_polarization"))
        else:
            # Transform spatial data into phitheta coordinate frame as that is what is required
            # This corresponds to theta and phi in the requested data's coordinate system
            phi, theta = conversions.uvw2phitheta(*uvw_request)

            # All transforms start from thetaphi and go to the new basis
            if basis != "thetaphi":
                data = tothetaphi(data, basis, phi, theta)

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
                uvw_prime = apply_rotation(rprod, uvw_request_xr, inverse=True)
            else:
                uvw_prime = rprod.apply(uvw_request_xr, inverse=True)
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
