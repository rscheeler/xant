from typing import Optional

import numpy as np
import xarray as xr
from hics import HCS
from pint import Quantity
from scipy.spatial.transform import Rotation
from scipy.special import jv
from xrench.units import ureg

from .core import Antenna, AntennaFunction
from .phasedarray import AntennaArray


class Dipole(Antenna):
    """Dipole antenna pattern."""

    def __init__(
        self,
        l: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
    ):
        # Set input properties
        self.l = l

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(
            polarization=["thetapol", "phipol"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Radiation pattern of the dipole.
        """
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # TODO: scale based on directivity
        etheta = (np.cos(k * self.l / 2 * np.cos(theta)) - np.cos(k * self.l / 2)) / np.sin(theta)
        etheta = etheta * xr.ones_like(phi)

        # Phi pol is zero
        ephi = xr.zeros_like(etheta)

        # Add polarization coordinate
        etheta = etheta.assign_coords(dict(polarization="theta"))
        ephi = ephi.assign_coords(dict(polarization="phi"))

        # Total pattern
        etot = xr.concat((etheta, ephi), dim="polarization")

        # Set nans to 0
        etot = etot.fillna(0)

        return etot


class Isotropic(Antenna):
    """Isotropic antenna pattern."""

    def __init__(
        self,
        frequency: Quantity,
        hcs: HCS | None = None,
    ):
        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Isotropic radiation pattern (ones in shape of dimensions).
        """
        data = xr.ones_like(frequency * phi * theta) * ureg.dimensionless
        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")
        return data


class Hemispherical(Antenna):
    """Hemispherical antenna pattern."""

    def __init__(
        self,
        frequency: Quantity,
        hcs: HCS | None = None,
    ):
        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Hemispherical radiation pattern (ones for theta <= np.pi/2 zero theta > np.pi/2).
        """
        data = xr.ones_like(frequency * phi * theta)
        dt = data * theta

        # Set lower hemisphere to zero
        data.data[np.where(abs(dt) > np.pi / 2)] = 0 * data.data[np.where(abs(dt) > np.pi / 2)]

        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")
        return data


class Cardioid(Antenna):
    """Cardioid antenna pattern."""

    def __init__(
        self,
        frequency: Quantity,
        hcs: HCS | None = None,
    ):
        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Cardioid radiation pattern.
        """
        # Cardioid pattern
        data = (1 + np.cos(xr.ones_like(frequency * phi * theta) * theta)) / 2

        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")
        return data


class Cosine(Antenna):
    """Cardioid antenna pattern."""

    def __init__(
        self,
        frequency: Quantity,
        n: float = 1,
        hcs: HCS | None = None,
    ):
        # Set input properties
        self.n = n

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """Cardioid radiation pattern."""
        # Cardioid pattern
        data = np.cos(xr.ones_like(frequency * phi * theta) * theta) ** (self.n / 2)

        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")
        return data


class ElementWeight(Antenna):
    """
    Weighted element pattern that provides a sloped pattern as a function of theta.
    """

    def __init__(
        self,
        frequency: Quantity,
        scale: Quantity = 90 * ureg.degree,
        slope: float = -12.0,
        hcs: HCS | None = None,
    ):
        # Store slope and scale
        self.slope = slope
        self.scale = scale

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, theta=None, **kwargs):
        """Weighted element pattern."""
        # Cardioid pattern
        data = 10 ** (self.slope * (theta / (self.scale)) ** 2 / 20)

        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")
        return data


class DipoleAboveGround(Antenna):
    def __init__(
        self,
        l: Quantity,
        h: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
        orientation: str | None = "x",
        cropping_antenna=Cardioid,
    ):
        # Create coordinate systems based on orientation
        if orientation.lower() == "x":
            arrcs = [
                HCS(
                    (0, 0, h.magnitude) * h.units,
                    rotation=Rotation.from_euler("ZYZ", [0, 90, 0], degrees=True),
                    reference=hcs,
                ),
                HCS(
                    (0, 0, -h.magnitude) * h.units,
                    rotation=Rotation.from_euler("ZYZ", [0, -90, 0], degrees=True),
                    reference=hcs,
                ),
            ]
        elif orientation.lower() == "y":
            arrcs = [
                HCS(
                    (0, 0, h.magnitude) * h.units,
                    rotation=Rotation.from_euler("ZYZ", [90, 90, 0], degrees=True),
                    reference=hcs,
                ),
                HCS(
                    (0, 0, -h.magnitude) * h.units,
                    rotation=Rotation.from_euler("ZYZ", [90, -90, 0], degrees=True),
                    reference=hcs,
                ),
            ]
        elif orientation.lower() == "z":
            arrcs = [
                HCS(
                    (0, 0, h.magnitude) * h.units,
                    reference=hcs,
                ),
                HCS(
                    (0, 0, -h.magnitude) * h.units,
                    reference=hcs,
                ),
            ]
        else:
            raise ValueError(f"Orientation {orientation} not valid.")

        # Create Dipole Element
        element = Dipole(l, frequency, hcs)

        # Create Array
        self._arr = AntennaArray(element, arrcs)

        # Make the pattern by taking the total and multiplying by a hemispherical antenna
        self._antenna = self._arr.total * cropping_antenna(frequency, hcs=hcs)

        # Make the function
        antenna_function = AntennaFunction(
            element.data.dims,
            element.data.coords,
            self._antenna_func,
            "phitheta",
        )

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, *args, **kwargs):
        """Radiation pattern of the dipole."""
        # Just get total pattern
        return self._antenna.request_data(*args, **kwargs)


class RectangularAperture(Antenna):
    """
    Rectangular aperture antenna pattern. The antenna pattern is derived from the formulas on page
    672-673 of Antenna Theory by Balanis 3rd ed.
    """

    def __init__(
        self,
        a: Quantity,
        b: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
        tau: Quantity = 0 * ureg.degree,
        truncate_lower_hemisphere=True,
    ):
        # Set input properties
        self.a = a
        self.b = b
        self.tau = tau
        self.truncate_lower_hemisphere = truncate_lower_hemisphere

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(
            polarization=["thetapol", "phipol"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """Radiation pattern of the rectangular aperture."""
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # Intermediate variables for far-zone fields
        X = (k * self.a / 2) * np.sin(theta) * np.cos(phi)
        Y = (k * self.b / 2) * np.sin(theta) * np.sin(phi)

        # Need to convert to base units before passing to sinc function
        # Remove quantity so xr.apply_ufunc works
        X.data = X.data.to_base_units().magnitude
        Y.data = Y.data.to_base_units().magnitude

        # Electric field components
        etheta = (
            np.sin(phi) * xr.apply_ufunc(np.sinc, X / np.pi) * xr.apply_ufunc(np.sinc, Y / np.pi)
        )
        ephi = (
            np.cos(theta)
            * np.cos(phi)
            * xr.apply_ufunc(np.sinc, X / np.pi)
            * xr.apply_ufunc(np.sinc, Y / np.pi)
        )

        # Set lower hemisphere to zero
        if self.truncate_lower_hemisphere:
            dt = xr.ones_like(etheta) * theta
            etheta.data[np.where(abs(dt) > np.pi / 2)] = (
                0 * etheta.data[np.where(abs(dt) > np.pi / 2)]
            )
            ephi.data[np.where(abs(dt) > np.pi / 2)] = 0 * ephi.data[np.where(abs(dt) > np.pi / 2)]

        # Add polarization coordinate
        etheta = etheta.assign_coords(dict(polarization="theta"))
        ephi = ephi.assign_coords(dict(polarization="phi"))

        # Total pattern
        etot = xr.concat((etheta, ephi), dim="polarization")

        # Polarization Transform
        A = xr.DataArray(
            np.array([[np.cos(self.tau), -np.sin(self.tau)], [np.sin(self.tau), np.cos(self.tau)]]),
            dims=("polarization", "new_polarization"),
            coords=dict(polarization=["theta", "phi"], new_polarization=["theta", "phi"]),
        )

        # Project data to new polarization
        etot = (etot * A).sum(dim="polarization")
        etot = etot.rename(dict(new_polarization="polarization"))

        # Scale based on areal directivity
        adir = np.sqrt(4 * np.pi * (self.a * self.b) / lam**2)
        adir.data = adir.data.to_base_units().magnitude
        etot *= adir

        return etot


class TE10Aperture(Antenna):
    """
    Rectangular aperture antenna pattern. The antenna pattern is derived from the formulas on page 672-673 of
    Antenna Theory by Balanis 3rd ed.
    """

    def __init__(
        self,
        a: Quantity,
        b: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
        tau: Quantity = 0 * ureg.degree,
        truncate_lower_hemisphere=True,
    ):
        # Set input properties
        self.a = a
        self.b = b
        self.tau = tau
        self.truncate_lower_hemisphere = truncate_lower_hemisphere

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(
            polarization=["thetapol", "phipol"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Radiation pattern of the rectangular aperture.
        """
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # Intermediate variables for far-zone fields
        X = (k * self.a / 2) * np.sin(theta) * np.cos(phi)
        Y = (k * self.b / 2) * np.sin(theta) * np.sin(phi)

        # Need to convert to base units before passing to sinc function
        # Remove quantity so xr.apply_ufunc works
        X.data = X.data.to_base_units().magnitude
        Y.data = Y.data.to_base_units().magnitude

        # Electric field components
        etheta = (
            np.sin(phi) * np.cos(X) / (X**2 - (np.pi / 2) ** 2) * xr.apply_ufunc(np.sinc, Y / np.pi)
        )
        ephi = (
            np.cos(theta)
            * np.cos(phi)
            * np.cos(X)
            / (X**2 - (np.pi / 2) ** 2)
            * xr.apply_ufunc(np.sinc, Y / np.pi)
        )
        max_field = 1 / (np.pi / 2) ** 2

        # Set lower hemisphere to zero
        if self.truncate_lower_hemisphere:
            dt = xr.ones_like(etheta) * theta
            etheta.data[np.where(abs(dt) > np.pi / 2)] = (
                0 * etheta.data[np.where(abs(dt) > np.pi / 2)]
            )
            ephi.data[np.where(abs(dt) > np.pi / 2)] = 0 * ephi.data[np.where(abs(dt) > np.pi / 2)]

        # Add polarization coordinate
        etheta = etheta.assign_coords(dict(polarization="theta"))
        ephi = ephi.assign_coords(dict(polarization="phi"))

        # Total pattern
        etot = xr.concat((etheta, ephi), dim="polarization")

        # Polarization Transform
        A = xr.DataArray(
            np.array([[np.cos(self.tau), -np.sin(self.tau)], [np.sin(self.tau), np.cos(self.tau)]]),
            dims=("polarization", "new_polarization"),
            coords=dict(polarization=["theta", "phi"], new_polarization=["theta", "phi"]),
        )

        # Project data to new polarization
        etot = (etot * A).sum(dim="polarization")
        etot = etot.rename(dict(new_polarization="polarization"))

        # Scale based on areal directivity
        adir = np.sqrt(4 * np.pi * (self.a * self.b) / lam**2)
        adir.data = adir.data.to_base_units().magnitude
        etot *= adir / max_field

        return etot


class CircularAperture(Antenna):
    """
    Circular aperture antenna pattern. The antenna pattern is derived from the formulas on page 688-689 of
    Antenna Theory by Balanis 3rd ed.
    """

    def __init__(
        self,
        r: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
        truncate_lower_hemisphere=True,
    ):
        # Set input properties
        self.r = r
        self.truncate_lower_hemisphere = truncate_lower_hemisphere

        # Default Coordinates
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency.to_base_units()),
            attrs=dict(units=frequency.units),
        )
        theta = xr.DataArray(
            np.linspace(0, 180, 181) * ureg.degree,
            dims=("theta",),
            coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        phi = xr.DataArray(
            np.arange(-180, 180, 1) * ureg.degree,
            dims=("phi",),
            coords=dict(phi=(np.arange(-180, 180, 1) * ureg.degree).to_base_units()),
            attrs=dict(units=ureg.degree),
        )
        dims = ("polarization", "frequency", "phi", "theta")
        coords = dict(
            polarization=["thetapol", "phipol"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """
        Radiation pattern of the circular aperture.
        """
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # Intermediate variables for far-zone fields
        Z = (k * self.r) * np.sin(theta)

        # Need to convert to base units before passing to sinc function
        # Remove quantity so xr.apply_ufunc works
        Z.data = Z.data.to_base_units().magnitude

        # Electric field components
        max_field = 0.5
        bessel = xr.apply_ufunc(jv, 1, Z) / Z
        # If Z is effectively 0, use the limit (0.5), otherwise calculate J1(Z)/Z
        bessel = xr.where(Z == 0, max_field, bessel)
        etheta = np.sin(phi) * bessel
        ephi = np.cos(theta) * np.cos(phi) * bessel

        # Set lower hemisphere to zero
        if self.truncate_lower_hemisphere:
            dt = xr.ones_like(etheta) * theta
            etheta.data[np.where(abs(dt) > np.pi / 2)] = (
                0 * etheta.data[np.where(abs(dt) > np.pi / 2)]
            )
            ephi.data[np.where(abs(dt) > np.pi / 2)] = 0 * ephi.data[np.where(abs(dt) > np.pi / 2)]

        # Add polarization coordinate
        etheta = etheta.assign_coords(dict(polarization="theta"))
        ephi = ephi.assign_coords(dict(polarization="phi"))

        # Total pattern
        etot = xr.concat((etheta, ephi), dim="polarization")

        # Scale based on areal directivity
        adir = np.sqrt(4 * np.pi * np.pi * self.r**2 / lam**2)
        adir.data = adir.data.to_base_units().magnitude
        etot *= adir / max_field

        return etot
