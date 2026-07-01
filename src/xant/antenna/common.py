from typing import Optional

import numpy as np
import xarray as xr
from hics import HCS
from pint import Quantity
from scipy.spatial.transform import Rotation
from scipy.special import fresnel, jv, sici
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
            polarization=["theta", "phi"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")
        # Initialize directivity
        self._dipole_d0 = self.get_dipole_d0(l, frequency)
        super().__init__(antenna_function, hcs)

    @classmethod
    def get_dipole_d0(cls, l, frequency) -> float:
        """
        Calculate the peak directivity using Balanis formula.

        References:
        ----------
        C. A. Balanis, Antenna theory: analysis and design, 3. ed. Hoboken, N.J: Wiley-Interscience, 2005.
        """
        # kl
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam
        kL = k * l
        kL.data = kL.data.to_base_units().magnitude

        # Euler's Constant (often gamma, Balanis uses C)
        C = 0.5772156649015328

        # Compute Si and Ci values
        Si_kL, Ci_kL = sici(kL)
        Si_2kL, Ci_2kL = sici(2 * kL)

        # Variable Q for computing directivity (4-75a)
        Q = (
            C
            + np.log(kL)
            - Ci_kL
            + 0.5 * np.sin(kL) * (Si_2kL - 2 * Si_kL)
            + 0.5 * np.cos(kL) * (C + np.log(kL / 2) + Ci_2kL - 2 * Ci_kL)
        )
        # Radiation intensity, F
        theta = np.array([np.linspace(0.001, np.pi - 0.001, 1000)]).T
        numerator = np.cos(np.outer(kL / 2, np.cos(theta))) - np.outer(
            np.cos(kL / 2),
            np.ones_like(theta),
        )
        denominator = np.outer(np.ones_like(kL), np.sin(theta))
        F_theta = (numerator / denominator) ** 2

        # Maximum radiation intensity, Fmax
        F_max = np.max(F_theta, axis=-1)

        # Calculate peak directivity D0

        return (2.0 * F_max) / Q

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """Radiation pattern of the dipole."""
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # Normalized Etheta
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

        # Scale by directivity
        etot = etot * np.sqrt(self._dipole_d0)

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
            polarization=["theta", "phi"],
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
            polarization=["theta", "phi"],
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
            polarization=["theta", "phi"],
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


class PyramidalHorn(Antenna):
    """
    Pyramidal horn antenna pattern. The antenna pattern is derived from the formulas on pages 769-779 of
    Antenna Theory by Balanis 3rd ed.
    """

    def __init__(
        self,
        a: Quantity,
        b: Quantity,
        a1: Quantity,
        b1: Quantity,
        p1: Quantity,
        p2: Quantity,
        frequency: Quantity,
        hcs: HCS | None = None,
    ):
        # Set input properties
        self.a = a
        self.b = b
        self.a1 = a1
        self.b1 = b1
        self.p1 = p1
        self.p2 = p2

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
            polarization=["theta", "phi"],
            frequency=frequency,
            phi=phi,
            theta=theta,
        )
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, hcs)

    def _antenna_func(self, frequency=None, phi=None, theta=None, **kwargs):
        """Radiation pattern of the pyramidal horn."""
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to("m")
        k = (2 * np.pi * ureg.radians) / lam

        # Intermediate equations
        kxp = k * np.sin(theta) * np.cos(phi) + np.pi * ureg.radians / self.a1
        kxpp = k * np.sin(theta) * np.cos(phi) - np.pi * ureg.radians / self.a1
        t1p = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (-k * self.a1 / 2 - kxp * self.p2)
        t2p = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (k * self.a1 / 2 - kxp * self.p2)
        t1pp = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (
            -k * self.a1 / 2 - kxpp * self.p2
        )
        t2pp = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (
            k * self.a1 / 2 - kxpp * self.p2
        )
        t1p.data = t1p.data.to_base_units().magnitude
        t2p.data = t2p.data.to_base_units().magnitude
        t1pp.data = t1pp.data.to_base_units().magnitude
        t2pp.data = t2pp.data.to_base_units().magnitude
        St1p, Ct1p = fresnel(t1p)
        St2p, Ct2p = fresnel(t2p)
        St1pp, Ct1pp = fresnel(t1pp)
        St2pp, Ct2pp = fresnel(t2pp)

        I1 = (
            0.5
            * np.sqrt(np.pi * ureg.radians * self.p2 / k)
            * (
                np.exp(
                    1j * kxp**2 * self.p2 / (2 * k),
                )
                * ((Ct2p - Ct1p) - 1j * (St2p - St1p))
                + np.exp(1j * kxpp**2 * self.p2 / (2 * k))
                * ((Ct2pp - Ct1pp) - 1j * (St2pp - St1pp))
            )
        )  # (13-46)

        ky = k * np.sin(theta) * np.sin(phi)
        t1 = np.sqrt(1 / (np.pi * ureg.radians * k * self.p1)) * (-k * self.b1 / 2 - ky * self.p1)
        t2 = np.sqrt(1 / (np.pi * ureg.radians * k * self.p1)) * (k * self.b1 / 2 - ky * self.p1)
        t1.data = t1.data.to_base_units().magnitude
        t2.data = t2.data.to_base_units().magnitude
        St1, Ct1 = fresnel(t1)
        St2, Ct2 = fresnel(t2)

        I2 = (
            np.sqrt(np.pi * ureg.radians * self.p1 / k)
            * np.exp(1j * ky**2 * self.p1 / (2 * k))
            * ((Ct2 - Ct1) - 1j * (St2 - St1))
        )  # (13-47)

        # Ntheta = np.cos(theta) * np.sin(phi) * I1 * I2  # (13-45a)
        # Nphi = -np.cos(phi) * I1 * I2  # (13-45b)
        # Ltheta = np.cos(theta) * np.cos(phi) * I1 * I2  # (13-45c)
        # Lphi = -np.sin(phi) * I1 * I2  # (13-45d)

        etheta = (
            1j
            * k
            / (4 * np.pi * ureg.radian)
            * (1 / (1 * ureg.m))
            * (np.sin(phi) * (np.cos(theta) + 1) * I1 * I2)
        )  # (13-48b)
        ephi = (
            1j
            * k
            / (4 * np.pi * ureg.radian)
            * (1 / (1 * ureg.m))
            * (np.cos(phi) * (np.cos(theta) + 1) * I1 * I2)
        )  # (13-48c)
        etheta.data = etheta.data.to_base_units()
        ephi.data = ephi.data.to_base_units()

        # Add polarization coordinate
        etheta = etheta.assign_coords({"polarization": "theta"})
        ephi = ephi.assign_coords({"polarization": "phi"})

        # Directivity Scale (13-50c), (13-51), (13-52)
        scale = 8 * np.pi / (self.a1 * self.b1).to_base_units().magnitude

        # Compute Dp
        # u = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (
        #     -k * self.a1 / 2 + np.pi * ureg.radian * self.p2 / self.a1
        # )
        # v = np.sqrt(1 / (np.pi * ureg.radians * k * self.p2)) * (
        #     k * self.a1 / 2 + np.pi * ureg.radian * self.p2 / self.a1
        # )
        # u.data = u.data.to_base_units().magnitude
        # v.data = v.data.to_base_units().magnitude
        # Su, Cu = fresnel(u)
        # Sv, Cv = fresnel(v)

        # dkern = self.b1 / np.sqrt(2 * lam * self.p1)
        # dkern.data = dkern.data.to_base_units().magnitude
        # Sdkern, Cdkern = fresnel(dkern)
        # Dp = (
        #     8
        #     * np.pi
        #     * self.p1
        #     * self.p2
        #     / (self.a1 * self.b1)
        #     * ((Cu - Cv) ** 2 + (Su - Sv) ** 2)
        #     * (Cdkern**2 + Sdkern**2)
        # ) # Note: This directivity is lower than the integrated directivity computed from the far-field.

        # Total pattern
        return xr.concat((etheta, ephi), dim="polarization") * np.sqrt(scale)
