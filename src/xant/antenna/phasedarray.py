from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hics import HCS
from hics.plotting import viewcs
from loguru import logger
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from pint import Quantity
from scipy.spatial.transform import Rotation
from xrench.units import ureg
from xrench.xrutils import apply_rotation, vector_norm

from ..utils import conversions
from .core import Antenna, AntennaFunction, concat


class AntennaArray:
    """
    Class for representing an array of antennas. Can be made up of either Antenna or
    AntennaArray elements. The array factor is calculated based on the positions of
    the elements and their excitations. The total pattern is calculated based on the
    element patterns and their excitations.

    Parameters
    ----------
    element : Antenna | AntennaArray
        The antenna or antenna array element.
    coordinate_systems : list[HCS]
        The coordinate systems for each element.
    broadcast_elements : bool, optional
        Whether to broadcast the elements, by default False.
    steering_method : str, optional
        The steering method to use, by default None.
    """

    def __init__(
        self,
        element: Antenna | AntennaArray,
        coordinate_systems: list[HCS],
        broadcast_elements: bool = False,
        steering_method: str = None,
    ) -> None:
        # Capture frequencies
        if isinstance(element, Antenna):
            self.frequency = element.data.frequency
        elif isinstance(element, AntennaArray):
            self.frequency = element.frequency
        elif isinstance(element, np.ndarray):
            if isinstance(element[0], Antenna):
                self.frequency = element[0].data.frequency
            elif isinstance(element[0], AntennaArray):
                self.frequency = element[0].frequency
        # Make Port DataArray
        # Store element and coordinate systems
        port = xr.DataArray(
            np.arange(len(coordinate_systems)),
            dims=("port",),
            coords=dict(port=np.arange(len(coordinate_systems))),
        )
        # Check to see if elements need to be broadcasted
        rot_bools = []
        iden_rot = Rotation.identity().as_quat()
        for x in [cs.rotation for cs in coordinate_systems]:
            rot_bools.append(
                all(np.atleast_1d((x.basemag == iden_rot).all(dim="quaternion"))),
            )
        if not all(rot_bools):
            # Broadcast element based on rotation and remove rotation from element coordinate systems
            elements = []
            for cs in coordinate_systems:
                # Make a copy of the element (either AntennaArray or Antenna)
                el = element.copy()
                # Change coordinate system for antenna or phased array
                if isinstance(el, Antenna):
                    el.move(cs)
                elif isinstance(el, AntennaArray):
                    # Change element
                    if isinstance(el.element, Antenna):
                        el.element.move(cs)
                    # Change each position's reference
                    for elcs in el.coordinate_systems:
                        elcs.reference = cs
                elements.append(el)

            # Make into an array
            element = np.array(elements)
            if isinstance(el, Antenna):
                element = concat(element, dict(port=port))

        elif broadcast_elements:
            els = []
            for cs in coordinate_systems:
                el = element.copy()
                el.move(cs)
                els.append(el)
            element = np.array(els)
            if isinstance(els[0], Antenna):
                element = concat(element, dict(port=port))
        # Store element and coordinate systems
        self.element = element
        self.coordinate_systems = coordinate_systems

        # Initialize Variables
        self._excitation = xr.ones_like(port).astype(np.complex128)
        self._taper = xr.ones_like(port).astype(np.complex128)
        self._steering_vector = xr.ones_like(port).astype(np.complex128)
        self.initialize()

    def initialize(self):
        self._tpf = None
        self._elements = None
        self._total = None
        self._af = None

    @classmethod
    def rectangular(
        cls,
        element: Antenna | AntennaArray,
        nx: int,
        dx: Quantity,
        ny: int,
        dy: Quantity,
        cs_reference: HCS,
        **kwargs,
    ):
        """
        Classmethod for making a rectangular array.

        Parameters
        ----------
        element : Antenna | AntennaArray
            The antenna or antenna array element.
        nx : int
            Number of elements in the x-direction.
        dx : Quantity
            Spacing between elements in the x-direction.
        ny : int
            Number of elements in the y-direction.
        dy : Quantity
            Spacing between elements in the y-direction.
        cs_reference : HCS
            The reference coordinate system.

        Returns:
        -------
        AntennaArray
            The rectangular array.

        """
        # Create Grid of points
        ix = np.arange(nx).astype(np.float64)
        iy = np.arange(ny).astype(np.float64)
        xs, ys = np.meshgrid(ix, iy)
        xs *= dx
        ys *= dy
        xs = xs.to_base_units()
        ys = ys.to_base_units()

        # Center grid
        xs -= xs.mean()
        ys -= ys.mean()

        coordinate_systems = [
            HCS(
                np.array([ai.magnitude for ai in [xi, yi, 0 * xi]]) * xs.units,
                reference=cs_reference,
            )
            for xi, yi in zip(xs.ravel(), ys.ravel())
        ]

        return cls(element, coordinate_systems, **kwargs)

    @classmethod
    def triangular(
        cls,
        element: Antenna | AntennaArray,
        nx: int,
        ny: int,
        dx: Quantity,
        cs_reference: HCS,
        **kwargs,
    ):
        """
        Classmethod for making a triangular array.

        Parameters
        ----------
        element : Antenna | AntennaArray
            The antenna or antenna array element.
        nx : int
            Number of elements in the x-direction.
        ny : int
            Number of elements in the y-direction.
        dx : Quantity
            Spacing between elements (will be spacing in x-dimension)
        cs_reference : HCS
            The reference coordinate system.

        Returns:
        -------
        AntennaArray
            The triangular array.
        """
        # Create Grid of points
        nx = np.arange(nx).astype(np.float64)
        ny = np.arange(ny).astype(np.float64)
        xs, ys = np.meshgrid(nx, ny)
        xs += np.mod(ys, 2) / 2
        ys *= np.sqrt(3) / 2.0
        xs *= dx
        ys *= dx
        xs = xs.to_base_units()
        ys = ys.to_base_units()

        # Center grid
        xs -= xs.mean()
        ys -= ys.mean()

        coordinate_systems = [
            HCS(
                np.array([ai.magnitude for ai in [xi, yi, 0 * xi]]) * xs.units,
                reference=cs_reference,
            )
            for xi, yi in zip(xs.ravel(), ys.ravel())
        ]

        return cls(element, coordinate_systems, **kwargs)

    @property
    def has_children(self) -> bool:
        """Determines if there are any AntennaArrays in the element."""
        _has_children = False
        if not isinstance(self.element, Antenna):
            if isinstance(self.element, np.ndarray):
                for e in self.element:
                    if isinstance(e, AntennaArray):
                        _has_children = True
                        break
            elif isinstance(self.element, AntennaArray):
                _has_children = True
        return _has_children

    @property
    def excitation(self):
        self._excitation = self.steering_vector * self.taper
        return self._excitation

    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self, value):
        # TODO: some validation
        self._taper = value
        # self.initialize()

    @property
    def steering_vector(self):
        return self._steering_vector

    @steering_vector.setter
    def steering_vector(self, value):
        # TODO: some validation
        self._steering_vector = value
        # self.initialize()

    @property
    def tpf(self) -> TranslatedPhase:
        """Translated phase factor."""
        if self._tpf is None:
            self._tpf = TranslatedPhase(
                self.frequency.data,
                coordinate_systems=self.coordinate_systems,
            )
        return self._tpf

    @property
    def elements(self):
        """Returns translated elements."""
        # if self._elements is None:
        if isinstance(self.element, Antenna):
            self._elements = self.element * self.tpf
        else:
            element = self.element
            if isinstance(element, np.ndarray):
                port = xr.DataArray(
                    np.arange(len(self.coordinate_systems)),
                    dims=("port",),
                    coords=dict(port=np.arange(len(self.coordinate_systems))),
                )
                element = np.array([e.total for e in element])
                element = concat(element, dict(port=port))
            elif isinstance(element, AntennaArray):
                element = element.total
            self._elements = element * self.tpf
        return self._elements

    @property
    def total(self) -> Antenna:
        """Returns the sum of the elements multiplied by their excitation."""
        # if self._total is None:
        self._total = (self.elements * self.excitation).sum(dim="port") / vector_norm(
            np.abs(self.excitation),
            dim="port",
        )
        return self._total

    @property
    def af(self) -> xr.DataArray:
        """Returns the array factor of the array."""
        # if self._af is None:
        self._af = (self.tpf * self.excitation).sum(dim="port") / vector_norm(
            np.abs(self.excitation),
            dim="port",
        )
        return self._af

    def move(self, hcs: HCS) -> None:
        """Move to a new reference CS. Iterates over cs references."""
        if isinstance(hcs, HCS):
            for ecs in self.coordinate_systems:
                ecs.reference = hcs

            self.initialize()

            # Move the broadcasted elements, too
            if isinstance(self.element, np.ndarray):
                for e, ncs in zip(self.element, self.coordinate_systems):
                    if isinstance(e, AntennaArray):
                        e.move(ncs)
        else:
            raise ValueError("Must be a HCS instance.")

    def positions(self, reference: HCS = None, recursive: bool = True, level: int = 0):
        if reference is None:
            reference = self.coordinate_systems[0].reference
        # Determine how to step recursively through
        if recursive and isinstance(self.element, np.ndarray):
            poss = []
            for idx, (e, cs) in enumerate(zip(self.element, self.coordinate_systems)):
                if isinstance(e, AntennaArray):
                    pos = e.positions(reference=reference, recursive=recursive, level=level + 1)
                else:
                    pos = reference.relative_position_basemag(cs)
                poss.append(pos.assign_coords(**{f"level{level + 1}": idx}))
            positions = xr.concat(poss, dim=f"level{level + 1}")
        elif recursive and isinstance(self.element, AntennaArray):
            poss = []
            for idx, cs in enumerate(self.coordinate_systems):
                # Make a copy of the array
                atemp = self.element.copy()
                # Change each position's reference
                for elcs in atemp.coordinate_systems:
                    elcs.reference = cs
                # Get positions
                pos = atemp.positions(reference=reference, recursive=recursive, level=level + 1)
                poss.append(pos.assign_coords(**{f"level{level + 1}": idx}))

            positions = xr.concat(poss, dim=f"level{level + 1}")
        else:
            # Create position data array
            positions = xr.concat(
                [
                    reference.relative_position_basemag(cs).assign_coords(port=idx)
                    for idx, cs in enumerate(self.coordinate_systems)
                ],
                dim="port",
            )

        return positions

    def show(self, coordinate_system=None, ax=None, **kwargs):
        """
        Inherits CS.show(). Will show the coordinate system of each element in the reference coordinate
        system of the array.
        """
        # Set coordinate system to reference if not specified
        if coordinate_system is None:
            coordinate_system = self.coordinate_systems[0].reference
        # Create figure if ax not given
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            ax.set_box_aspect(aspect=(1, 1, 1))

        # Determine what to plot based on what the element is (antenna or array)
        if isinstance(self.element, np.ndarray):
            for e, cs in zip(self.element, self.coordinate_systems):
                if isinstance(e, AntennaArray):
                    e.show(coordinate_system=coordinate_system, ax=ax, **kwargs)
                else:
                    viewcs(cs, reference_cs=coordinate_system, ax=ax, **kwargs)

        elif isinstance(self.element, AntennaArray):
            for cs in self.coordinate_systems:
                # Make a copy of the array
                atemp = self.element.copy()
                # Change each position's reference
                for elcs in atemp.coordinate_systems:
                    elcs.reference = cs
                # Show
                atemp.show(coordinate_system=coordinate_system, ax=ax, **kwargs)
        else:
            for cs in self.coordinate_systems:
                viewcs(cs, reference_cs=coordinate_system, ax=ax, **kwargs)

        return ax

    def showxy(
        self,
        coordinate_system: HCS | None = None,
        ax: plt.Axes | None = None,
        rs: float = 0.9,
        units: str = "cm",
        unit_shape: str = "circle",
        color_type: str = "shape",
        cmap: plt.Colormap = plt.get_cmap("viridis"),
        exc_isel: dict = {},
        **kwargs,
    ) -> plt.Axes:
        """
        Show elements as circles in xy-plane. Will show the coordinate system of each
        element in the reference coordinate system of the array.
        """
        unit_shape = unit_shape.lower()
        if unit_shape == "circle":
            shp_cls = plt.Circle
        elif unit_shape == "hexagon":
            shp_cls = partial(RegularPolygon, numVertices=6)
        elif unit_shape == "square":
            shp_cls = partial(RegularPolygon, orientation=np.deg2rad(45), numVertices=4)

        if color_type.lower() == "shape":
            pass
        elif color_type.lower() == "mag":
            exc = 20 * np.log10(np.abs(self._get_excitations(isel=exc_isel))).values.ravel()
            norm = plt.Normalize(vmin=exc.max() - 20, vmax=exc.max())
            colors = cmap(norm(exc))
            kwargs = {**kwargs, **dict(facecolors=colors)}
        elif color_type.lower() == "phase":
            exc = np.angle(self._get_excitations(isel=exc_isel).values, deg=True).ravel()
            cmap = plt.get_cmap("viridis")
            norm = plt.Normalize(vmin=-180, vmax=180)
            colors = cmap(norm(exc))
            kwargs = {**kwargs, **dict(facecolors=colors)}
        elif color_type.lower() == "subarray":
            pass
        else:
            raise ValueError(f"Color type {color_type} not supported.")

        # Set coordinate system to reference if not specified
        if coordinate_system is None:
            coordinate_system = self.coordinate_systems[0].reference
        # Create figure if ax not given
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")

        centers = self.positions(reference=coordinate_system, recursive=True)
        centers = centers.sel(position=["x", "y"])
        # Take time at first index if time in dims
        if "time" in centers.dims:
            centers = centers.isel(time=0)
        centers.data = (centers.data * ureg.get_base_units(ureg.m)[1]).to(units)
        if color_type.lower() == "subarray":
            tmp = centers.values
            l = tmp.reshape(-1, *list(tmp.shape[-2:])).shape[0]
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i % 10) for i in range(l) for p in range(centers.port.size)]
            kwargs = {**kwargs, **dict(facecolors=colors)}

        # Set axis
        poss = np.array(centers).reshape(-1, 2)
        pmin = poss.min(axis=0)
        pmax = poss.max(axis=0)
        dx = dy = np.inf
        ax.set_xlim(pmin[0] * 1.2, pmax[0] * 1.2)
        ax.set_ylim(pmin[1] * 1.2, pmax[1] * 1.2)
        # Set radii
        a = np.diff(poss, axis=0)[:, 0]
        a = abs(a[abs(a) > 0.001])
        if len(Counter(a).most_common(1)) > 0:
            dx = Counter(a).most_common(1)[0][0]

        a = np.diff(poss, axis=0)[:, 1]
        a = abs(a[abs(a) > 0.001])
        if len(Counter(a).most_common(1)) > 0:
            dy = Counter(a).most_common(1)[0][0]
        r = np.min([dx, dy]) * rs / 2
        patches = []
        for p in poss:
            patches.append(shp_cls(p, radius=r, clip_on=False))

        ax.add_collection(PatchCollection(patches, **kwargs))
        # Label
        ax.set_xlabel(f"x [{units}]")
        ax.set_ylabel(f"y [{units}]")

        return ax

    def _get_excitations(self, level: int = 0, isel: dict = {}) -> xr.DataArray:
        excitations = self.excitation
        if isel:
            excitations = excitations.isel({k: v for k, v in isel.items() if k in excitations.dims})

        # Determine whether to recurse (antenna or array)
        if isinstance(self.element, np.ndarray):
            # TODO Not sure how to handle this as it could be sparse
            excs = []
            for idx, (e, ei) in enumerate(zip(self.element, excitations)):
                if isinstance(e, AntennaArray):
                    exc = e._get_excitations(level=level + 1, isel=isel)
                    excs.append((ei * exc).assign_coords(**{f"level{level + 1}": idx}))
            excitations = xr.concat(excs, dim=f"level{level + 1}")
        elif isinstance(self.element, AntennaArray):
            subexc = self.element._get_excitations(level=level + 1, isel=isel)
            excitations = xr.DataArray(
                np.outer(excitations, subexc),
                dims=("port", f"level{level + 1}"),
                coords={"port": excitations.port.values, f"level{level + 1}": subexc.port.values},
            )
        return excitations

    def copy(self) -> AntennaArray:
        """Return a copy of self."""
        cs = []
        for csi in self.coordinate_systems:
            csn = HCS(csi.origin, reference=csi.reference, rotation=csi.rotation)
            cs.append(csn)
        el = self.element.copy()
        if isinstance(el, np.ndarray):
            els = []
            for e in el:
                els.append(e.copy())
            el = np.array(els)
        return AntennaArray(el, cs)


class TranslatedPhase(Antenna):
    """Translated phase antenna pattern."""

    def __init__(
        self,
        frequency: Quantity | np.typing.ArrayLike,
        coordinate_systems: HCS | None = None,
    ) -> None:
        # Default Coordinates
        if isinstance(frequency, Quantity):
            frequency = frequency.to_base_units()
        frequency = xr.DataArray(
            frequency,
            dims=("frequency",),
            coords=dict(frequency=frequency),
            attrs=dict(units="Hz"),
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
        port = xr.DataArray(
            coordinate_systems,
            dims=("port",),
            coords=dict(port=np.arange(len(coordinate_systems))),
        )
        # self._port_pos
        dims = ("polarization", "port", "frequency", "phi", "theta")
        coords = dict(polarization=["apolar"], port=port, frequency=frequency, phi=phi, theta=theta)
        antenna_function = AntennaFunction(dims, coords, self._antenna_func, "phitheta")

        super().__init__(antenna_function, coordinate_systems[0].reference)

    def _antenna_func(self, port=None, frequency=None, phi=None, theta=None):
        """Translated phase radiation pattern (isotropic with phase difference dependent on position)."""
        # Determine propagation constant from wavelength
        lam = 1 / frequency * ureg.speed_of_light
        lam.data = lam.data.to_base_units().magnitude
        k = (2 * np.pi) / lam

        # Spatial Phase
        dimsgrid = theta * phi
        sp = np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta) * xr.ones_like(phi),
            ],
        )
        sp = xr.DataArray(
            sp,
            dims=["position"] + list(dimsgrid.dims),
            coords={**dict(position=["x", "y", "z"]), **dimsgrid.coords},
        )

        # Create position data array
        # Take only port dimension
        iport = port.isel(dict.fromkeys(set(port.dims) - {"port"}, 0))
        pos = xr.concat(
            [cs.origin.basemag for cs in iport.data.ravel()],
            dim=iport.port,
        )

        # k * spatial phase mapping
        kds = k * pos * sp
        # kdsunits = kds.data.units
        kds = kds.sum(dim="position")
        # Element phase
        data = np.exp(1j * kds)

        # Assign polarization dim
        data = data.assign_coords(dict(polarization="apolar"))
        data = data.expand_dims(dim="polarization")

        return data


def get_position_in_base(
    reference_cs: HCS,
    coordinate_frame: str = "phitheta",
    coordinate_system: HCS | None = None,
    base_coordinate_frame: str = "phitheta",
    convert_kwargs: dict | None = None,
    **kwargs,
):
    """
    Dispersive phase steering where constant phase is determined and the desired steering frequency. Sets the excitation
    property of the input array.
    """
    # Default convert kwargs
    if convert_kwargs is None:
        convert_kwargs = dict()
    # Make kwargs xr.DataArrays
    for k, v in kwargs.items():
        if not isinstance(v, xr.DataArray):
            if not isinstance(v, Quantity):
                raise ValueError("Input must be specified as a pint.Quantity")
            if v.shape == ():
                # Needs to have a dimension
                v = np.array([v.magnitude]) * v.units
            # Make xr.DataArray and assign to kwarg dict
            kwargs[k] = xr.DataArray(v, coords={k: v.to_base_units()}, dims=(k,))

    # Get requested angle
    request_spatial_dims = list(conversions.COORDINATE_DIMS[coordinate_frame])
    requestangles = [kwargs[k] for k in request_spatial_dims]

    # Coordinate frame transforms
    request2uvw = getattr(conversions, f"{coordinate_frame.lower()}2uvw")
    uvw2base = getattr(conversions, f"uvw2{base_coordinate_frame.lower()}")

    # Add special kwargs to convert coordinate frame function
    if coordinate_frame.lower() == "llh" or coordinate_frame.lower() == "ecef":
        ref_cs = coordinate_system
        if ref_cs is None:
            ref_cs = reference_cs

        convert_kwargs = {**convert_kwargs, **dict(reference_cs=ref_cs)}

    # Convert requested spatial dims uvw in self.coordinate_system
    uvw = request2uvw(*requestangles, **convert_kwargs)

    # Rotate all the points if different coordinate system
    if coordinate_system is not None:
        # Map requested points to base coordinate_system
        # Get relative rotation from requested coordinate_system and self.coordinate_system
        # TODO: Check this is called from the correct CS
        rprod = reference_cs.get_relative_rotation(coordinate_system)

        # Make uvw a dataarray
        uvw_xr = []
        for da, coord in zip(uvw, ["x", "y", "z"]):
            da = da.assign_coords(dict(position=coord))
            uvw_xr.append(da)
        uvw_xr = xr.concat(uvw_xr, dim="position")

        # Rotate uvw points
        uvw_prime = rprod.apply(uvw_xr, inverse=True)

        # Format back to tuple
        uvw = []
        for coord in ["x", "y", "z"]:
            da = uvw_prime.sel(position=coord)
            da = da.drop_vars("position")
            uvw.append(da)

    # Convert the uvw positions to the base coordinate frame of the data
    base = uvw2base(*uvw)

    return base


def apply_taper(
    phased_array: AntennaArray,
    window,
    dim="x",
    cs: HCS = None,
):
    if cs is None:
        cs = phased_array.coordinate_systems[0].reference
    if not isinstance(window, xr.DataArray):
        positions = phased_array.positions(reference=cs, recursive=True)
        posdim = positions.sel(position=dim)
        window = xr.DataArray(
            window,
            dims=(dim,),
            coords={dim: np.linspace(posdim.min().item(), posdim.max().item(), window.size)},
        )

    # Recursive application
    if isinstance(phased_array.element, np.ndarray):
        taper = []
        for e in phased_array.element:
            # if isinstance(e, AntennaArray):
            taper.append(apply_taper(e, window, dim, cs))
        taper = np.array(taper)
        taper_norm = np.linalg.norm(taper)
        phased_array.taper *= xr.DataArray(
            taper / taper_norm,
            dims=("port",),
            coords=dict(port=np.arange(phased_array.element.size)),
        )

        return taper_norm

    if isinstance(phased_array.element, AntennaArray):
        raise NotImplementedError("Cannot taper subarrays unless they are broadcasted.")
    pos = phased_array.positions(reference=cs, recursive=False)
    posdim = pos.sel(position=dim)
    # Get window
    windowed = window.interp({dim: posdim}).drop_vars(["position", dim])
    # Get Norm
    taper_norm = np.linalg.norm(windowed)
    # Normalized
    phased_array.taper *= xr.DataArray(
        windowed / taper_norm,
        dims=("port",),
        coords=dict(port=np.arange(windowed.size)),
    )

    return taper_norm


def steer_phase_centers(
    phased_array: AntennaArray,
    frequency: Quantity = None,
    coordinate_frame="phitheta",
    coordinate_system=None,
    convert_kwargs: dict | None = None,
    **kwargs,
):
    """
    Dispersive phase steering where constant phase is determined and the desired steering frequency. Sets the excitation
    property of the input array
    """
    # Default convert kwargs
    if convert_kwargs is None:
        convert_kwargs = dict()

    # Steer subarrays
    if isinstance(phased_array.element, AntennaArray):
        steer_phase_centers(
            phased_array.element,
            frequency=frequency,
            coordinate_frame=coordinate_frame,
            coordinate_system=coordinate_system,
            convert_kwargs=convert_kwargs,
            **kwargs,
        )
    elif isinstance(phased_array.element, np.ndarray):
        for e in phased_array.element:
            if isinstance(e, AntennaArray):
                steer_phase_centers(
                    e,
                    frequency=frequency,
                    coordinate_frame=coordinate_frame,
                    coordinate_system=coordinate_system,
                    convert_kwargs=convert_kwargs,
                    **kwargs,
                )

        phased_array._elements = None

    # Convert requested positions to base phi/theta
    p0, t0 = get_position_in_base(
        phased_array.coordinate_systems[0].reference,
        coordinate_frame=coordinate_frame,
        coordinate_system=coordinate_system,
        base_coordinate_frame="phitheta",
        convert_kwargs=convert_kwargs,
        **kwargs,
    )
    p0 = p0.squeeze()
    t0 = t0.squeeze()

    # Wavelength and propagation constant
    lam = 1 / frequency * ureg.speed_of_light
    lam = lam.to_base_units().magnitude
    k = (2 * np.pi) / lam

    # Port
    port = phased_array.tpf.data.coords["port"]

    # Create position data array
    pos = xr.concat(
        [cs.origin.basemag for cs in port.data.ravel()],
        dim=port.port,
    )

    # Phase Progression is just the negative of the spatial phase factor
    sp = [np.sin(t0) * np.cos(p0), np.sin(t0) * np.sin(p0), np.cos(t0)]
    sp = xr.DataArray(
        sp,
        dims=["position"] + list(t0.dims),
        coords={**dict(position=["x", "y", "z"]), **t0.coords},
    )

    # Excitation
    exc = np.exp(-1j * (k * pos * sp).sum(dim="position"))

    # Assign to steering vector
    phased_array.steering_vector = exc

    # Initialize
    phased_array.initialize()

    # Return angles in base for inspection
    return p0, t0


def plot_grating_lobe_diagram(
    dx: Quantity | float,
    dy: Quantity | float,
    steering_angles: tuple,
    lattice_type: str = "rectangular",
    max_scan: Quantity = 60 * ureg.degree,
    figsize: tuple = (8, 8),
    show_pairs: bool | None = None,
) -> plt.Axes:
    """
    Plots the grating lobe diagram for a 2D phased array with grid.

    Parameters
    ----------
    dx : Quantity or float
        x-spacing in wavelengths
    dy : Quantity or float
        y-spacing in wavelengths
    steering_angles_deg : tuple
        Tuple of steering angles (phi, theta) in degrees.
    lattice_type:str
        'rectangular' or 'triangular' lattice type.
    max_scan: Quantity
        Maximum scan (theta) to draw circle
    steering_angles_deg: Tuple of steering angles (azimuth, elevation) in degrees.


    Reference
    ---------
        Grating Lobe Suppression with Element Count Optimization in Planar Antenna Array,
        https://www.scirp.org/journal/paperinformation?paperid=54250
    """
    # Accept plain floats as dimensionless wavelength units
    if not isinstance(dx, Quantity):
        dx = dx * ureg.dimensionless
    if not isinstance(dy, Quantity):
        dy = dy * ureg.dimensionless

    phi_steer, theta_steer = steering_angles

    u_steer = np.sin(theta_steer) * np.cos(phi_steer)
    v_steer = np.sin(theta_steer) * np.sin(phi_steer)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("u (sin(theta)cos(phi))")
    ax.set_ylabel("v (sin(theta)sin(phi))")
    ax.set_title(
        f"Grating Lobe Diagram\ndx={dx:.3f~#P}λ,dy={dy:.3f~#P}λ\nMax. Scan {max_scan.to('degree').magnitude}°",
    )
    ax.add_patch(plt.Circle((0, 0), 1, color="C0", alpha=0.25))  # Visible region
    ax.add_patch(plt.Circle((0, 0), np.sin(max_scan), color="C0", alpha=0.25))  # Scan region

    # Plot grating lobe locations
    if lattice_type == "rectangular":
        lattice_pairs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        if show_pairs is None:
            show_pairs = [True] * len(lattice_pairs)
        for (m, n), sp in zip(lattice_pairs, show_pairs):
            if sp:
                u_lobe = u_steer + m / dx
                v_lobe = v_steer + n / dy

                ax.plot(u_lobe, v_lobe, "ro", markersize=5, mec="grey", mfc="white")
                ax.add_patch(plt.Circle((m / dx, n / dy), 1, color="grey", alpha=0.25))
                ax.add_patch(
                    plt.Circle((m / dx, n / dy), np.sin(max_scan), color="grey", alpha=0.25),
                )

    elif lattice_type == "triangular":
        lattice_pairs = [(-1, -1), (0, -1), (-1, 0), (1, 0), (0, 1), (1, 1)]
        if show_pairs is None:
            show_pairs = [True] * len(lattice_pairs)
        for (m, n), sp in zip(lattice_pairs, show_pairs):
            if sp:
                u_lobe = u_steer + m / dx
                v_lobe = v_steer + (2 * n - m) / (dy * np.sqrt(3))
                ax.plot(u_lobe, v_lobe, "ro", markersize=5, mec="grey", mfc="white")
                ax.add_patch(
                    plt.Circle(
                        (m / dx, (2 * n - m) / (dy * np.sqrt(3))),
                        1,
                        color="grey",
                        alpha=0.25,
                    ),
                )
                ax.add_patch(
                    plt.Circle(
                        (m / dx, (2 * n - m) / (dy * np.sqrt(3))),
                        np.sin(max_scan),
                        color="grey",
                        alpha=0.25,
                    ),
                )

    ax.plot(
        u_steer,
        v_steer,
        "bo",
        markersize=6,
        mec="k",
        mfc="white",
        label="Main Beam",
    )  # Main beam location

    ax.legend()
    ax.grid(False)
    ax.set_aspect("equal")

    return ax
