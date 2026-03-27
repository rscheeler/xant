"""xant - Spatial antenna analysis with xarray."""
import operator as opr
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pint
import xarray as xr
from hics import GLOBAL_CS, HCS
from loguru import logger
from scipy.ndimage import map_coordinates, spline_filter
from scipy.spatial.transform import Rotation

from . import conversions, polarization, ureg
from .utils import apply_rotation, fast_nearest_indices

warnings.filterwarnings("ignore")


class Antenna:
    """
    Class for representing antenna patterns as an object.

    Parameters
    ----------
    data : str, Path, xr.DataArray, or callable
        Path to data file, xr.DataArray of antenna data, or function representing antenna data
    hcs : optional, HCS
        Coordinate system to define antennas point in space
    """

    REQUIRED_ATTRS = ["coordinate_frame"]

    def __init__(
        self,
        data: Union[str, Path, xr.DataArray, "AntennaFunction"],
        hcs: HCS | None = None,
    ):
        # Make sure if str is passed implying file import that it is converted to Path object
        if isinstance(data, str) or isinstance(data, Path):
            data = Path(data)
            if data.suffix == ".antnc":
                data = xr.open_dataarray(data, engine="h5netcdf")
        self.data = data

        # Verify data
        self.validate(data)

        # Set cs
        if hcs is None:
            hcs = GLOBAL_CS
        self.hcs = hcs

        # Initialize variables
        self.__spline_filter = None

    @staticmethod
    def validate(data):
        """
        Validation of the data structure.
        """
        # Verify required attributes
        if not set(Antenna.REQUIRED_ATTRS).issubset(list(data.attrs.keys())):
            raise AttributeError(
                f"Required attributes not found in data. Be sure {', '.join(Antenna.REQUIRED_ATTRS)} is an attribute."
            )

    # Properties
    @property
    def hcs(self):
        return self._hcs

    @hcs.setter
    def hcs(self, value):
        # Change coordinate system
        if isinstance(value, HCS):
            self._hcs = value
        else:
            raise ValueError("Must be a HCS instance.")

    def move(self, hcs: HCS):
        """
        Move antenna to coordinate system, will propagate changes if computation has been applied to antenna.

        Parameters
        ----------
        hcs : HCS
            Coordinate system to move the antenna to.
        """
        if isinstance(hcs, HCS):
            self.hcs = hcs
            # If operator propagate change
            if isinstance(self.data, OperatorFunction):
                if isinstance(self.data.self_patt, Antenna):
                    self.data.self_patt.hcs = hcs
                if isinstance(self.data.other, Antenna):
                    self.data.other.hcs = hcs

        else:
            raise ValueError("Must be a CoordinateSystem instance.")

    # Operators
    def __math_operation__(self, other, operator):
        """
        Performs math operation on object.

        Parameters
        ----------
        """
        # Do some validation of the operation
        # Constrain addition subtraction to not work with floats
        if not isinstance(other, Antenna):
            if operator == opr.add or operator == opr.sub:
                raise TypeError("Can not add constant to antenna.")
        # Can't multiply two patterns that are vector patterns
        if isinstance(other, Antenna):
            if (
                not np.array_equal(np.array(self.data.coords["polarization"]), np.array(["apolar"]))
                and not np.array_equal(
                    np.array(other.data.coords["polarization"]), np.array(["apolar"])
                )
                and operator in [opr.mul, opr.truediv]
            ):
                raise TypeError("Can't multiply polar patterns.")

        # Verify self and other have a coordinate system in common
        if isinstance(other, Antenna):
            common_hcs = self.hcs.find_common_cs(other.hcs)
            if common_hcs is None:
                raise ValueError("Antennas have no common coordinate system.")
        else:
            common_hcs = self.hcs
        # Create instance of OperatorFunction which stores self and other
        func = OperatorFunction(self, other, operator)
        # Create new antenna object with the operator function as the data input
        operated_ant = Antenna(func, common_hcs)

        return operated_ant

    def __add__(self, other):
        return self.__math_operation__(other, opr.add)

    def __sub__(self, other):
        return self.__math_operation__(other, opr.sub)

    def __neg__(self):
        return self.__math_operation__(-1, opr.mul)

    def __mul__(self, other):
        return self.__math_operation__(other, opr.mul)

    def __truediv__(self, other):
        return self.__math_operation__(other, opr.truediv)

    def __pow__(self, other):
        return self.__math_operation__(other, opr.pow)

    def sum(self, dim):
        """
        Sum pattern along specified dimension.
        """
        # Create instance of OperatorFunction which stores self and other
        func = OperatorFunction(self, None, "sum", is_xrda_method=True, dim=dim)
        # Create new antenna object with the operator function as the data input
        operated_ant = Antenna(func, self.hcs)

        return operated_ant

    # Private methods for interpolation
    def _spline_filter(self,order=3,mode="nearest",**kwargs):
        """"""
        if self.__spline_filter is None and isinstance(self.data, xr.DataArray):
            # 1. Compute coefficients in the ORIGINAL data order
            
            srcdata = self.data.transpose("polarization", ...)
            raw_vals = srcdata.values
            if hasattr(raw_vals, "magnitude"):
                raw_vals = raw_vals.magnitude
            
            # We still loop over polarization to keep it 2D/3D spatial
            # but we keep the result as an xarray object
            coeffs_raw = np.array([
                spline_filter(raw_vals[i, ...], mode=mode, order=order,**kwargs)
                for i in range(raw_vals.shape[0])
            ])
            
            # 2. Wrap it back into a DataArray with the same dims as self.data
            self.__spline_filter = xr.DataArray(
                coeffs_raw, 
                dims=srcdata.dims, 
                coords=srcdata.coords
            )
        return self.__spline_filter

    def _interp(self, order=3, **kwargs):
        if isinstance(self.data, AntennaFunction):
            return self.data.antenna_callable(**kwargs)

        # 1. STRIP UNITS AND METADATA IMMEDIATELY
        # Pre-process all kwargs into raw float64 magnitudes in base units
        clean_kwargs = OrderedDict()
        for k, v in kwargs.items():
            if hasattr(v, "data") and isinstance(v.data, pint.Quantity):
                clean_kwargs[k] = v.data.to_base_units().magnitude
            elif isinstance(v, pint.Quantity):
                clean_kwargs[k] = v.to_base_units().magnitude
            else:
                clean_kwargs[k] = np.asarray(v)

        req_dims = list(clean_kwargs.keys())
        tpos = list(set(self.data.dims) - set(req_dims)) + req_dims
        # Get filter coefficients and transpose
        coeffs = self._spline_filter(order=order)
        coeffs = coeffs.transpose(*tpos).values

        idxs = []
        for k, v_raw in clean_kwargs.items():
            coord = self.data.coords[k]
            coord_vals = coord.values
            if hasattr(coord_vals, "magnitude"):
                coord_vals = coord_vals.magnitude

            # Case A: Singleton dimension
            if coord.size == 1:
                idxs.append(np.zeros_like(v_raw))
                continue

            # Case B: Uniform Grid
            # We check if the diffs are all the same
            diffs = np.diff(coord_vals)
            if np.allclose(diffs, diffs[0]):
                dv = diffs[0]
                mn = coord_vals.min()
                idxs.append((v_raw - mn) / dv)
            
            # Case C: Monotonic but Non-Uniform
            elif np.all(diffs > 0) or np.all(diffs < 0):
                # Find insertion point
                side = 'left' if diffs[0] > 0 else 'right'
                idx = np.searchsorted(coord_vals, v_raw, side=side)
                # Clip to bounds and adjust for nearest neighbor
                idx = np.clip(idx, 1, len(coord_vals) - 1)
                # Check if left or right is closer
                left = coord_vals[idx - 1]
                right = coord_vals[idx]
                idx = np.where(np.abs(v_raw - left) < np.abs(v_raw - right), idx - 1, idx)
                idxs.append(idx.astype(np.float64))

            # Case D: Truly Arbitrary (Parallelized via Numba)
            else:
                idxs.append(fast_nearest_indices(coord_vals, v_raw).astype(np.float64))
        pixel_coords = [idx.ravel() for idx in idxs]

        # Loop over polarization coordinate and interpolate
        data = []

        for i in range(coeffs.shape[0]):
            data.append(
                map_coordinates(
                    coeffs[i,...],
                    pixel_coords,
                    order=order,
                    prefilter=False,
                    mode="nearest",
                ).reshape(idxs[0].shape),
            )

        # Make into DataArray
        data = np.array(data)
        addeddims = ["polarization"]

        # Assemble coords
        coords = {}
        for k in addeddims:
            coords[k] = self.data.coords[k]

        interpdims = list(list(kwargs.values())[0].dims)
        for k in interpdims:
            coords[k] = v.coords[k]

        data = xr.DataArray(data, dims=addeddims + interpdims, coords=coords)

        return data

    def request_data(
        self,
        coordinate_frame: str | None = None,
        hcs: HCS | None = None,
        convert_kwargs: dict | None = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        Returns gridded data in the desired coordinate frame.

        Parameters
        ----------
        coordinate_frame : optional,str
            Antenna coordinate frame to request the data in. Default is the coordinate_frame of the data
        hcs : optional, HCS
            Coordinate system to view the data in
        """
        # Make kwargs xr.DataArrays
        for k, v in kwargs.items():
            if not isinstance(v, xr.DataArray):
                if not isinstance(v, pint.Quantity):
                    raise ValueError("Input must be specified as a pint.Quantity")
                elif np.array(v).shape == ():
                    # Needs to have a dimension
                    v = np.array([v.magnitude]) * v.units
                # Make xr.DataArray and assign to kwarg dict
                kwargs[k] = xr.DataArray(v, coords={k: v.to_base_units()}, dims=(k,))

        # Set default coordinate frame
        if coordinate_frame is None:
            coordinate_frame = self.data.coordinate_frame

        # Set default coordinate system
        if hcs is None:
            hcs = self.hcs

        # Default convert kwargs
        if convert_kwargs is None:
            convert_kwargs = dict()

        # First pass to operator function since it will ultimately be passed to request_data
        if isinstance(self.data, OperatorFunction) or isinstance(self.data, ConcatFunction):
            data = self.data.antenna_callable(
                coordinate_frame=coordinate_frame,
                hcs=hcs,
                convert_kwargs=convert_kwargs,
                **kwargs,
            )
        # Process inputs and get data
        else:
            # Get relative rotation from requested hcs and self.hcs
            rprod = self.hcs.get_relative_rotation(hcs)

            # Check to see if rotation and data dims intersect
            rdims = OrderedDict()
            if isinstance(rprod, xr.DataArray):
                intersect_dims = list(set(rprod.dims) & set(self.data.dims))
                for d in intersect_dims:
                    rdims[d] = rprod.coords[d]

            # Shared keys in rdims and kwargs
            # TODO: see if this can be simplified with xr.align
            shared_keys = rdims.keys() & kwargs.keys()
            # Loop through shared keys and  change the kwarg and pop them from rdims
            for k in shared_keys:
                # Make into arrays
                rv = rdims[k]
                if not isinstance(rv, xr.DataArray):
                    rv = np.array(rv)
                    if rv.shape == ():
                        rv = np.array([rv])

                kv = kwargs[k]
                if not isinstance(kv, xr.DataArray):
                    kv = np.array(kv)
                    if kv.shape == ():
                        kv = np.array([kv])

                # Get intersecting values and convert to xarray DataArray
                intersecting_values = np.intersect1d(rv, kv)
                intersecting_values = xr.DataArray(
                    intersecting_values, coords={k: intersecting_values}, dims=(k,)
                )

                # Change kwarg
                kwargs[k] = intersecting_values

                # Down-select rprod
                rprod = rprod.sel({k: intersecting_values})

                # Pop from rdims
                rdims.pop(k)

            # Concatenate rotational dims and kwargs
            kwargs = {**rdims, **kwargs}

            # Make kwargs ordered
            kwargs = OrderedDict(kwargs)

            # Spatial dims
            request_spatial_dims = list(conversions.COORDINATE_DIMS[coordinate_frame])
            base_spatial_dims = list(conversions.COORDINATE_DIMS[self.data.coordinate_frame])

            # Base Dims
            base_dims = list(self.data.dims)
            # Don't interpolate polarization
            base_dims.pop(base_dims.index("polarization"))

            # Remove base spatial dimensions
            for k in base_spatial_dims:
                base_dims.pop(base_dims.index(k))
            request_dims = base_dims + request_spatial_dims

            # Remove dimensions that are not present
            if not set(list(kwargs.keys())).issubset(request_dims):
                missing_dims = list(set(kwargs.keys()).difference(set(request_dims)))
                for k in missing_dims:
                    kwargs.pop(k)

            # Add dims not requested
            missing_dims = request_dims.copy()
            for k in kwargs.keys():
                missing_dims.pop(missing_dims.index(k))
            kwargs = OrderedDict({**{k: self.data.coords[k] for k in missing_dims}, **kwargs})

            # Grid requested data
            gridcoords = xr.broadcast(*list(kwargs.values()))
            gridcoords = list(gridcoords)

            # Coordinate frame transforms
            request2uvw = getattr(conversions, f"{coordinate_frame.lower()}2uvw")
            uvw2base = getattr(conversions, f"uvw2{self.data.coordinate_frame}")

            # Request Angles
            requestangles = [gridcoords[list(kwargs.keys()).index(k)] for k in request_spatial_dims]

            # Add special kwargs to convert coordinate frame function
            if coordinate_frame.lower() in ("llh", "ecef", "h3"):
                ref_hcs = hcs
                if ref_hcs is None:
                    ref_hcs = self.hcs

                convert_kwargs = {**convert_kwargs, **dict(reference_hcs=ref_hcs)}

            # Convert requested spatial dims uvw in self.hcs
            uvw = request2uvw(*requestangles, **convert_kwargs)
            uvw_request = deepcopy(uvw)

            # Make uvw a DataArray
            uvw_xr = []
            for da, coord in zip(uvw, ["x", "y", "z"]):
                da = da.assign_coords(dict(position=coord))
                uvw_xr.append(da)
            uvw_xr = xr.concat(uvw_xr, dim="position")

            # Rotate uvw points
            if isinstance(rprod, Rotation):
                uvw_prime = apply_rotation(rprod, uvw_xr, inverse=False)
            else:
                uvw_prime = rprod.apply(uvw_xr, inverse=False)

            # Format back to tuple
            uvw = []
            for coord in ["x", "y", "z"]:
                da = uvw_prime.sel(position=coord)
                da = da.drop_vars("position")
                uvw.append(da)

            # Convert the uvw positions to the base coordinate frame of the data
            baseangles = uvw2base(*uvw)

            # Create interpolation dictionary
            gridcoords = OrderedDict(
                {
                    **{k: gridcoords[list(kwargs.keys()).index(k)] for k in base_dims},
                    **{k: v for k, v in zip(base_spatial_dims, baseangles)},
                }
            )

            # Make sure values are in base units
            interp_dict = {}
            for k, v in gridcoords.items():
                if isinstance(v.data, pint.Quantity):
                    v.data = v.data.to_base_units()
                interp_dict[k] = v
            interp_dict = OrderedDict(interp_dict)

            # Call interpolator
            data = self._interp(**interp_dict)

            # Rotate polarization
            data = polarization.rotate_polarization(data, uvw_request, rprod)

            # Convert nans to zeros
            data = data.fillna(0)

            # Add in required attrs
            # If kwargs have already been processed coordinate frame is not correct so grab from coord keys
            # TODO: This is going to be an issue with azel/elaz
            cf = coordinate_frame
            for k, v in conversions.COORDINATE_DIMS.items():
                if set(v).issubset(list(data.coords.keys())):
                    cf = k
                    break
            data.attrs = {**data.attrs, **dict(coordinate_frame=cf)}

        # Project the polarizations reduce data first
        if data.polarization.size > 1:
            data = data.sel(polarization=["theta", "phi"])

        # Add special kwargs to convert coordinate frame function - need to do this again as coordinate_frame
        # of data may differ
        if data.coordinate_frame.lower() in ("llh", "ecef", "h3"):
            ref_hcs = hcs
            if ref_hcs is None:
                ref_hcs = self.hcs
            convert_kwargs = {**convert_kwargs, **dict(reference_hcs=ref_hcs)}
        # Copy back in dims if not present for polarization transform
        if not set(conversions.COORDINATE_DIMS[data.coordinate_frame]).issubset(data.dims):
            for d in conversions.COORDINATE_DIMS[data.coordinate_frame]:
                data = data.assign_coords({d: kwargs[d]})
        # Project all the polarizations
        data = polarization.project_all_polarizations(data, convert_kwargs)

        # Some pint/xarray issues just force if not a quantity
        if not isinstance(data.data, pint.Quantity):
            data.data = data.data * ureg.dimensionless

        # Add name
        data.name = "gain"

        return data

    def beamwidth(
        self,
        dim: str,
        coordinate_frame: str,
        pwr: float = ureg("-3 dB"),
        angles=np.linspace(-180, 180, 36001) * ureg.degree,
        **kwargs,
    ) -> pint.Quantity:
        """
        Calculate the beamwidth in the coordinate frame.

        Parameters
        ----------
        dim : str
            Dimension to calculate beamwidth along
        coordinate_frame : str
            What coordinate_frame to perform beamwidth calculation in
        pwr : Quantity
            Power relative to peak to calculate beamwidth at. Default is 3 dB


        Returns:
        -------
        bw : Quantity
            Beamwidth in degrees
        """
        # Slice the data
        slc = self.request_data(**{dim: angles, "coordinate_frame": coordinate_frame}, **kwargs)

        # Normalize
        slc = abs(slc).sel(polarization="apolar")
        slc /= slc.max()

        # Find 3dB points - handle omni antennas also
        slcsearch = slc - np.sqrt(pwr.to_base_units())
        slcsearch = abs(slcsearch.where(slcsearch <= 0))
        if np.all(slcsearch.isnull().values):
            bw = angles.max() - angles.min()
        else:
            bw = (
                np.rad2deg(
                    abs(
                        slcsearch.squeeze()
                        .sortby(slcsearch.squeeze())[:2]
                        .coords[dim]
                        .diff(dim)
                        .item()
                    )
                )
                * ureg.degree
            )

        return bw

    def d0(
        self,
        theta=np.linspace(0, 180, 91) * ureg.degree,
        phi=np.arange(-180, 180, 5) * ureg.degree,
        coordinate_frame="phitheta",
        **kwargs,
    ):
        dat = self.request_data(theta=theta, phi=phi, coordinate_frame=coordinate_frame, **kwargs)
        dat = dat.sel(polarization="apolar")
        u = np.abs(dat) ** 2
        dth = np.diff(theta).mean().to("radian").magnitude
        dph = np.diff(phi).mean().to("radian").magnitude
        dat = u * np.sin(dat.theta) * dth * dph
        trp = dat.sum(dim=["theta", "phi"])
        return 4 * np.pi * u.max(dim=["theta", "phi"]) / trp

    def trp(
        self,
        theta=np.linspace(0, 180, 91) * ureg.degree,
        phi=np.arange(-180, 180, 5) * ureg.degree,
        coordinate_frame="phitheta",
        **kwargs,
    ):
        dat = self.request_data(theta=theta, phi=phi, coordinate_frame=coordinate_frame, **kwargs)
        dat = dat.sel(polarization="apolar")
        u = np.abs(dat) ** 2
        dth = np.diff(theta).mean().to("radian").magnitude
        dph = np.diff(phi).mean().to("radian").magnitude
        dat = u * np.sin(dat.theta) * dth * dph
        trp = dat.sum(dim=["theta", "phi"])
        return trp

    def static_scene(self, llas, uns, uxs) -> xr.DataArray:
        return

    def export(self, filename: Path):
        """Exports data to a NetCDF file using the h5netcdf engine."""
        # Create a dataset that contains the complex data
        self.data.to_netcdf(Path(filename).with_suffix(".antnc"), engine="h5netcdf")

    def copy(self):
        """Return a copy of self."""
        return Antenna(deepcopy(self.data), self.hcs)


class AntennaFunction:
    """
    Class for defining antenna to emulate an xarray object. Contains required attributes and function
    for interpolating.

    Parameters
    ----------
    dims : tuple
        Dimension names
    coords : dict
        Dictionary of coordinates
    antenna_callable : callable
        Function to call when requesting data, emulates data interpolation
    coordinate_frame : str
        Antenna coordinate frame
    attrs : dict, optional
        Optional attributes
    """

    def __init__(
        self,
        dims: tuple,
        coords: dict,
        antenna_callable: callable,
        coordinate_frame: str,
        attrs: dict | None = None,
    ):
        # Collect dimensions and coordinates to emulate xarray.DataArray
        self.dims = dims
        self.coords = coords

        # Initialize attrs
        if attrs is None:
            attrs = dict()

        # Set coordinates as attributes
        for k, v in self.coords.items():
            self.__setattr__(k, v)
        self.attrs = {**dict(coordinate_frame=coordinate_frame), **attrs}

        # Set attributes dict as attributes
        for k, v in self.attrs.items():
            self.__setattr__(k, v)
        self.antenna_callable = antenna_callable


class OperatorFunction(AntennaFunction):
    """
    Class to return a function for Antenna math operations. Stores self, other, and the operation which is only computed
    when request data is called.

    Parameters
    ----------
    self_patt : Antenna
        Self from operator
    other : Antenna, float
        Other for operation
    operator : callable
        Operation function
    """

    def __init__(
        self,
        self_patt: Antenna,
        other: Antenna | float,
        operator: callable,
        is_xrda_method: bool = False,
        **operator_kwargs: dict,
    ):
        # Store self, other, and operator
        self.self_patt = self_patt
        self.other = other
        self.operator = operator
        self.operator_kwargs = operator_kwargs
        self.is_xrda_method = is_xrda_method

        # Get dims and coords from self
        dims = self_patt.data.dims
        coords = self_patt.data.coords
        coordinate_frame = self_patt.data.coordinate_frame

        # Initialize AntennaFunction
        super().__init__(dims, coords, self.antenna_func, coordinate_frame)

    def antenna_func(self, *args, **kwargs):
        """
        Function for returning antenna data
        """
        # Get self data
        selfdata = self.self_patt.request_data(*args, **kwargs)

        # Constrain data to a single basis
        if selfdata.polarization.size > 1:
            selfdata = selfdata.sel(polarization=["theta", "phi"])

        # Get other data depending on whether it is an Antenna or not
        if isinstance(self.other, Antenna):
            otherdata = self.other.request_data(*args, **kwargs)

            # Constrain data to a single basis
            if otherdata.polarization.size > 1:
                otherdata = otherdata.sel(polarization=["theta", "phi"])

            # Drop polarization dimension if apolar and self is not apolar
            if otherdata.polarization.shape == (1,) and selfdata.polarization.shape != (1,):
                otherdata = otherdata.squeeze(dim="polarization")
                otherdata = otherdata.drop_vars("polarization")
            # Drop polarization dimension if self is apolar and other is not apolar
            elif selfdata.polarization.shape == (1,) and otherdata.polarization.shape != (1,):
                selfdata = selfdata.squeeze(dim="polarization")
                selfdata = selfdata.drop_vars("polarization")

        else:
            otherdata = self.other

        # Perform math operation
        if self.is_xrda_method:
            data = selfdata.__getattribute__(self.operator)(**self.operator_kwargs)
        else:
            data = self.operator(selfdata, otherdata)

        # Set attributes
        data.attrs = selfdata.attrs

        return data


class ConcatFunction(AntennaFunction):
    """"""

    def __init__(
        self,
        antennas: list,
        coord: dict,
    ):
        # Store inputs

        self.coord = coord
        self.dim = tuple([k for k in coord.keys()])
        self.antennas = xr.DataArray(antennas, dims=self.dim, coords=coord)

        # Get dims and coords from self
        dims = tuple(list(self.dim) + list(antennas[0].data.dims))
        coords = {**self.coord, **antennas[0].data.coords}
        coordinate_frame = antennas[0].data.coordinate_frame

        # Initialize AntennaFunction
        super().__init__(dims, coords, self.antenna_func, coordinate_frame)

    def antenna_func(self, *args, **kwargs):
        """
        Function for returning antenna data
        """
        # Remove concatenated dim from kwargs if present
        if self.dim[0] in kwargs.keys():
            concat_coord = kwargs.pop(self.dim[0])
        else:
            concat_coord = self.coord[self.dim[0]]
        dropped_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, xr.DataArray):
                if self.dim[0] in v.dims:
                    v = v.isel({self.dim[0]: 0})
                    v = v.drop_vars(self.dim[0])
            dropped_kwargs[k] = v

        # Select antennas
        selantennas = self.antennas.sel({self.dim[0]: concat_coord})

        # Get data
        data = [a.request_data(*args, **dropped_kwargs) for a in selantennas.data.ravel()]

        # Add coord for concatenation
        newdata = []
        for d, v in zip(data, concat_coord.data.ravel()):
            d = d.assign_coords({self.dim[0]: v})
            newdata.append(d)

        # Concatenate data along dimension
        data = xr.concat(newdata, dim=self.dim[0])

        return data


def concat(antennas, coord):
    # Create instance of OperatorFunction which stores self and other
    func = ConcatFunction(antennas, coord)
    # Create new antenna object with the operator function as the data input
    operated_ant = Antenna(func, antennas[0].hcs.reference)

    return operated_ant
