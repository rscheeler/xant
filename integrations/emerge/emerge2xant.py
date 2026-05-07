from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from pint import Quantity
from scipy import constants
from xrench.units import ureg

from xant import Antenna

_eta0 = np.sqrt(constants.mu_0 / constants.epsilon_0)
if TYPE_CHECKING:
    from emerge import Simulation
    from emerge._emerge.geometry import GeoSurface
    from emerge._emerge.physics.microwave.microwave_data import MWData
    from emerge._emerge.selection import FaceSelection


def emerge2xr(
    model: Simulation,
    mwdata: MWData,
    faces: FaceSelection | GeoSurface,
    thetas: Quantity,
    phis: Quantity,
    element_pattern: bool = True,
    export: bool | None = None,
) -> Antenna:
    """Export emerge model to an antenna object and export data. Will export embedded elements if desired."""
    if element_pattern:
        epatts = []
        # Get ports and setup zeros excitation
        ports = list(mwdata.field[0].excitation.keys())
        ports.sort()
        zeros_excitation = dict.fromkeys(ports, 0 + 1j * 0)
        for i in ports:
            # Set excitation
            excitation = zeros_excitation.copy()
            excitation[i] = 1
            # Set excitation at each frequency
            for fld in mwdata.field:
                fld.excitation = excitation
            # Get data
            epatt = _emerge2xrworker(mwdata, faces, thetas, phis)
            # Add port dimension
            epatt = epatt.expand_dims(port=[i])
            epatts.append(epatt)
        xrdat = xr.concat(epatts, "port")

    else:
        xrdat = _emerge2xrworker(mwdata, faces, thetas, phis)
    ant = Antenna(xrdat)
    if export:
        ant.export(model.modelpath / model.modelname)

    return ant


def _emerge2xrworker(
    mwdata: MWData,
    faces: FaceSelection | GeoSurface,
    thetas: Quantity,
    phis: Quantity,
) -> xr.DataArray:
    # Iterate ports
    dat = []
    freqs = []
    for f in mwdata.field:
        freqs.append(f.freq)
        ff3d = f.farfield_3d(
            faces,
            thetas=thetas.to_base_units().magnitude,
            phis=phis.to_base_units().magnitude,
        )
        dat.append(ff3d.E.F)

    # Convert to array and scale to gain field quantity
    # G=4piU/Pinc, U=r|E|^2/(2eta)
    dat = np.array(dat) * np.sqrt(2 * np.pi / _eta0)

    # Collect dims and coords
    coords = {
        **dict(frequency=np.array(freqs) * ureg.Hz),
        **dict(
            polarization=["x", "y", "z"],
            phi=phis.to_base_units().magnitude,
            theta=thetas.to_base_units().magnitude,
        ),
    }
    dims = [k for k in coords]
    xrdat = xr.DataArray(dat, dims=dims, coords=coords, attrs=dict(coordinate_frame="phitheta"))

    return xrdat
