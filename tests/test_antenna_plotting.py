import unittest
from datetime import datetime

import numpy as np
import xarray as xr
from hics import HCS
from scipy.spatial.transform import Rotation

from xant import ureg
from xant.common import TE10Aperture
from xant.plotting import plot_antenna_pattern


class TestAntennaPlot(unittest.TestCase):
    def setUp(self) -> None:
        # Setup the time dependent coordinate system
        # Map center location in Geodetic coordinates
        # Estimate for roof of NASCTN Lab
        lla = [40.015 * ureg.degree, -105.270556 * ureg.degree, 0 * ureg.m]

        nasctn = HCS.from_crs(lla, un="h", ux="lon", hagl=True)
        tower = HCS(
            (0, 0, 3) * ureg.m,
            reference=nasctn,
        )
        # Generate time array
        n = 51
        times = np.datetime64(datetime.now().isoformat()) + (np.arange(n) * 1e6).astype(np.int32)

        # Use same position for each time
        pos = [(i * 10, 0, 0) for i in range(n)] * ureg.m

        # Make into DataArrays
        pos_xr = xr.DataArray(
            pos,
            dims=("time", "position"),
            coords=dict(
                time=times,
                position=["x", "y", "z"],
            ),
        )
        rot_xr = xr.DataArray(
            [
                Rotation.from_euler("ZYZ", [0, 90, -ang], degrees=True)
                for ang in np.linspace(0, 360, n)
            ],
            dims=("time",),
            coords=dict(time=times),
        )

        # Create a time dependent CoordinateSystem
        mount_temporal = HCS(pos_xr, rotation=rot_xr, reference=tower)

        # Frequencies to analyze - note they are pint Quantities
        fs = np.array([37, 38, 39, 40]) * ureg.GHz
        self.f0 = fs[0]
        # Derive wavelength also as a pint Quantity
        lam = 1 / fs * ureg.speed_of_light

        lam0 = lam[0]

        self.patt_slice = np.linspace(-180, 180, 361) * ureg.degree

        a = 4 * lam0
        b = lam0
        txant = TE10Aperture(a, b, fs, hcs=mount_temporal)
        txant *= 4
        self.antenna = txant
        self.data = self.antenna.request_data(
            theta=self.patt_slice,
            phi=[0] * ureg.degree,
            frequency=self.f0,
            hcs=nasctn,
        )

    def test_polar(self):
        ax = plot_antenna_pattern(self.data.isel(time=0).sel(polarization="apolar"), "theta")

    def test_rect(self):
        ax = plot_antenna_pattern(
            self.data.isel(time=0).sel(polarization="apolar"), "theta", projection="rectilinear"
        )

    def test_movie(self):
        ax = plot_antenna_pattern(self.data.sel(polarization="apolar"), "theta", animate="time")


if __name__ == "__main__":
    unittest.main()
