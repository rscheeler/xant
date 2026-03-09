import unittest

import numpy as np
import xarray as xr
from hics import HCS
from scipy.spatial.transform import Rotation

from xant import ureg
from xant.common import Cardioid, Dipole, TE10Aperture


class TestOperators(unittest.TestCase):
    def setUp(self) -> None:
        self.antenna = Dipole(5 * ureg.cm, np.array([1, 2, 3]) * ureg.GHz)

    def test_add_float(self):
        other = 23.6

        with self.assertRaises(TypeError):
            self.antenna + other

    def test_mul_float(self):
        other = 7.45
        res = self.antenna * other

        origdata = self.antenna.request_data()
        resdata = res.request_data()
        np.testing.assert_array_almost_equal(origdata * other, resdata)

    def test_add_antenna(self):
        res = self.antenna + self.antenna

        origdata = self.antenna.request_data() + self.antenna.request_data()
        resdata = res.request_data()
        np.testing.assert_array_almost_equal(origdata, resdata)

    def test_mul_antenna(self):
        with self.assertRaises(TypeError):
            self.antenna * self.antenna


class TestOperatedOrthoSlice(unittest.TestCase):
    def setUp(self) -> None:
        fs = np.linspace(3550, 3800, 6) * ureg.MHz

        cs = HCS((0, 0, 0) * ureg.m)
        elem_hcs = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [45, 0, 0], degrees=True),
            reference=cs,
        )

        lam_l = 1.2 * ureg.speed_of_light / fs[0]

        l = lam_l / 2
        h = lam_l / 4

        self.ant1 = TE10Aperture(
            l,
            h * 16,
            fs,
            tau=0 * ureg.degree,
            hcs=elem_hcs,
            truncate_lower_hemisphere=False,
        )
        self.ant2 = Cardioid(fs, hcs=elem_hcs)
        self.ant3 = self.ant1 * self.ant2

        self.point = (0, 60.0) * ureg.degree

    def test_point(self):
        datap_az = self.ant2.request_data(
            phi=self.point[0],
            theta=self.point[1],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_el = self.ant2.request_data(
            theta=self.point[1],
            phi=self.point[0],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        xr.testing.assert_equal(datap_el, datap_az.transpose(*datap_el.dims))

    def test_slice_nomath(self):
        datap_azs = self.ant1.request_data(
            phi=[-1.0, 0.0, 1.0] * ureg.degree + self.point[0],
            theta=self.point[1],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_els = self.ant1.request_data(
            theta=[-1.0, 0.0, 1.0] * ureg.degree + self.point[1],
            phi=self.point[0],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_el = datap_els.sel(polarization="apolar").sel(theta=self.point[1].to("radian"))
        datap_az = datap_azs.sel(polarization="apolar").sel(phi=self.point[0].to("radian"))

        xr.testing.assert_equal(datap_el.squeeze(), datap_az.squeeze())

    def test_slice_ok(self):
        delt = 10.0
        datap_azs = self.ant3.request_data(
            phi=np.linspace(0.0, delt, 2) * ureg.degree + self.point[0],
            theta=self.point[1],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_els = self.ant3.request_data(
            theta=np.linspace(0.0, delt, 2) * ureg.degree + self.point[1],
            phi=self.point[0],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_el = datap_els.sel(polarization="apolar").sel(theta=self.point[1].to("radian"))
        datap_az = datap_azs.sel(polarization="apolar").sel(phi=self.point[0].to("radian"))

        xr.testing.assert_equal(datap_el.squeeze(), datap_az.squeeze())

    def test_slice_bad(self):
        delt = 10.0
        datap_azs = self.ant3.request_data(
            phi=np.linspace(-delt, delt, 3) * ureg.degree + self.point[0],
            theta=self.point[1],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_els = self.ant3.request_data(
            theta=np.linspace(-delt, delt, 3) * ureg.degree + self.point[1],
            phi=self.point[0],
            frequency=3.7 * ureg.GHz,
            coordinate_frame="phitheta",
        )

        datap_el = datap_els.sel(polarization="apolar").sel(theta=self.point[1].to("radian"))
        datap_az = datap_azs.sel(polarization="apolar").sel(phi=self.point[0].to("radian"))

        xr.testing.assert_equal(datap_el.squeeze(), datap_az.squeeze())


if __name__ == "__main__":
    unittest.main()
