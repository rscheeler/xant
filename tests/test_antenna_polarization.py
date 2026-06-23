import unittest

import numpy as np
import xarray as xr
from hics import HCS
from scipy.spatial.transform import Rotation
from xrench.xrutils import vector_norm

from xant import ureg
from xant.antenna.common import DipoleAboveGround, TE10Aperture
from xant.antenna.phasedarray import AntennaArray


class TestArrayTotalPowerPanel(unittest.TestCase):
    def setUp(self) -> None:
        f0 = 1500 * ureg.MHz
        fs = np.linspace(1400, 1600, 5) * ureg.MHz
        lam0 = ureg.speed_of_light / f0

        hcs = HCS((0, 0, 0) * ureg.m)
        elem_hcs = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [45, 0, 0], degrees=True),
            reference=hcs,
        )

        lam_l = 1.2 * ureg.speed_of_light / fs[0]

        l = lam_l / 2
        h = lam_l / 4
        element = DipoleAboveGround(l, h, fs, hcs=elem_hcs)

        nz = 2
        dz = lam0 / 2

        arr_cs = HCS((0, 0, 0) * ureg.m)
        coordsyss = [
            HCS(
                (0, 0, dz.magnitude * n) * ureg.inch,
                rotation=Rotation.from_euler("ZYZ", [45, -90, 0], degrees=True),
                reference=arr_cs,
            )
            for n in range(nz)
        ]
        self.arr_p45 = AntennaArray(element, coordsyss)

    def test_apolar(self):
        theta = [0, 21, 36, 77] * ureg.degree
        phi = [0, 22, 66, 82] * ureg.degree
        frequency = 3.7 * ureg.GHz

        data = self.arr_p45.total.request_data(theta=theta, phi=phi, frequency=frequency)
        data_db = 20 * np.log10(abs(data.sel(polarization="apolar")))
        data_db = data_db.drop_vars("polarization")
        apol = 20 * np.log10(
            abs(vector_norm(data.sel(polarization=["theta", "phi"]), "polarization")),
        )

        xr.testing.assert_allclose(data_db, apol)


class TestArrayTotalPowerOmni(unittest.TestCase):
    def setUp(self) -> None:
        f0 = 1500 * ureg.MHz
        fs = np.linspace(1400, 1600, 5) * ureg.MHz
        lam0 = ureg.speed_of_light / f0

        hcs = HCS((0, 0, 0) * ureg.m)
        elem_hcs = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [45, 0, 0], degrees=True),
            reference=hcs,
        )

        lam_l = 1.2 * ureg.speed_of_light / fs[0]

        l = lam_l / 2
        h = lam_l / 4
        element = DipoleAboveGround(l, h, fs, hcs=elem_hcs)

        dz = lam0 / 2
        nz = 2
        r = 1.0

        arr_hcs = HCS((0, 0, 0) * ureg.m)

        azs = np.arange(0, 360, 90)

        coordsyss = []
        for n in range(nz):
            for az in azs:
                coordsyss.append(
                    HCS(
                        (
                            r * np.cos(np.deg2rad(az)),
                            r * np.sin(np.deg2rad(az)),
                            dz.magnitude * n,
                        )
                        * ureg.inch,
                        rotation=Rotation.from_euler("ZYZ", [45, -90, -az], degrees=True),
                        reference=arr_hcs,
                    ),
                )

        self.omni_p45 = AntennaArray(element, coordsyss)

    def test_apolar(self):
        theta = [0, 21, 36, 77] * ureg.degree
        phi = [0, 22, 66, 82] * ureg.degree
        theta = [36] * ureg.degree
        phi = [66] * ureg.degree
        frequency = 1.5 * ureg.GHz

        data = self.omni_p45.total.request_data(theta=theta, phi=phi, frequency=frequency)
        data_db = 20 * np.log10(abs(data.sel(polarization="apolar")))
        data_db = data_db.drop_vars("polarization")
        apol = 20 * np.log10(
            abs(vector_norm(data.sel(polarization=["theta", "phi"]), "polarization")),
        )

        xr.testing.assert_allclose(data_db, apol)


class TestPolarizationRotation(unittest.TestCase):
    def setUp(self) -> None:
        # Create TX CS
        self.base = HCS((0, 0, 0) * ureg.m)
        self.tower = HCS((0, 0, 10) * ureg.m, reference=self.base)
        self.txzr = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [0, -89.8, 0.2], degrees=True),
            reference=self.tower,
        )
        self.txcs = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [0, -90, 0], degrees=True),
            reference=self.tower,
        )

        # Create tx and rx antennas
        fs = np.array([1.5]) * ureg.GHz
        lam = (ureg.speed_of_light / fs).to("m")
        self.tx = TE10Aperture(lam[0] * 32, lam[0] * 8, fs, hcs=self.txcs)

    def test_rotated_polarization(self):
        tx_gain1 = self.tx.request_data(theta=0 * ureg.degree, phi=0 * ureg.degree)
        tx_gain2 = self.tx.request_data(theta=0 * ureg.degree, phi=0 * ureg.degree, hcs=self.txzr)

        xr.testing.assert_allclose(tx_gain1, tx_gain2, atol=1.0)

    def test_simple(self):
        txzr = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [90, -90, 0], degrees=True),
            reference=self.tower,
        )
        txcs = HCS(
            (0, 0, 0) * ureg.m,
            rotation=Rotation.from_euler("ZYZ", [0, -90, 0], degrees=True),
            reference=self.tower,
        )

        # Create tx and rx antennas
        fs = np.array([1.5]) * ureg.GHz
        lam = (ureg.speed_of_light / fs).to("m")
        tx = TE10Aperture(lam[0] * 32, lam[0] * 8, fs, hcs=txcs)

        txg1 = tx.request_data(theta=0 * ureg.degree, phi=0 * ureg.degree)
        txg2 = tx.request_data(theta=0 * ureg.degree, phi=90 * ureg.degree, hcs=txzr)

        tx_gain1 = txg1.sel(polarization=["theta", "phi"]).drop_vars("phi")
        tx_gain2 = txg2.sel(polarization=["theta", "phi"]).drop_vars("phi")

        xr.testing.assert_allclose(tx_gain1, tx_gain2)

        tx_gain1 = txg1.sel(polarization=["y"]).drop_vars("phi").drop_vars("polarization")
        tx_gain2 = txg2.sel(polarization=["x"]).drop_vars("phi").drop_vars("polarization")

        xr.testing.assert_allclose(tx_gain1, -1 * tx_gain2)


if __name__ == "__main__":
    unittest.main()
