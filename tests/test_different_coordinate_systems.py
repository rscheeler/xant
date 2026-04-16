import unittest

import numpy as np
from hics import GLOBAL_CS, HCS, ureg
from scipy.spatial.transform import Rotation

from xant.antenna.common import TE10Aperture


class TestRotatedCS(unittest.TestCase):
    def setUp(self) -> None:
        self.mount = HCS((0, 0, 2.5) * ureg.m, reference=GLOBAL_CS)
        self.mount2 = HCS(
            (0, 0, 0) * ureg.m,
            reference=self.mount,
            rotation=Rotation.from_euler("ZYZ", [45, 0, 0], degrees=True),
        )
        # Frequencies to analyze - note they are pint Quantities
        fs = np.array([37]) * ureg.GHz

        # Derive wavelength also as a pint Quantity
        lam = 1 / fs * ureg.speed_of_light

        a = 68.8 * lam[-1] / 98  # 98 degree beamwidth
        b = 50.6 * lam[-1] / 18  # 18 degree beamwidth
        # self.antenna = TE10Aperture(a, b, fs, hcs=self.mount) * 10 ** ((11.8 - 13.25) / 20)
        self.antenna = TE10Aperture(a, b, fs, hcs=self.mount)
        # Spatial slice
        self.patt_slice = [45] * ureg.degree

    def test_apolar_diag(self) -> None:
        phi_slc = [45] * ureg.degree
        d0 = self.antenna.request_data(theta=self.patt_slice, phi=phi_slc)
        d1 = self.antenna.request_data(
            theta=self.patt_slice, phi=phi_slc + 45 * ureg.degree, hcs=self.mount2
        )

        np.testing.assert_array_almost_equal(
            d0.sel(polarization="apolar").data, d1.sel(polarization="apolar").data
        )

    def test_apolar_boresight(self) -> None:
        phi_slc = [-45] * ureg.degree
        th_slc = [0] * ureg.degree
        d0 = self.antenna.request_data(theta=th_slc, phi=phi_slc, hcs=self.mount)
        d1 = self.antenna.request_data(
            theta=th_slc, phi=phi_slc + 45 * ureg.degree, hcs=self.mount2
        )

        np.testing.assert_array_almost_equal(
            d0.sel(polarization="apolar").data, d1.sel(polarization="apolar").data
        )


if __name__ == "__main__":
    unittest.main()
