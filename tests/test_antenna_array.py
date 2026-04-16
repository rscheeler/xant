import unittest

import numpy as np
from hics import HCS
from scipy.spatial.transform import Rotation

from xant import ureg
from xant.antenna.common import Dipole
from xant.antenna.phasedarray import AntennaArray


class TestArrays(unittest.TestCase):
    def setUp(self) -> None:
        # Create a coordinate system with assumed global reference
        self.mount = HCS(
            (0, 0, 0) * ureg.m,
            Rotation.from_euler("ZYZ", angles=[45, 90, 0], degrees=True),
        )
        # Frequency and wavelength
        self.fs = np.array([1, 2, 3]) * ureg.GHz
        self.lam = 1 / self.fs * ureg.speed_of_light

        # Create element pattern
        self.element = Dipole(5 * ureg.cm, self.fs)

    def test_array(self):
        dxy = self.lam[1] / 2
        n = 4
        arr = AntennaArray.rectangular(self.element, n, dxy, n, dxy, cs_reference=self.mount)

        a = arr.total.request_data(
            frequency=3 * ureg.GHz,
            phi=[0] * ureg.degree,
            theta=np.linspace(-90, 90, 181) * ureg.degree,
        )
        a = 20 * np.log10(abs(a))


if __name__ == "__main__":
    unittest.main()
