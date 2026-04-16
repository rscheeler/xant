import unittest

import numpy as np
import xarray as xr

from xant import ureg
from xant.utils import conversions


class TestSpatialTransforms(unittest.TestCase):
    """
    Testing of the coordinate frame transforms.
    """

    def test_phitheta2uvw2phitheta_phi(self):
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

        phi, theta = xr.broadcast(phi, theta)
        phix, thetax = conversions.uvw2phitheta(*conversions.phitheta2uvw(phi, theta))

        np.testing.assert_array_almost_equal(phi.values, phix.data.to("degrees"))

    def test_phitheta2uvw2phitheta_theta(self):
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

        phi, theta = xr.broadcast(phi, theta)
        phix, thetax = conversions.uvw2phitheta(*conversions.phitheta2uvw(phi, theta))

        np.testing.assert_array_almost_equal(theta.values, thetax.data.to("degrees"))


if __name__ == "__main__":
    unittest.main()
