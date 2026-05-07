from pathlib import Path

import h5py
import pytest

from xant import Antenna

_DIR = Path(__file__).parent


def test_load():
    test = Antenna(_DIR / "testdata/demo4-testiron.xant")


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import numpy as np

    from xant import ureg
    from xant.plotting import plot_antenna_pattern

    test = Antenna(_DIR / "testdata/demo4-testiron.xant")
    t0 = time.time()
    data = test.request_data(
        theta=np.linspace(-180, 180, 361) * ureg.degree,
        phi=[0, 90] * ureg.degree,
    )
    print(f"Data in {time.time() - t0} seconds")
    # t0 = time.time()
    # data = test.request_data(theta=np.linspace(-180,180,361)*ureg.degree,phi=[0,90]*ureg.degree)
    # print(f"Data in {time.time()-t0} seconds")
    # t0 = time.time()
    # data = test.request_data(theta=np.linspace(-180,180,361)*ureg.degree,phi=[0,90]*ureg.degree)
    # print(f"Data in {time.time()-t0} seconds")
    # ax = plot_antenna_pattern(
    #         data.sel(frequency=1.575e9,polarization="apolar"), "theta", projection="rectilinear"
    #     )
    # ax.grid()
    # ax2 = plot_antenna_pattern(
    #         data.sel(frequency=1.575e9,polarization="apolar"), "theta", projection="polar",yspan=33
    #     )

    # plt.show()
