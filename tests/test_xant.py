import pytest
from xant import Antenna
from pathlib import Path
_DIR = Path(__file__).parent
def test_load():
    test = Antenna(_DIR / "testdata/demo4-testiron.antnc")
    
if __name__ == "__main__":
    import numpy as np
    from xant import ureg
    from xant.plotting import plot_antenna_pattern
    import matplotlib.pyplot as plt
    import time
    test = Antenna(_DIR / "testdata/demo4-testiron.antnc")
    t0 = time.time()
    data = test.request_data(theta=np.linspace(-180,180,361)*ureg.degree,phi=[0,90]*ureg.degree)
    print(f"Data in {time.time()-t0} seconds")
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