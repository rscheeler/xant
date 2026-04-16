import numpy as np
import xarray as xr
from hics import GLOBAL_CS, HCS
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from xant import ureg, Antenna
from xant.antenna.common import Dipole
from xant.antenna.phasedarray import AntennaArray, TranslatedPhase
from xant.plotting import plot_antenna_pattern

if __name__ == "__main__":
    test = Dipole(5 * ureg.cm, np.array([1, 2, 3]) * ureg.GHz)
    theta = xr.DataArray(
        np.linspace(0, 180, 181) * ureg.degree,
        dims=("theta",),
        coords=dict(theta=(np.linspace(0, 180, 181) * ureg.degree).to_base_units()),
        attrs=dict(units=ureg.degree),
    )
    phi = xr.DataArray(
        np.arange(-180, 180, 90) * ureg.degree,
        dims=("phi",),
        coords=dict(phi=(np.arange(-180, 180, 90) * ureg.degree).to_base_units()),
        attrs=dict(units=ureg.degree),
    )
    fs = np.array([2, 3]) * ureg.GHz
    frequency = xr.DataArray(fs, dims=("frequency",), coords=dict(frequency=fs.to_base_units()))
    a = test.request_data(frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=theta)
    a = 20 * np.log10(abs(a))
    a.sel(polarization="theta", frequency=3e9, phi=0).plot()

    b = test.data.antenna_callable(frequency=frequency, phi=phi, theta=theta)
    b = 20 * np.log10(abs(b))
    b.sel(polarization="theta", frequency=3e9, phi=0).plot(ls="--")

    # Try slicing and creating new object
    ds = test.request_data()
    ds.data = ds.data.magnitude
    test2 = Antenna(ds)
    c = test2.request_data(frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=theta)
    c = 20 * np.log10(abs(c))
    c.sel(polarization="theta", frequency=3e9, phi=0).plot(ls=":")

    hcs1 = HCS(
        (0, 0, 0) * ureg.m,
        Rotation.from_euler("ZYZ", angles=(0, 20, 0), degrees=True),
        reference=GLOBAL_CS,
    )
    d = test2.request_data(
        frequency=3 * ureg.GHz,
        phi=[0] * ureg.degree,
        theta=theta,
        coordinate_frame="phitheta",
        hcs=hcs1,
    )
    d = 20 * np.log10(abs(d))
    d.sel(polarization="theta", frequency=3e9, phi=0).plot()

    hcs2 = HCS((0, 0, 0) * ureg.m)
    test3 = Antenna(ds, hcs=hcs1)
    e = test3.request_data(
        frequency=3 * ureg.GHz,
        phi=[0] * ureg.degree,
        theta=theta,
        coordinate_frame="phitheta",
        hcs=hcs2,
    )
    e = 20 * np.log10(abs(e))
    e.sel(polarization="theta", frequency=3e9, phi=0).plot()

    e = test3.request_data(
        frequency=3 * ureg.GHz,
        phi=[0] * ureg.degree,
        theta=theta,
        coordinate_frame="phitheta",
        hcs=hcs2,
    )

    # Add a constant
    test4 = test * 2

    f = test4.request_data(
        frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=theta, coordinate_frame="phitheta"
    )
    f = 20 * np.log10(abs(f))
    f.sel(polarization="theta", frequency=3e9, phi=0).plot()

    # Add a pattern
    test5 = test + test3 * 0.22

    g = test5.request_data(
        frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=theta, coordinate_frame="phitheta"
    )
    g = 20 * np.log10(abs(g))
    g.sel(polarization="theta", frequency=3e9, phi=0).plot()

    # Translated Phase
    cs3 = HCS((5, 0, 0) * ureg.cm)
    tpf = TranslatedPhase(np.array([1, 2, 3]) * ureg.GHz, coordinate_systems=[cs3])
    cs4 = HCS((-5, 0, 0) * ureg.cm)
    tpf2 = TranslatedPhase(np.array([1, 2, 3]) * ureg.GHz, coordinate_systems=[cs4])

    # Translate dipole
    tdip = test * (tpf + tpf2)
    a = tdip.request_data(frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=theta)
    a = 20 * np.log10(abs(a))
    plt.figure()
    a.sel(polarization="apolar", frequency=3e9, phi=0).plot()

    # # AF
    # af = np.array(
    #     [
    #         TranslatedPhase(np.array([1, 2, 3]) * ureg.GHz, coordinate_system=CoordinateSystem((5 * i, 0, 0) * ureg.cm))
    #         for i in range(10)
    #     ]
    # ).sum()
    # a = af.request_data(frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=np.linspace(-90, 90, 181) * ureg.degree)
    # a = 20 * np.log10(abs(a))
    # plt.figure()
    # a.sel(polarization="apolar", frequency=3e9, phi=0).plot()
    # plot_antenna_pattern(a, "theta")
    # # Larger array
    # print("test")
    # gbl = CoordinateSystem((0, 0, 0) * ureg.m)
    # fs = np.array([1, 2, 3]) * ureg.GHz
    # cs = CoordinateSystem((0, 0, 0) * ureg.m, Rotation.from_euler("ZYZ", angles=[0, 90, 45], degrees=True))
    # dip = Dipole(5 * ureg.cm, fs, coordinate_system=cs)
    # a = dip.request_data(
    #     frequency=3 * ureg.GHz,
    #     phi=[0] * ureg.degree,
    #     theta=np.linspace(-90, 90, 181) * ureg.degree,
    #     coordinate_system=dip.coordinate_system.reference,
    # )
    # a = 20 * np.log10(abs(a))
    # plt.figure()
    # a.sel(polarization="apolar", frequency=3e9, phi=0).plot()
    #
    # d = 0.5 / fs[1] * ureg.speed_of_light
    # arr = AntennaArray.rectangular(dip, 8, d, 8, d, GLOBAL_CS)
    #
    # a = arr.total.request_data(
    #     frequency=3 * ureg.GHz, phi=[0] * ureg.degree, theta=np.linspace(-90, 90, 181) * ureg.degree
    # )
    # a = 20 * np.log10(abs(a))
    # plt.figure()
    # a.sel(polarization="apolar", frequency=3e9, phi=0).plot()
    # plt.gca().set_ylim(0, 40)
