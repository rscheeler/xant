import numpy as np
import pytest
from hics import HCS
from pint.testing import assert_allclose
from xrench.units import ureg

from xant.antenna.common import Isotropic
from xant.propagation.rflink import (
    calculate_spatial_link,
    plot_transmit_pd,
    transmit_power_density,
    view_link_horizon,
)

lats = np.linspace(40.00, 40.04, 50) * ureg.degree
lons = np.linspace(-105.30, -105.20, 50) * ureg.degree
hs = [0] * ureg.m
freq = np.array([2.4e9]) * ureg.Hz


@pytest.fixture(scope="module")
def setuptxrx_apolar():

    # Position TX and RX using geodetic coordinates
    tx_cs = HCS.from_crs((40.015 * ureg.degree, -105.27 * ureg.degree, 30 * ureg.m), hagl=True)
    rx_cs = HCS.from_crs((40.020 * ureg.degree, -105.24 * ureg.degree, 30 * ureg.m), hagl=True)

    tx = Isotropic(frequency=freq, hcs=tx_cs)
    rx = Isotropic(frequency=freq, hcs=rx_cs)
    return tx, rx


@pytest.fixture(scope="module")
def get_linkres(setuptxrx_apolar):
    tx, rx = setuptxrx_apolar
    # rx_power_fspl, prop_loss_fspl, incident_pol, txcs, rxcs
    res_fspl = calculate_spatial_link(
        tx,
        power=1 * ureg.watt,
        rx=rx,
        propagation="fspl",
    )

    res_itm = calculate_spatial_link(
        tx,
        power=1 * ureg.watt,
        rx=rx,
        propagation="itm_rflink",
        gnd="good",  # ground conductivity: "poor", "average", "good", "fresh_water"
        time=[50],  # reliability percentile
        situation=[50],  # confidence percentile
    )
    return tx, rx, res_fspl, res_itm


def test_apolar_prop(get_linkres):
    tx, rx, res_fspl, res_itm = get_linkres
    assert_allclose(res_fspl[1].data, res_itm[1].fspl.data, rtol=1e-3)
    assert_allclose(res_fspl[0].data, res_itm[0].data, rtol=1e-3)


def test_plot_pd(setuptxrx_apolar):

    pd_map = transmit_power_density(
        setuptxrx_apolar[0],
        power=1 * ureg.watt,
        lats=lats,
        lons=lons,
        hs=hs,
        propagation="fspl",
    )

    plot_transmit_pd(pd_map)


def test_plot_horizon(get_linkres):
    tx, rx, res_fspl, res_itm = get_linkres

    ax = view_link_horizon(tx, rx, res_itm[0], res_itm[2], res_itm[1], gsize=0.02)
