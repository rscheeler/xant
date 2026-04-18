import pytest
from pint.testing import assert_allclose

from xant import ureg
from xant.propagation.itm.p2108 import p2108_2


def test_p2108_2():
    result = p2108_2(3 * ureg.GHz, 0.5 * ureg.km, 0.5)

    expected = 26.6281195 * ureg.dB
    # Assert equality
    assert_allclose(result, expected, atol=1e-7)
