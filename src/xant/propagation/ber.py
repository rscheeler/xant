from typing import Union, Optional

import numpy as np
import xarray as xr
from pint import Quantity
from scipy.special import erfc, erfcinv

from .. import ureg

__all__ = ("mqam_ber", "mpsk_ber", "mqam_ebno", "mpsk_ebno")

DEFAULT_EBNO = xr.DataArray(
    np.linspace(0, 15, 101) * ureg.dB, dims=("ebno",), coords=dict(ebno=np.linspace(0, 15, 101))
)
DEFAULT_ORDER = xr.DataArray(
    [2, 3, 4, 6], dims=("modulation_order",), coords=dict(modulation_order=[2, 3, 4, 6])
)


def Q(x: Union[xr.DataArray, Quantity, float]) -> Union[xr.DataArray, Quantity, float]:
    """
    Q-Function which is the "tail distribution function of the standard normal distribution" [1] Expressed in terms of
    the error function.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Q-function
    """
    if isinstance(x, xr.DataArray):
        q = 0.5 * xr.apply_ufunc(erfc, x / np.sqrt(2), vectorize=True)
    else:
        if isinstance(x, Quantity):
            x = x.magnitude
        q = 0.5 * erfc(x / np.sqrt(2))
    return q


def Qinv(q: Union[xr.DataArray, float]) -> Union[xr.DataArray, float]:
    """
    Inverse of the Q-function.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Q-function#Inverse_Q
    """
    if isinstance(q, xr.DataArray):
        x = np.sqrt(2) * xr.apply_ufunc(erfcinv, 2 * q, vectorize=True)
    else:
        if isinstance(q, Quantity):
            q = q.magnitude
        x = np.sqrt(2) * erfcinv(2 * q)
    return x


def mqam_ber(
    p: Optional[Union[int, xr.DataArray]] = DEFAULT_ORDER,
    ebno: Optional[Union[Quantity, xr.DataArray]] = DEFAULT_EBNO,
):
    """
    Calculates the M-ary bit error rate (BER) for a quadrature amplitude modulation scheme (QAM) signal over
    an additive white Gaussian noise channel.

    Parameters
    ----------
    p : int, DataArray
        Modulation order
    ebno : Quantity, DataArray
        Energy per bit to noise power spectral density ratio

    References
    ----------
    Jianhua Lu, K. B. Letaief, J. C. . -I. Chuang and M. L. Liou, "M-PSK and M-QAM BER computation using signal-space
    concepts," in IEEE Transactions on Communications, vol. 47, no. 2, pp. 181-184, Feb. 1999, doi: 10.1109/26.752121.
    """
    # Convert ebno to base units
    if isinstance(ebno, xr.DataArray):
        ebno.data = ebno.data.to_base_units()
    else:
        ebno = ebno.to_base_units()
    # Calculate number of constellation points m
    m = 2**p
    qkern = np.sqrt(3 * np.log2(m) * ebno / (m - 1))

    if isinstance(m, xr.DataArray):
        qsums = []
        for mi in m:
            itmp = range(1, int(np.sqrt(mi) / 2) + 1)
            itmp = xr.DataArray(itmp, dims=("tmp"), coords=dict(tmp=itmp))
            iqkern = (2 * itmp - 1) * qkern.sel(modulation_order=mi.modulation_order)
            qsums.append(Q(iqkern).sum(dim="tmp"))
        qsum = xr.concat(qsums, "modulation_order")
    else:
        qsum = np.zeros_like(ebno)
        for i in range(1, int(np.sqrt(m) / 2) + 1):
            qsum += Q((2 * i - 1) * qkern)
    pb = 4 / np.log2(m) * (1 - 1 / np.sqrt(m)) * qsum
    # Update data array
    if isinstance(pb, xr.DataArray):
        pb.name = "BER"
        pb.attrs = {
            **pb.attrs,
            **dict(long_name="QAM Bit Error Rate", units="", description="QAM BER"),
        }
    return pb


def mpsk_ber(
    p: Optional[Union[int, xr.DataArray]] = DEFAULT_ORDER,
    ebno: Optional[Union[Quantity, xr.DataArray]] = DEFAULT_EBNO,
):
    """
    Calculates the M-ary bit error rate (BER) for a phase-shift keying scheme (PSK) signal over
    an additive white Gaussian noise channel.

    Parameters
    ----------
    p : int, DataArray
        Modulation order
    ebno : Quantity, DataArray
        Energy per bit to noise power spectral density ratio

    References
    ----------
    Jianhua Lu, K. B. Letaief, J. C. . -I. Chuang and M. L. Liou, "M-PSK and M-QAM BER computation using signal-space
    concepts," in IEEE Transactions on Communications, vol. 47, no. 2, pp. 181-184, Feb. 1999, doi: 10.1109/26.752121.
    """
    # Convert ebno to base units
    if isinstance(ebno, xr.DataArray):
        ebno.data = ebno.data.to_base_units()
    else:
        ebno = ebno.to_base_units()
    # Calculate number of constellation points m
    m = 2**p
    qkern = np.sqrt(2 * np.log2(m) * ebno) * np.sin(np.pi / m)
    pb = 2 / np.log2(m) * Q(qkern)
    # Update data array
    if isinstance(pb, xr.DataArray):
        pb.name = "BER"
        pb.attrs = {
            **pb.attrs,
            **dict(long_name="PSK Bit Error Rate", units="", description="PSK BER"),
        }
    return pb


def mpsk_ebno(p: int, ber: float):
    """
    Solve for the Eb/No required to meet the specified BER for a phase-shift keying scheme (PSK) signal over
    an additive white Gaussian noise channel.

    Parameters
    ----------
    p : int
        Modulation order
    ber : float
        Desired bit error rate

    References
    ----------
    Jianhua Lu, K. B. Letaief, J. C. . -I. Chuang and M. L. Liou, "M-PSK and M-QAM BER computation using signal-space
    concepts," in IEEE Transactions on Communications, vol. 47, no. 2, pp. 181-184, Feb. 1999, doi: 10.1109/26.752121.
    """
    # Calculate number of constellation points m
    m = 2**p
    if m == 2:
        ebno_lin = (Qinv(ber) / np.sin(np.pi / m)) ** 2 / (2 * np.log2(m))
    elif m >= 4:
        ebno_lin = (Qinv(ber * np.log2(m) / 2) / np.sin(np.pi / m)) ** 2 / (2 * np.log2(m))
    ebno = 10 * np.log10(ebno_lin) * ureg.dB
    return ebno


def mqam_ebno(p, ber):
    """
    Solve for the Eb/No required to meet the specified BER for a quadrature amplitude modulation scheme (QAM) signal
    over an additive white Gaussian noise channel.

    Parameters
    ----------
    p : int
        Modulation order
    ber : float
        Desired bit error rate

    References
    ----------
    Jianhua Lu, K. B. Letaief, J. C. . -I. Chuang and M. L. Liou, "M-PSK and M-QAM BER computation using signal-space
    concepts," in IEEE Transactions on Communications, vol. 47, no. 2, pp. 181-184, Feb. 1999, doi: 10.1109/26.752121.
    """

    # Coarse
    ebno_coarse = np.linspace(0, 50, 50001) * ureg.dB
    ebno_coarse = xr.DataArray(ebno_coarse, dims=("ebno",), coords=dict(ebno=ebno_coarse))
    qamber = mqam_ber(p, ebno_coarse)
    ebno = qamber.isel(ebno=abs(qamber - ber).argmin()).ebno

    # Fine
    ebno_fine = np.linspace(ebno.item() - 0.002, ebno.item() + 0.002, 40001) * ureg.dB
    ebno_fine = xr.DataArray(ebno_fine, dims=("ebno",), coords=dict(ebno=ebno_fine))
    qamber = mqam_ber(p, ebno_fine)
    ebno = qamber.isel(ebno=abs(qamber - ber).argmin()).ebno
    return ebno.item() * ureg.dB


if __name__ == "__main__":
    print(mqam_ber(2, 11.972055 * ureg.dB) == mpsk_ber(2, 11.972055 * ureg.dB))
    print(mqam_ebno(2, 1e-8), mpsk_ebno(2, 1e-8))
