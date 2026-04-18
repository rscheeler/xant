from collections import OrderedDict
from dataclasses import dataclass, asdict
from itertools import islice
from typing import Union

import numpy as np
from pint import Quantity
from xarray import DataArray

from .. import ureg
from . import ber


def nf2temp(nf: Quantity | DataArray, t0: Quantity) -> Quantity | DataArray:
    """
    Converts noise figure to noise temperature.

    Parameters
    ----------
    nf : Quantity, DataArray
        Noise figure
    t0 : Quantity, DataArray
        Reference temperature

    Returns
    -------
    te : Quantity, DataArray
        Equivalent input noise temperature.
    """
    return t0 * (nf - 1)


@dataclass
class LinkBudget:
    """Calculate link budget based off input parameters. Will attempt to solve for parameters not given."""

    name: str
    pt: Union[Quantity, DataArray]
    gt: Union[Quantity, DataArray]
    gr: Union[Quantity, DataArray]
    nf: Union[Quantity, DataArray]
    distance: Union[Quantity, DataArray]
    frequency: Union[Quantity, DataArray]
    bandwidth: Union[Quantity, DataArray]
    data_rate: Union[Quantity, DataArray]
    ber_req: float
    modulation: str
    modulation_order: int
    t0: Union[Quantity, DataArray] = 290 * ureg.kelvin
    ta: Union[Quantity, DataArray] = 290 * ureg.kelvin
    additional_loss: Union[Quantity, DataArray] = 0 * ureg.dB
    coding_gain: Union[Quantity, DataArray] = 0 * ureg.dB

    def __repr__(self):
        table = f"| Parameter | {self.name} |\r\n"
        table += "|:--:|:--:|\r\n"
        rows = asdict(self)
        rows.pop("name")
        for k, v in rows.items():
            table += f"| {k} | {v}|\r\n"
        return table

    def _repr_markdown_(self):
        return self.__repr__()

    def view_link_table(self, link="tabular_margin", markdown=False):
        if markdown:
            table = f"| Parameter |{self.name} {link} | Unit |\r\n"
            table += f"|-:|:-:|:-|\r\n"
            link_dictionary = getattr(self, link)

            for k, v in islice(link_dictionary.items(), 0, len(link_dictionary) - 1):
                if not isinstance(v, Quantity):
                    v = Quantity(v * 1.0)
                table += f"|{k}|{v.magnitude:0.2f}|{v.units}|\r\n"

            table += f"|{list(link_dictionary.keys())[-1]}|{list(link_dictionary.values())[-1].magnitude:0.2f}|{list(link_dictionary.values())[-1].units}|\r\n"

        else:
            table = f"<table><caption>{self.name} {link} Link Budget</caption><thead><tr><th>Parameter</th><th>Value</th><th style='text-align:left'>Unit</th></tdhead></tr>\r\n<tbody>"

            link_dictionary = getattr(self, link)

            for k, v in islice(link_dictionary.items(), 0, len(link_dictionary) - 1):
                if not isinstance(v, Quantity):
                    v = Quantity(v * 1.0)
                table += f"<tr><th>{k}</th><td>{v.magnitude:0.2f}</td><td style='text-align:left'>{v.units}</td></tr>\r\n"

            table += f"<tr style='border-top: solid 1px'><th>{list(link_dictionary.keys())[-1]}</th><td>{list(link_dictionary.values())[-1].magnitude:0.2f}</td><td style='text-align:left'>{list(link_dictionary.values())[-1].units}</td></tr>\r\n"

            table += "</tbody></table>"
        return table

    @property
    def wavelength(self):
        """
        Return the wavelength in meters
        """
        return (1 / self.frequency * ureg.speed_of_light).to_base_units()

    @property
    def path_loss(self):
        """
        Return free space path loss based off of range. If range given calculate from range and wavelength. If not given, solve for.
        """
        if self.distance is None:
            raise NotImplementedError
        else:
            pl = (self.distance * 4 * np.pi / self.wavelength).to_base_units() ** 2
        return pl.to("dB")

    @property
    def eirp(self):
        """
        Returns the effective isotropic radiated power (EIRP) which is the transmit power times the transmit antenna
        gain.
        """
        return self.pt * self.gt

    @property
    def tsys(self):
        """
        Returns the system temperature. Note it is assumed that the antenna radiometric temperature is equal to the
        reference temperature t0, resulting in the following.
        .. math::
            T_{sys}=T_{A'}+T_{rec},
            T_{\mathrm{sys}} = T_0+T_0(F-1)=T_0 F


        """
        return self.ta + nf2temp(self.nf, self.t0)

    @property
    def rxnp(self):
        """
        Return the receiver sensitivity or noise power.
        """
        return (ureg.boltzmann_constant * self.tsys * self.bandwidth).to("dBm")

    @property
    def rxpwr(self):
        """
        Received power.
        """
        return (self.pt * self.gt * self.gr / (self.path_loss * self.additional_loss)).to("dBm")

    @property
    def minrxp(self):
        """
        Minimum received power.
        """
        return (self.pt * self.gt * self.gr / (self.max_path_loss * self.additional_loss)).to("dBm")

    @property
    def g_over_t(self):
        """
        Return the receiver gain over system temperature ratio.
        """
        return self.gr / self.tsys

    @property
    def cnr(self):
        """
        Returns the carrier-to-noise ratio
        """

        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.path_loss * ureg.boltzmann_constant * self.bandwidth * self.additional_loss)
        ).to("dB")

    @property
    def ebno(self):
        """
        Returns the energy per bit to noise power spectral density ratio
        """
        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.path_loss * ureg.boltzmann_constant * self.data_rate * self.additional_loss)
        ).to("dB")

    @property
    def ebno_req(self):
        """
        Returns the required energy per bit to noise power spectral density ratio to meet the link based off inputs.
        """
        return getattr(ber, f"m{self.modulation}_ebno")(self.modulation_order, self.ber_req)

    @property
    def link_margin(self):
        """
        Returns margin for specified link
        """
        return (self.ebno / self.ebno_req).to("dB")

    @property
    def ber(self):
        """
        Returns the bit-error-rate from the ebno
        """
        return getattr(ber, f"m{self.modulation}_ber")(self.modulation_order, self.ebno)

    @property
    def max_path_loss(self):
        """
        Returns the maximum path loss to still meet link
        """
        return (
            self.eirp
            * self.g_over_t
            * self.coding_gain
            / (self.ebno_req * ureg.boltzmann_constant * self.data_rate * self.additional_loss)
        ).to("dB")

    @property
    def max_link_distance(self):
        """
        Returns the maximum link distance
        """
        return (np.sqrt(self.max_path_loss.to_base_units()) * self.wavelength / (4 * np.pi)).to(
            "km"
        )

    @property
    def tabular_cnr(self):
        numer = dict(pt=self.pt.to("dBm"), gt=self.gt, gr=self.gr, coding_gain=self.coding_gain)
        denom = dict(
            path_loss=self.path_loss,
            additional_loss=self.additional_loss,
            k=(1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            tsys=self.tsys.to("decibelkelvin"),
            bandwidth=self.bandwidth.to("dBHz"),
        )
        return OrderedDict({**numer, **denom, **dict(cnr=dictionary_ratio(numer, denom).to("dB"))})

    @property
    def tabular_ebno(self):
        numer = dict(pt=self.pt.to("dBm"), gt=self.gt, gr=self.gr, coding_gain=self.coding_gain)
        denom = dict(
            path_loss=self.path_loss,
            additional_loss=self.additional_loss,
            k=(1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            tsys=self.tsys.to("decibelkelvin"),
            data_rate=self.data_rate.to("decibelhertz"),
        )
        return OrderedDict({**numer, **denom, **dict(ebno=dictionary_ratio(numer, denom).to("dB"))})

    @property
    def tabular_margin(self):
        numer = dict(pt=self.pt.to("dBm"), gt=self.gt, gr=self.gr, coding_gain=self.coding_gain)
        denom = dict(
            path_loss=self.path_loss,
            additional_loss=self.additional_loss,
            k=(1 * ureg.boltzmann_constant).to(ureg.dBm / ureg.kelvin / ureg.Hz),
            tsys=self.tsys.to("decibelkelvin"),
            data_rate=self.data_rate.to("decibelhertz"),
            required_ebno=self.ebno_req,
        )
        return OrderedDict(
            {**numer, **denom, **dict(margin=dictionary_ratio(numer, denom).to("dB"))}
        )

    def set_distance_from_margin(self, link_margin: Quantity):
        scale = self.link_margin / link_margin

        # Path loss is proportional to the square of the distance
        return self._replace(distance=self.distance * np.sqrt(scale))


def dictionary_ratio(numerator: dict, denominator: dict):
    """
    Calculate ratio from dictionary of numerator and denominator.

    Parameters
    ----------
    numerator : dict
    denominator : dict

    Returns
    -------
    ratio
    """

    numer = None
    for v in numerator.values():
        v = v.to_base_units()
        if numer is None:
            numer = v
        else:
            numer = numer * v

    denom = None
    for v in denominator.values():
        v = v.to_base_units()
        if denom is None:
            denom = v
        else:
            denom = denom * v

    return numer / denom
