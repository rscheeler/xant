"""xant: Spatial antenna analysis with xarray."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xant")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0-dev"

from xrench.units import ureg

from .antenna.core import Antenna
from .config import XANTLogger

__all__ = ["Antenna", "XANTLogger", "ureg"]
