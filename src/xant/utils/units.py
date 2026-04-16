from __future__ import annotations

from typing import TYPE_CHECKING

import pint

if TYPE_CHECKING:
    from pint import UnitRegistry


def get_ureg() -> UnitRegistry:
    """Retrieves or initializes the application-wide UnitRegistry."""
    # Get current application registry
    _ureg = pint.get_application_registry()

    # If the returned registry is the default 'internal' one it isn't cached
    if _ureg.cache_folder is None:
        # Create new with caching
        _ureg = pint.UnitRegistry(cache_folder=":auto:")

        # Set application registry
        pint.set_application_registry(_ureg)

    # Enforce required configuration settings
    _ureg.autoconvert_offset_to_baseunit = True
    _ureg.force_ndarray_like = True

    return _ureg


# Top-level instance for easy importing
ureg = get_ureg()
# Custom units - can be defined in a file too
# Decibel kelvin
ureg.define("decibelkelvin = 1.0 kelvin; logbase: 10; logfactor: 10 = dBK")
# Deibel hertz
ureg.define("decibelhertz = 1.0 Hz; logbase: 10; logfactor: 10 = dBHz")
