from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import constants

from xant.antenna.core import Antenna

# Free-space wave impedance η = sqrt(μ₀/ε₀), computed once at import time.
_ETA: float = float(np.sqrt(constants.mu_0 / constants.epsilon_0))  # ≈ 376.73 Ω
# Conversion scalar: sqrt(2π/η), dimensionless after treating sr as dimensionless.
_SCALE: float = float(np.sqrt(2 * np.pi / _ETA))  # units: 1/√V (cancels with rE·V)


def read_ffd(path: str | Path, p_in_watts: float = 1.0) -> Antenna:
    """
    Reader for Ansys HFSS Far-Field Data (.ffd) files.

    Parses into an xr.DataArray with dimensions:
        (frequency, theta, phi, polarization)

    where polarization = ["E_theta", "E_phi"] and values are the complex
    gain field sqrt(G̃), defined as:

        sqrt(G̃) = rE · sqrt(2π / η) / sqrt(P_in)

    with η = sqrt(μ₀/ε₀) (free-space wave impedance) and P_in = 1 W.
    Solid angle is treated as dimensionless so the result is dimensionless,
    stored as a pint Quantity duckarray inside the DataArray.


    Parameters
    ----------
    path:
        Path to the .ffd file (must have a ``.ffd`` extension).
    p_in_watts:
        Input power in watts used to normalise the gain field.
        Defaults to 1 W (HFSS convention).

    Returns:
    -------
    Antenna

    Raises:
    ------
    ValueError
        Bad file extension or unparseable content.
    FileNotFoundError
        File not found.
    """
    return Antenna(_ffd2xr(path, p_in_watts))


def _ffd2xr(path: str | Path, p_in_watts: float = 1.0) -> xr.DataArray:
    """
    Reader for Ansys HFSS Far-Field Data (.ffd) files.

    Parses into an xr.DataArray with dimensions:
        (frequency, theta, phi, polarization)

    where polarization = ["E_theta", "E_phi"] and values are the complex
    gain field sqrt(G̃), defined as:

        sqrt(G̃) = rE · sqrt(2π / η) / sqrt(P_in)

    with η = sqrt(μ₀/ε₀) (free-space wave impedance) and P_in = 1 W.
    Solid angle is treated as dimensionless so the result is dimensionless,
    stored as a pint Quantity duckarray inside the DataArray.


    Parameters
    ----------
    path:
        Path to the .ffd file (must have a ``.ffd`` extension).
    p_in_watts:
        Input power in watts used to normalise the gain field.
        Defaults to 1 W (HFSS convention).

    Returns:
    -------
    xr.DataArray
        Complex gain-field array ``sqrt(G̃)``, dims
        ``(frequency, theta, phi, polarization)``.
        The underlying duckarray is a dimensionless ``pint.Quantity``.
        Frequency in Hz, angles in degrees.

    Raises:
    ------
    ValueError
        Bad file extension or unparseable content.
    FileNotFoundError
        File not found.
    """
    path = Path(path)
    if path.suffix.lower() != ".ffd":
        raise ValueError(
            f"Expected a .ffd file, got extension '{path.suffix}': {path}",
        )
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return _parse_ffd(path, p_in_watts)


def _parse_ffd(path: Path, p_in_watts: float) -> xr.DataArray:
    # ------------------------------------------------------------------
    # 1. Slurp + split in one shot — fast for files up to several GB.
    # ------------------------------------------------------------------
    tokens = path.read_bytes().decode("ascii", errors="replace").split()
    pos = 0  # index into tokens

    def take(n: int = 1):
        nonlocal pos
        chunk = tokens[pos : pos + n]
        pos += n
        return chunk

    def take1():
        return take(1)[0]

    # ------------------------------------------------------------------
    # 2. Header
    # ------------------------------------------------------------------
    theta_start, theta_stop = float(take1()), float(take1())
    theta_npts = int(take1())

    phi_start, phi_stop = float(take1()), float(take1())
    phi_npts = int(take1())

    take1()  # literal "Frequencies"
    num_freqs = int(take1())

    theta_coords = np.linspace(theta_start, theta_stop, theta_npts)
    phi_coords = np.linspace(phi_start, phi_stop, phi_npts)
    n_spatial = theta_npts * phi_npts  # data points per freq block

    # ------------------------------------------------------------------
    # 3. Pre-allocate
    # ------------------------------------------------------------------
    data = np.empty((num_freqs, theta_npts, phi_npts, 2), dtype=np.complex128)
    freq_values = np.empty(num_freqs, dtype=np.float64)

    # ------------------------------------------------------------------
    # 4. Parse each frequency block with bulk NumPy conversion.
    # ------------------------------------------------------------------
    for f_idx in range(num_freqs):
        take1()  # literal "Frequency"
        freq_values[f_idx] = float(take1())

        n_vals = n_spatial * 4
        raw_vals = np.array(take(n_vals), dtype=np.float32)  # bulk convert
        raw_vals = raw_vals.reshape(n_spatial, 4)

        # cols: [E_theta_re, E_theta_im, E_phi_re, E_phi_im]
        e_theta = (raw_vals[:, 0] + 1j * raw_vals[:, 1]).reshape(theta_npts, phi_npts)
        e_phi = (raw_vals[:, 2] + 1j * raw_vals[:, 3]).reshape(theta_npts, phi_npts)

        data[f_idx, :, :, 0] = e_theta
        data[f_idx, :, :, 1] = e_phi

    # ------------------------------------------------------------------
    # 5. Convert rE to complex gain field: sqrt(G̃) = rE · sqrt(2π/η) / sqrt(P_in)
    #    All physical units cancel; treat result as dimensionless pint Quantity.
    # ------------------------------------------------------------------
    sqrt_p_in = float(np.sqrt(p_in_watts))  # √(W) — scalar
    gain_field = data * _SCALE / sqrt_p_in

    # ------------------------------------------------------------------
    # 6. Build DataArray with pint duckarray
    # ------------------------------------------------------------------
    return xr.DataArray(
        gain_field,
        dims=["frequency", "theta", "phi", "polarization"],
        coords={
            "frequency": ("frequency", freq_values),
            "theta": ("theta", np.deg2rad(theta_coords)),
            "phi": ("phi", np.deg2rad(phi_coords)),
            "polarization": ("polarization", ["theta", "phi"]),
        },
        attrs={
            "source": str(path),
            "coordinate_frame": "phitheta",
            "p_in": p_in_watts,
        },
    )
