"""
Fast total radiated power (TRP) and directivity.

Why this module exists
----------------------
Brute-force directivity needs ``∫|E|² dΩ`` on a grid fine enough to resolve the narrowest feature
of the pattern.  For an N-element array that means evaluating N element patterns (x 16
polarization bases) on a grid whose size grows with the array aperture.

This module replaces that with two ideas:

1. **Gauss-Legendre x uniform-phi quadrature** (:func:`gl_sphere`).  Exact for band-limited
   integrands, so the grid is sized from the *physical bandwidth* ``k*D`` of the antenna rather
   than guessed.  Roughly 5-10x fewer points than an equispaced rectangle rule for the same
   accuracy, and no aliasing of narrow beams.

2. **Gram-matrix quadratic form** for arrays (:func:`gram_identical`).
   With the array output ``E = sum_n w_n E_n / ||w||``::

       TRP(w) = w^H G w / ||w||^2,     G_mn = integral(E_m^H E_n dOmega)

   ``G`` depends on geometry/elements/frequency only.  Compute it once (cached), then *every*
   steering angle / taper / scan is an N x N matrix-vector product with **no angular grid**.
   For identical elements ``G_mn = integral(U_e(Omega) exp(j k d_mn . r_hat) dOmega)`` which
   depends only on the pair separation ``d_mn``; on a lattice only ~4N of the N^2 separations are
   unique, and the element pattern is evaluated once.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xrench.units import ureg

FOUR_PI = 4.0 * np.pi


class NoDataError(ValueError):
    """The antenna returned no data at the requested frequencies / angles (e.g. outside its pattern data)."""


_C0 = 299_792_458.0


# --------------------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------------------
def lmax_for(kd: float, l_element: int = 24) -> int:
    """
    Spherical-harmonic bandwidth needed to integrate ``|E|^2`` of a structure of extent ``d``.

    ``exp(j k d . r_hat)`` has negligible content above degree ``~ kd + a*(kd)^(1/3)``; the
    element power pattern adds its own (small) bandwidth ``l_element``.
    """
    kd = max(float(kd), 0.0)
    return int(np.ceil(kd + 6.0 * kd ** (1.0 / 3.0) + 10.0)) + int(l_element)


@dataclass(frozen=True)
class SphereGrid:
    """Tensor quadrature grid on the sphere (theta Gauss-Legendre, phi uniform)."""

    theta: np.ndarray  # (nt,) rad, ascending, in (0, pi)  -> poles excluded
    phi: np.ndarray  # (nph,) rad, in [-pi, pi)
    w_theta: np.ndarray  # (nt,) Gauss-Legendre weights (sum = 2)
    w_phi: float  # uniform phi weight (2 pi / nph)
    lmax: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.theta.size, self.phi.size)

    @property
    def weights(self) -> np.ndarray:
        """(nt, nph) quadrature weights; ``sum(weights * f)`` = ``integral f dOmega``."""
        return self.w_theta[:, None] * self.w_phi * np.ones(self.phi.size)[None, :]

    @property
    def unit_vectors(self) -> np.ndarray:
        """(nt, nph, 3) unit vectors r_hat."""
        st = np.sin(self.theta)[:, None]
        ct = np.cos(self.theta)[:, None]
        return np.stack(
            [
                st * np.cos(self.phi)[None, :],
                st * np.sin(self.phi)[None, :],
                ct * np.ones(self.phi.size)[None, :],
            ],
            axis=-1,
        )

    @property
    def spacing(self) -> float:
        """Characteristic angular node spacing [rad]."""
        return np.pi / (self.lmax + 1)

    def theta_slice(self, sl: slice) -> SphereGrid:
        return SphereGrid(self.theta[sl], self.phi, self.w_theta[sl], self.w_phi, self.lmax)


def gl_sphere(lmax: int, oversample: float = 1.0) -> SphereGrid:
    """
    Quadrature grid exact for spherical band-limit ``lmax`` (times ``oversample``).

    Gauss-Legendre with ``n`` nodes integrates polynomials in ``cos(theta)`` of degree
    ``2n-1`` exactly; the trapezoid rule in phi is exact for ``|m| < nphi``.
    """
    leff = int(np.ceil(lmax * oversample))
    nt = leff // 2 + 2
    nph = 2 * (leff // 2 + 2)  # even, >= leff + 2
    x, w = np.polynomial.legendre.leggauss(nt)  # ascending cos(theta)
    theta = np.arccos(x[::-1])  # ascending theta
    w_theta = w[::-1].copy()
    phi = -np.pi + (np.arange(nph) + 0.5) * (2.0 * np.pi / nph)
    return SphereGrid(theta, phi, w_theta, 2.0 * np.pi / nph, leff)


# --------------------------------------------------------------------------------------
# Generic antenna (request_data based)
# --------------------------------------------------------------------------------------
def _request_power(antenna, grid: SphereGrid, request_kwargs: dict, max_points: int = 400_000):
    """
    Evaluate ``|E|^2`` (apolar) on ``grid`` through ``Antenna.request_data``.

    Returns ``(U, other_dims, other_coords)`` with ``U`` shaped ``(*other, nt, nph)``.  Requests
    are chunked in theta so memory stays bounded for big grids.
    """
    nt, nph = grid.shape
    rows = max(1, min(nt, max_points // nph))
    chunks = []
    other = None
    coords = None
    for i0 in range(0, nt, rows):
        sub = grid.theta_slice(slice(i0, i0 + rows))
        dat = antenna.request_data(
            theta=sub.theta * ureg.radian,
            phi=sub.phi * ureg.radian,
            **request_kwargs,
        ).sel(polarization="apolar")
        if dat.size == 0:
            raise NoDataError("no pattern data at the requested frequencies")
        if "polarization" in dat.coords:
            dat = dat.drop_vars("polarization")
        if other is None:
            other = [d for d in dat.dims if d not in ("theta", "phi")]
            coords = {d: dat.coords[d] for d in other if d in dat.coords}
        dat = dat.transpose(*other, "theta", "phi")
        v = dat.data
        v = v.magnitude if hasattr(v, "magnitude") else v
        chunks.append(np.abs(np.asarray(v)) ** 2)
    return np.concatenate(chunks, axis=-2), other, coords


def trp_generic(
    antenna,
    request_kwargs: dict | None = None,
    rtol: float = 1e-4,
    lmax0: int = 48,
    lmax_cap: int = 1536,
):
    """
    Adaptive TRP of any ``Antenna``: quadrature resolution is doubled until TRP stops changing.

    Returns ``(trp, other_dims, coords)`` where ``trp`` has shape ``(*other,)``.  This is the slow,
    always-correct reference/fallback; arrays normally use the Gram-matrix path instead.
    """
    request_kwargs = request_kwargs or {}
    lmax = lmax0
    prev = None
    while True:
        grid = gl_sphere(lmax)
        U, other, coords = _request_power(antenna, grid, request_kwargs)
        trp = np.tensordot(U, grid.weights, axes=([-2, -1], [0, 1]))
        if prev is not None:
            err = np.max(np.abs(trp - prev) / np.maximum(np.abs(trp), 1e-300))
            if err < rtol or lmax >= lmax_cap:
                return trp, other, coords
        prev = trp
        lmax = min(2 * lmax, lmax_cap)


# --------------------------------------------------------------------------------------
# Array Gram matrices
# --------------------------------------------------------------------------------------
def unique_separations(pos: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Unique pair separations ``d[m, n] = r_n - r_m`` quantised to ``tol`` [m].

    Returns ``(unique (nu, 3), inverse (N, N))``.  On a regular lattice ``nu ~ 4N`` rather than
    ``N^2``.
    """
    n = pos.shape[0]
    d = pos[None, :, :] - pos[:, None, :]
    q = np.rint(d.reshape(-1, 3) / tol).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    return uq.astype(float) * tol, inv.reshape(n, n)


def gram_identical(
    pos: np.ndarray,
    k: float,
    ue: np.ndarray,
    grid: SphereGrid,
    tol: float | None = None,
    block_bytes: int = 256_000_000,
) -> np.ndarray:
    """
    Gram matrix ``G_mn = integral(U_e exp(j k (r_n - r_m) . r_hat) dOmega)`` for identical elements.

    Parameters
    ----------
    pos : (N, 3) element positions [m].
    k : wavenumber [rad/m].
    ue : (nt, nph) element power pattern ``|E_e|^2`` on ``grid`` (includes element directivity).
    """
    if tol is None:
        tol = 1e-7 / max(k, 1e-12)
    du, inv = unique_separations(pos, tol)
    u = grid.unit_vectors.reshape(-1, 3)
    wq = (grid.weights * ue).reshape(-1)
    nq = u.shape[0]
    g = np.zeros(du.shape[0], dtype=np.complex128)
    # block over quadrature nodes so the (nu x nq) phase matrix stays bounded in memory
    step = max(1, int(block_bytes // (16 * max(du.shape[0], 1))))
    for q0 in range(0, nq, step):
        ph = (k * du) @ u[q0 : q0 + step].T
        g += np.exp(1j * ph) @ wq[q0 : q0 + step]
    return g[inv]


def quad_form_trp(gram: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``w^H G w / ||w||^2`` for ``w`` shaped ``(N, S)``; returns ``(S,)`` (real)."""
    gw = gram @ w
    num = np.einsum("ns,ns->s", w.conj(), gw).real
    den = np.sum(np.abs(w) ** 2, axis=0)
    return num / den
