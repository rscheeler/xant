"""
Whole-sphere TRP, directivity and gain normalisation for :class:`~xant.antenna.phasedarray.AntennaArray`.

Power radiated into grating lobes lowers the gain of the main beam, so TRP must be integrated over the
*whole sphere*.  With ``E = sum_n w_n E_n / ||w||`` the radiated power is a quadratic form::

    TRP(w) = w^H G w / ||w||^2,        G_mn = integral(E_m^H E_n dOmega)

``G`` depends on geometry, elements and frequency only, so it is computed once (cached) and every
steering angle / taper is then a matrix-vector product.

For *identical* elements ``G_mn = integral(U_e exp(j k d_mn . r_hat) dOmega)``.  Hierarchies of identical
sub-arrays are handled by two such models, fastest first:

* **flat**: the whole hierarchy is one flat array of the leaf element at positions ``r_top + r_sub`` with
  product weights.  G never depends on any weight, so re-steering (sub-arrays included) is free.
* **identical**: an identical sub-array, at its current weights, is just "the element".  Nests by recursion;
  the cache key includes the sub-array weights, so re-steering a sub-array rebuilds G.

Robustness: every fast model checks itself against the real pattern at a few directions and falls back
instead of returning a wrong number::

    flat -> identical -> per-port fields (rotated / non-identical elements) -> adaptive quadrature
"""

from __future__ import annotations

import operator as opr

import numpy as np
import xarray as xr
from hics import HCS
from loguru import logger
from xrench.units import ureg

from . import directivity as _dir
from .core import Antenna, OperatorFunction
from .directivity import NoDataError

_CACHE_MAX = 64  # bounded: each entry is an (N, N) matrix or an (nt, nph) pattern


def _mag(da: xr.DataArray) -> np.ndarray:
    v = da.data
    return np.asarray(v.magnitude if hasattr(v, "magnitude") else v)


class ArrayDirectivityMixin:
    """Mixed into ``AntennaArray``; needs ``element``, ``coordinate_systems``, ``excitation``, ``frequency``,
    ``elements``, ``_raw_total``, ``normalize_total`` and the ``_gram_cache`` / ``_steer_info`` attributes.
    """

    # ------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------
    @property
    def reference_cs(self) -> HCS:
        """Coordinate system in which element positions (and array angles) are expressed."""
        return self.coordinate_systems[0].reference

    def clear_cache(self) -> None:
        """Drop cached Gram matrices (call if an element pattern was modified in place)."""
        self._gram_cache = {}

    def _cache_put(self, key, value):
        if len(self._gram_cache) >= _CACHE_MAX:
            self._gram_cache.pop(next(iter(self._gram_cache)))
        self._gram_cache[key] = value
        return value

    def _freqs_hz(self, frequency=None) -> np.ndarray:
        if frequency is None:
            frequency = self.frequency.data
        if isinstance(frequency, xr.DataArray):
            frequency = frequency.data
        f = np.asarray(frequency.to("Hz").magnitude, dtype=float).reshape(-1)
        _, idx = np.unique(
            f,
            return_index=True,
        )  # nested requests can stack/duplicate frequency dims
        return f[np.sort(idx)]

    def _element_positions(self) -> np.ndarray:
        """(N, 3) element positions [m] in the array reference frame (same data TranslatedPhase uses)."""
        pos = []
        for cs in self.coordinate_systems:
            v = np.asarray(cs.origin.basemag.values, dtype=float)
            if v.shape != (3,):
                raise NotImplementedError("Fast TRP requires static (non time-varying) positions.")
            pos.append(v)
        return np.array(pos)

    def _identical_element(self) -> Antenna | None:
        """The single Antenna every port shares: a plain element, or an identical sub-array (None otherwise)."""
        if isinstance(self.element, Antenna):
            # An element carrying its own 'port' dimension is aligned per-port with the array
            # (not one shared element), so it must take the per-port path.
            return None if "port" in self.element.data.dims else self.element
        if isinstance(self.element, ArrayDirectivityMixin):
            return self.element._raw_total  # sub-array at its *current* weights acts as one element
        return None

    def _signature(self, weights: bool = True):
        """Fingerprint of what G depends on besides frequency/lmax: geometry (+ all sub-array weights if ``weights``)."""
        sig = [hash(self._element_positions().tobytes())]
        for e in self.element if isinstance(self.element, np.ndarray) else [self.element]:
            if isinstance(e, ArrayDirectivityMixin):
                sig += ([hash(np.asarray(e.excitation.values).tobytes())] if weights else []) + [
                    e._signature(weights),
                ]
        return tuple(sig)

    def _required_lmax(self, k: float, l_element: int = 24) -> int:
        """Spherical bandwidth of |E|^2 at wavenumber ``k``: array extent + the element's own bandwidth."""
        el = self.element[0] if isinstance(self.element, np.ndarray) else self.element
        if isinstance(el, ArrayDirectivityMixin):
            l_element = el._required_lmax(k)
        return self._lmax_for_pos(k, self._element_positions(), l_element)

    @staticmethod
    def _lmax_for_pos(k: float, pos: np.ndarray, l_element: int = 24) -> int:
        return _dir.lmax_for(
            k * float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))),
            l_element,
        )

    @staticmethod
    def _unit_weights(w: xr.DataArray) -> xr.DataArray:
        return w / np.sqrt((np.abs(w) ** 2).sum("port"))

    def _flat_leaf_model(self, top: xr.DataArray):
        """
        Flatten a hierarchy of identical sub-arrays into ``(leaf, positions, weights)``, or None.

        ``E_total = sum_t w_t/|w_t| * sum_s w_s/|w_s| * e(r) exp(jk (r_t + r_s) . r_hat)`` is a flat array of
        the leaf element at positions ``r_t + r_s`` with product weights.
        """
        if "port" not in top.dims:
            return None
        pos = self._element_positions()
        if isinstance(self.element, Antenna):
            if "port" in self.element.data.dims:
                return None
            return self.element, pos, self._unit_weights(top)
        if not isinstance(self.element, ArrayDirectivityMixin):
            return None
        child = self.element._flat_leaf_model(self.element.excitation)
        if child is None:
            return None
        leaf, cpos, cw = child
        # Sub-array positions live in the sub-array's reference frame: rotate them into ours.
        rot = self.reference_cs.get_relative_rotation(self.element.reference_cs).as_multirotation()
        if len(rot) != 1:
            return None
        cpos = cpos @ rot.apply(np.eye(3), inverse=True)
        w = (self._unit_weights(top).rename(port="_pt") * cw.rename(port="_pc")).stack(
            port=("_pt", "_pc"),
        )
        w = w.reset_index("port", drop=True)
        w = w.assign_coords(port=np.arange(w.sizes["port"]))
        return leaf, (pos[:, None, :] + cpos[None, :, :]).reshape(-1, 3), w

    def _models(self, top: xr.DataArray):
        """Candidate Gram models, fastest first: ``(name, leaf or None, positions, weights)``."""
        try:
            flat = (
                self._flat_leaf_model(top)
                if isinstance(self.element, ArrayDirectivityMixin)
                else None
            )
        except Exception as e:
            logger.debug(f"flattening failed ({type(e).__name__}: {e})")
            flat = None
        if flat is not None:
            yield ("flat", *flat)
        pos = self._element_positions()
        leaf = self._identical_element()
        if leaf is not None:
            yield ("identical", leaf, pos, top)
        yield ("per-port", None, pos, top)

    # ------------------------------------------------------------------------------------
    # Gram matrix (cached, self-checking)
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _collapse_frame_dims(dat: xr.DataArray, keep) -> xr.DataArray:
        """
        Drop dims that the pattern does not depend on (e.g. the ``time`` axis inherited from a moving parent frame
        when the field is requested in the array's own frame).  Verified, not assumed: a dim along which the field
        actually changes raises, so the caller falls through to the next (slower, general) model.
        """
        extra = [d for d in dat.dims if d not in keep]
        if not extra:
            return dat
        first = dat.isel(dict.fromkeys(extra, 0))
        v0 = _mag(first)
        scale = max(float(np.max(np.abs(v0))), 1e-300)
        for d in extra:
            if dat.sizes[d] > 1 and not np.allclose(
                _mag(dat.isel({**dict.fromkeys(extra, 0), d: -1})),
                v0,
                rtol=1e-9,
                atol=1e-12 * scale,
            ):
                raise NotImplementedError(f"pattern varies along '{d}' in the array frame")
        return first

    def _element_power(self, f_hz: float, grid: _dir.SphereGrid, leaf: Antenna, tag) -> np.ndarray:
        """Element power pattern |E_e|^2 (nt, nph) in the array reference frame (cached)."""
        key = ("ue", float(f_hz), grid.lmax, tag)
        if key not in self._gram_cache:
            kw = dict(frequency=np.array([f_hz]) * ureg.Hz, hcs=self.reference_cs)
            u = _dir._request_power(leaf, grid, kw)[0]
            self._cache_put(key, u.reshape(u.shape[-2:]) if u.ndim == 2 else u[(0,) * (u.ndim - 2)])
        return self._gram_cache[key]

    def _model_matches(self, leaf: Antenna, pos: np.ndarray, w: xr.DataArray, f_hz: float) -> bool:
        """
        Self-check: ``U_element * |AF|^2`` of the model must reproduce the real array pattern at a few directions.
        Catches anything a model gets wrong (rotated / offset sub-array frames, unexpected element dims,
        convention mismatches) before it can corrupt TRP.
        """
        try:
            th = np.radians([20.0, 50.0, 80.0, 110.0, 150.0])
            ph = np.radians([-120.0, -30.0, 40.0, 100.0, 170.0])
            k = 2 * np.pi * f_hz / _dir._C0
            kw = dict(frequency=np.array([f_hz]) * ureg.Hz, hcs=self.reference_cs)
            extra = [d for d in w.dims if d != "port"]
            w0 = np.asarray(w.isel(dict.fromkeys(extra, 0)).values, dtype=complex)
            w0 = w0 / np.linalg.norm(w0)
            tot = self._raw_total.request_data(theta=th * ureg.radian, phi=ph * ureg.radian, **kw)
            tot = tot.sel(polarization="apolar")
            tot = tot.isel({d: 0 for d in tot.dims if d in extra})
            tot = self._collapse_frame_dims(tot, keep=("frequency", "theta", "phi")).transpose(
                ..., "theta", "phi"
            )
            v = tot.data
            u_tot = np.abs(np.asarray(v.magnitude if hasattr(v, "magnitude") else v)).squeeze() ** 2
            ue = _dir._request_power(leaf, _dir.SphereGrid(th, ph, th * 0, 0.0, 0), kw)[0]
            ue = ue.reshape(-1, *ue.shape[-2:])
            if not np.allclose(ue, ue[:1], rtol=1e-9, atol=1e-12 * ue.max()):
                return False  # element pattern depends on something other than direction (e.g. moving frame)
            ue = ue[0]
            T, P = np.meshgrid(th, ph, indexing="ij")
            r = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(
                -1,
                3,
            )
            u_mod = (ue.reshape(-1) * np.abs(np.exp(1j * k * (r @ pos.T)) @ w0) ** 2).reshape(
                T.shape,
            )
            return bool(
                np.allclose(u_mod, u_tot.reshape(T.shape), rtol=1e-6, atol=1e-9 * u_tot.max()),
            )
        except NoDataError:
            raise
        except Exception as e:
            logger.debug(f"model check failed ({type(e).__name__}: {e})")
            return False

    def _dense_gram(
        self,
        f_hz: float,
        grid: _dir.SphereGrid,
        max_bytes: int = 400_000_000,
    ) -> np.ndarray:
        """Gram matrix from explicit per-port fields -- general (rotated / non-identical elements)."""
        n = len(self.coordinate_systems)
        nt, nph = grid.shape
        rows = max(1, int(max_bytes // (n * 3 * nph * 16)))
        w2d = grid.weights
        g = np.zeros((n, n), dtype=np.complex128)
        for i0 in range(0, nt, rows):
            sub = grid.theta_slice(slice(i0, i0 + rows))
            dat = self.elements.request_data(
                theta=sub.theta * ureg.radian,
                phi=sub.phi * ureg.radian,
                frequency=np.array([f_hz]) * ureg.Hz,
                hcs=self.reference_cs,
            )
            pols = ["x", "y", "z"] if "x" in dat.polarization.values else ["apolar"]
            dat = dat.sel(polarization=pols)
            if dat.size == 0:
                raise NoDataError("no element data at the requested frequency")
            dat = self._collapse_frame_dims(
                dat, keep=("port", "polarization", "frequency", "theta", "phi")
            )
            dat = dat.transpose("port", "polarization", "frequency", "theta", "phi")
            v = dat.data
            v = np.asarray(v.magnitude if hasattr(v, "magnitude") else v)
            v = v.reshape(n, len(pols), -1)
            fw = (v * np.sqrt(w2d[i0 : i0 + sub.theta.size].reshape(-1))).reshape(
                n,
                -1,
            )  # (port, pol x node)
            g += fw.conj() @ fw.T
        return g

    def _subarray_power(
        self,
        sub,
        f_hz: float,
        grid: _dir.SphereGrid,
        max_bytes: int = 1_500_000_000,
    ) -> np.ndarray:
        """
        ``|E_sub|^2`` (nt, nph) of an identical sub-array at its *current* weights, in this array's frame.

        The per-port fields of the sub-array do not depend on its weights, so they are requested once (cached) and
        each re-steer is just a weighted sum -- no pattern request on the (large) quadrature grid.
        """
        n = len(sub.coordinate_systems)
        nt, nph = grid.shape
        key = ("ports", float(f_hz), grid.lmax, sub._signature(weights=True))
        if key not in self._gram_cache:
            if n * 3 * nt * nph * 16 > max_bytes:
                raise MemoryError("sub-array port fields too large to cache")
            rows = max(1, int(1.0e8 // (n * 3 * nph * 16)))
            chunks = []
            for i0 in range(0, nt, rows):
                sl = grid.theta_slice(slice(i0, i0 + rows))
                dat = sub.elements.request_data(
                    theta=sl.theta * ureg.radian,
                    phi=sl.phi * ureg.radian,
                    frequency=np.array([f_hz]) * ureg.Hz,
                    hcs=self.reference_cs,
                )
                pols = ["x", "y", "z"] if "x" in dat.polarization.values else ["apolar"]
                dat = dat.sel(polarization=pols)
                if dat.size == 0:
                    raise NoDataError("no element data at the requested frequency")
                dat = self._collapse_frame_dims(
                    dat, keep=("port", "polarization", "frequency", "theta", "phi")
                )
                v = dat.transpose("port", "polarization", "frequency", "theta", "phi").data
                v = np.asarray(v.magnitude if hasattr(v, "magnitude") else v)
                chunks.append(v.reshape(n, len(pols), sl.theta.size, nph))
            self._cache_put(key, np.concatenate(chunks, axis=2))
        w = sub.excitation
        if "frequency" in w.dims:
            w = w.sel(frequency=f_hz, method="nearest")
        wv = np.asarray(w.values, dtype=complex)
        if wv.ndim != 1:
            raise NotImplementedError("sub-array excitation carries scan dimensions")
        field = np.tensordot(
            wv / np.linalg.norm(wv),
            self._gram_cache[key],
            axes=(0, 0),
        )  # (pol, nt, nph)
        return (np.abs(field) ** 2).sum(0)

    def _gram_for(
        self,
        f_hz: float,
        lmax,
        name: str,
        leaf,
        pos: np.ndarray,
        w,
    ) -> np.ndarray | None:
        """Cached Gram matrix of one model at one frequency; None if the model fails its self-check."""
        k = 2 * np.pi * f_hz / _dir._C0
        if name == "flat":
            L, sig = (
                (lmax or self._lmax_for_pos(k, pos)),
                ("flat", hash(pos.tobytes())),
            )  # weight independent
        else:
            L, sig = (lmax or self._required_lmax(k)), (name, self._signature())
        key = (float(f_hz), int(L), sig)
        if key in self._gram_cache:
            return self._gram_cache[key]
        grid = _dir.gl_sphere(L)
        if leaf is None:
            g = self._dense_gram(f_hz, grid)
        else:
            # The self-check concerns geometry/frames/conventions, not weights: once per geometry, not per steer.
            vkey = ("checked", name, float(f_hz), self._signature(weights=False))
            if vkey not in self._gram_cache:
                self._cache_put(vkey, self._model_matches(leaf, pos, w, f_hz))
            if not self._gram_cache[vkey]:
                logger.debug(
                    f"'{name}' array model does not reproduce the array pattern; trying the next model.",
                )
                return None
            ue = None
            if name == "identical" and isinstance(self.element, ArrayDirectivityMixin):
                try:
                    ue = self._subarray_power(self.element, f_hz, grid)
                except NoDataError:
                    raise
                except Exception as e:
                    logger.debug(
                        f"cached sub-array fields unavailable ({type(e).__name__}: {e}); requesting the pattern",
                    )
            if ue is None:
                ue = self._element_power(f_hz, grid, leaf, tag=sig)
            g = _dir.gram_identical(pos, k, ue, grid)
        return self._cache_put(key, g)

    def _solve(self, freqs: np.ndarray, lmax=None, excitation=None):
        """
        ``(G per frequency, weights (F, N, S), other dims, shape, coords)`` from the first model that works.

        Raises ``NoDataError`` if the elements have no data at the frequencies; any other failure of every model
        raises ``RuntimeError`` (callers then fall back to quadrature).
        """
        top = self.excitation if excitation is None else excitation
        errors = []
        for name, leaf, pos, w in self._models(top):
            try:
                gs = [self._gram_for(f, lmax, name, leaf, pos, w) for f in freqs]
                if any(g is None for g in gs):
                    continue
                return (np.array(gs), *self._weights(w, freqs))
            except NoDataError:
                raise
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
                logger.debug(f"'{name}' model failed ({type(e).__name__}: {e})")
        raise RuntimeError("no Gram model worked: " + "; ".join(errors))

    # ------------------------------------------------------------------------------------
    # TRP
    # ------------------------------------------------------------------------------------
    def _weights(self, w: xr.DataArray, freqs: np.ndarray):
        """Excitation as an ``(F, N, S)`` complex array (+ labels of the S scan combinations)."""
        if "port" not in w.dims:
            raise ValueError("excitation needs a 'port' dimension")
        n = w.sizes["port"]
        per_f = []
        for f in freqs:
            wf = w.sel(frequency=f, method="nearest") if "frequency" in w.dims else w
            other = [d for d in wf.dims if d != "port"]
            wf = wf.transpose("port", *other)
            per_f.append(np.asarray(wf.values, dtype=np.complex128).reshape(n, -1))
        other = [d for d in w.dims if d not in ("port", "frequency")]
        shape = [w.sizes[d] for d in other]
        coords = {d: w.coords[d] for d in other if d in w.coords}
        return np.array(per_f), other, shape, coords

    def _trp_quadrature(self, freqs: np.ndarray) -> xr.DataArray:
        """Last resort: integrate the actual array pattern on an adaptive Gauss-Legendre grid (slow, always right)."""
        trp, other, coords = _dir.trp_generic(self._raw_total, dict(frequency=freqs * ureg.Hz))
        da = xr.DataArray(np.asarray(trp), dims=other)
        da = da.assign_coords(
            {d: c for d, c in coords.items() if d != "frequency" and c.size == da.sizes[d]},
        )
        if "frequency" in other:
            return da.transpose("frequency", ...).assign_coords(frequency=freqs)
        return da.expand_dims(frequency=freqs)

    def _trp_da(self, frequency=None, excitation=None, lmax=None) -> xr.DataArray:
        freqs = self._freqs_hz(frequency)
        try:
            g, W, other, shape, coords = self._solve(freqs, lmax, excitation)
        except NoDataError:
            raise
        except Exception as e:
            if excitation is not None:
                raise
            logger.warning(
                f"Gram-matrix TRP unavailable ({type(e).__name__}: {e}); using adaptive quadrature (slow).",
            )
            return self._trp_quadrature(freqs)
        trp = np.array([_dir.quad_form_trp(g[i], W[i]) for i in range(len(freqs))]).reshape(
            len(freqs),
            *shape,
        )
        return xr.DataArray(trp, dims=["frequency", *other], coords={"frequency": freqs, **coords})

    def trp(self, frequency=None, excitation=None, lmax=None) -> xr.DataArray:
        """
        Total radiated power ``integral(|E_total|^2 dOmega)`` of the array [dimensionless, real].

        Independent of any angular grid and essentially free after the first call.  Pass ``excitation``
        (dims ``port`` + any scan dims) to evaluate many weightings at once without touching the array.
        """
        return self._trp_da(frequency, excitation, lmax) * ureg.dimensionless

    def _leaf_trp(self, leaf: Antenna, f_hz: float) -> float:
        key = ("leaf_trp", float(f_hz), id(leaf))
        if key not in self._gram_cache:
            trp = _dir.trp_generic(leaf, dict(frequency=np.array([f_hz]) * ureg.Hz))[0]
            self._cache_put(key, float(np.asarray(trp).reshape(-1)[0]))
        return self._gram_cache[key]

    def _radiated_element_power(self, frequency=None) -> xr.DataArray:
        """
        Power the *elements themselves* radiate (``4*pi*eta`` for elements of radiation efficiency ``eta``) --
        the array TRP without the overlap between different elements' fields.
        """
        freqs = self._freqs_hz(frequency)
        W, other, shape, coords = self._weights(self.excitation, freqs)
        leaf = self.element
        while isinstance(leaf, ArrayDirectivityMixin):
            leaf = leaf.element
        if (
            isinstance(leaf, Antenna) and "port" not in leaf.data.dims
        ):  # identical leaves: its own TRP
            out = np.repeat(
                np.array([self._leaf_trp(leaf, f) for f in freqs])[:, None],
                W.shape[-1],
                axis=1,
            )
        else:  # per-port elements: weighted mean of the diagonal of G
            g = np.array(
                [
                    self._gram_for(f, None, "per-port", None, self._element_positions(), None)
                    for f in freqs
                ],
            )
            out = np.array(
                [
                    (np.abs(W[i]) ** 2 * np.real(np.diag(g[i]))[:, None]).sum(0)
                    / (np.abs(W[i]) ** 2).sum(0)
                    for i in range(len(freqs))
                ],
            )
        return xr.DataArray(
            out.reshape(len(freqs), *shape),
            dims=["frequency", *other],
            coords={"frequency": freqs, **coords},
        )

    # ------------------------------------------------------------------------------------
    # Directivity / gain at angles
    # ------------------------------------------------------------------------------------
    def _steer_angles(self):
        """Stored scalar steering direction (theta, phi) as radian Quantities."""
        info = self._steer_info
        if info is None or info[0] is not self._steering_vector:
            raise ValueError(
                "No steering direction stored: pass theta and phi, or call steer_phase_centers first.",
            )
        p0, t0 = (
            np.asarray(
                (x.data if hasattr(x, "data") and not hasattr(x, "to") else x)
                .to("radian")
                .magnitude,
            )
            for x in info[1:]
        )
        if p0.size != 1 or t0.size != 1:
            raise ValueError(
                "The array is steered to several directions: pass theta and phi explicitly.",
            )
        return float(t0.squeeze()) * ureg.radian, float(p0.squeeze()) * ureg.radian

    def directivity(self, theta=None, phi=None, frequency=None, **kwargs) -> xr.DataArray:
        """
        Directivity ``4*pi*|E_total|^2 / TRP`` on a ``theta`` x ``phi`` grid.  With no angles it is evaluated
        at the stored steering direction (the *scanned beam*, not whichever lobe happens to be largest).
        """
        steer = theta is None and phi is None
        if steer:
            theta, phi = (np.atleast_1d(a.magnitude) * a.units for a in self._steer_angles())
        freqs = self._freqs_hz(frequency)
        kwargs.setdefault(
            "hcs",
            self.reference_cs,
        )  # angles are in the array reference frame (as for steering)
        dat = self._raw_total.request_data(
            theta=theta,
            phi=phi,
            frequency=freqs * ureg.Hz,
            **kwargs,
        )
        u = np.abs(dat.sel(polarization="apolar")) ** 2
        u = xr.DataArray(
            np.asarray(u.data.magnitude if hasattr(u.data, "magnitude") else u.data),
            dims=u.dims,
            coords=u.coords,
        )
        d = _dir.FOUR_PI * u / self._trp_da(frequency=frequency)
        return d.squeeze(["theta", "phi"], drop=True) if steer else d

    def d0(self, frequency=None) -> xr.DataArray:
        """
        Directivity of the scanned beam, ``4*pi*U(steering direction)/TRP`` [dimensionless].

        TRP is the exact whole-sphere value (grating lobes included); the numerator is the real pattern at the
        stored steering direction, so this is the beam's directivity -- not the peak of whichever lobe is
        tallest, and not dependent on any angular grid.  Differs from the true local peak only by the element
        pattern's squint, normally well under 0.1 dB.
        """
        return self.directivity(frequency=frequency) * ureg.dimensionless

    def _total_trp(self, kind, frequency=None) -> xr.DataArray:
        """TRP of ``total`` for pattern normalisation ``kind`` (False | "gain" | "directivity")."""
        trp = self.trp(frequency=frequency)
        if kind == "directivity":  # pattern scaled so that integral|E|^2 = 4*pi
            return trp * 0 + 4 * np.pi
        if kind == "gain":  # pattern scaled so that integral|E|^2 = power radiated by the elements
            rad = self._radiated_element_power(frequency=frequency)
            return xr.DataArray(rad.values * ureg.dimensionless, dims=rad.dims, coords=rad.coords)
        return trp

    def gain(self, theta=None, phi=None, frequency=None, **kwargs) -> xr.DataArray:
        """:meth:`directivity` times the radiation efficiency carried by the element patterns."""
        eta = self._radiated_element_power(frequency) / _dir.FOUR_PI
        return self.directivity(theta, phi, frequency, **kwargs) * eta

    # ------------------------------------------------------------------------------------
    # Normalised ``total``
    # ------------------------------------------------------------------------------------
    def normalization_factor(self, frequency=None, lmax=None, kind: str = "gain") -> xr.DataArray:
        """
        Amplitude scale to multiply the raw ``total`` by.

        ``kind="gain"`` (default): ``sqrt(P_elements / TRP)`` -- ``|E|^2`` is the *gain* pattern (directivity times the
        radiation efficiency carried by the element patterns; ``integral|E|^2 = P_elements``).
        ``kind="directivity"``: ``sqrt(4*pi / TRP)`` -- ``integral|E|^2 = 4*pi`` (assumes lossless).
        """
        trp = self._trp_da(frequency=frequency, lmax=lmax)
        if kind == "directivity":
            return np.sqrt(_dir.FOUR_PI / trp)
        if kind == "gain":
            return np.sqrt(self._radiated_element_power(frequency=frequency) / trp)
        raise ValueError("kind must be 'gain' or 'directivity'")

    def normalized_total(self, kind: str = "gain", lmax=None) -> Antenna:
        """
        ``total`` rescaled so its peak follows the array's true gain (or directivity) at every steering angle,
        grating-lobe and scan loss included.  The scale is evaluated at every ``request_data`` call from the cached
        Gram matrix, using the array's *current* weights and the *requested* frequencies.
        """
        raw = self._raw_total

        def scale(*args, **kwargs):
            try:
                return self.normalization_factor(
                    frequency=kwargs.get("frequency"),
                    lmax=lmax,
                    kind=kind,
                )
            except (
                NoDataError
            ):  # element has no data there: the raw request is empty too, nothing to scale
                return 1.0

        out = Antenna(OperatorFunction(raw, scale, opr.mul), raw.hcs)
        out._array_source = (self, kind)
        return out
