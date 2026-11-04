"""Quantile forecasts as distributions: CDF (PIT), inverse CDF and non-crossing fix.

The quantile function is interpolated linearly in normal-score space: knots at
z_i = Phi^-1(tau_i), linear between knots and linearly extrapolated beyond the outermost levels.
This is exact for Gaussian marginals, monotone, and gives unbounded tails (so PIT and inverse are
well defined for values beyond q_0.01 / q_0.99).

Shapes: ``values`` is ``[..., Q]`` (one distribution per leading index) and ``y`` / ``u`` match
the leading shape; or ``values`` is ``[Q]`` (one distribution) and ``y`` / ``u`` is any shape.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

_EPS = 1e-9  # makes flat quantile segments strictly increasing
_MIN_SLOPE = 1e-9
U_CLIP = 1e-9  # keeps PIT strictly inside (0, 1); |z| <= 6 (spikes live in the tails)


def rearrange(values: np.ndarray) -> np.ndarray:
    """Sort quantiles along the last axis (fixes quantile crossing)."""
    return np.sort(values, axis=-1)


def _knots(levels: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    z = ndtri(levels)
    q = np.sort(q) + _EPS * np.arange(q.size)
    lo = max((q[1] - q[0]) / (z[1] - z[0]), _MIN_SLOPE)
    hi = max((q[-1] - q[-2]) / (z[-1] - z[-2]), _MIN_SLOPE)
    return z, q, lo, hi


def _cdf_z(levels: np.ndarray, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    z, q, lo, hi = _knots(levels, q)
    out = np.interp(y, q, z)
    out = np.where(y < q[0], z[0] + (y - q[0]) / lo, out)
    return np.where(y > q[-1], z[-1] + (y - q[-1]) / hi, out)


def _quantile_z(levels: np.ndarray, q: np.ndarray, zz: np.ndarray) -> np.ndarray:
    z, q, lo, hi = _knots(levels, q)
    out = np.interp(zz, z, q)
    out = np.where(zz < z[0], q[0] + (zz - z[0]) * lo, out)
    return np.where(zz > z[-1], q[-1] + (zz - z[-1]) * hi, out)


def _apply(fn, levels, values, x) -> np.ndarray:
    levels, values, x = np.asarray(levels, float), np.asarray(values, float), np.asarray(x, float)
    if values.ndim == 1:
        return fn(levels, values, x)
    flat_v = values.reshape(-1, values.shape[-1])
    flat_x = x.reshape(-1)
    if flat_x.size != flat_v.shape[0]:
        raise ValueError("one point per distribution expected")
    out = np.array([fn(levels, v, xi) for v, xi in zip(flat_v, flat_x, strict=True)])
    return out.reshape(x.shape)


def pit(levels, values, y) -> np.ndarray:
    """Probability integral transform F(y) in (0, 1)."""
    z = _apply(_cdf_z, levels, values, y)
    return np.clip(ndtr(z), U_CLIP, 1 - U_CLIP)


def inverse_cdf(levels, values, u) -> np.ndarray:
    """Quantile function F^-1(u)."""
    zz = ndtri(np.clip(np.asarray(u, float), U_CLIP, 1 - U_CLIP))
    return _apply(_quantile_z, levels, values, zz)
