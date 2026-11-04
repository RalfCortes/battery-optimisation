"""Correlated price scenarios from marginal quantile forecasts (CLAUDE.md).

1. PIT of realised prices under past forecasts -> normal scores z (one 48-vector per day).
2. Gaussian copula: correlation of z across days (shrunk towards identity, made PD).
3. Sample correlated z -> u = Phi(z) -> paths via each step's inverse CDF.
4. k-medoids reduction to 20-50 scenarios, always keeping the highest-price (spike) paths.

Steps are never sampled independently (that creates fake spreads and overtrading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.special import ndtr, ndtri

from battery_optimisation.forecast.quantiles import U_CLIP, inverse_cdf

_MIN_EIG = 1e-6


@dataclass(frozen=True)
class GaussianCopula:
    corr: np.ndarray  # [T, T]

    @classmethod
    def independent(cls, steps: int) -> Self:
        return cls(np.eye(steps))

    @classmethod
    def fit(cls, u: np.ndarray, shrinkage: float) -> Self:
        """Fit from PIT values ``u`` [days, T] (rows with NaN are dropped)."""
        u = u[~np.isnan(u).any(axis=1)]
        z = ndtri(np.clip(u, U_CLIP, 1 - U_CLIP))
        corr = np.corrcoef(z, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr = (1 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
        # nearest PD with unit diagonal: clip eigenvalues, rescale
        w, v = np.linalg.eigh((corr + corr.T) / 2)
        corr = (v * np.maximum(w, _MIN_EIG)) @ v.T
        d = np.sqrt(np.diag(corr))
        return cls(corr / np.outer(d, d))

    def sample_u(self, n: int, rng: np.random.Generator) -> np.ndarray:
        chol = np.linalg.cholesky(self.corr)
        return ndtr(rng.standard_normal((n, self.corr.shape[0])) @ chol.T)


def sample_paths(
    levels: np.ndarray,
    quantiles: np.ndarray,
    copula: GaussianCopula,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` correlated paths [n, T] from per-step quantiles [T, Q]."""
    u = copula.sample_u(n, rng)
    return np.stack(
        [inverse_cdf(levels, quantiles[t], u[:, t]) for t in range(quantiles.shape[0])], axis=1
    )


@dataclass(frozen=True)
class Scenarios:
    paths: np.ndarray  # [K, T]
    weights: np.ndarray  # [K], sums to 1


def reduce_scenarios(
    paths: np.ndarray, n_scenarios: int, n_spike: int, rng: np.random.Generator, max_iter: int = 50
) -> Scenarios:
    """k-medoids (Euclidean) with the ``n_spike`` highest-peak paths fixed as medoids.

    Weights are the share of sampled paths assigned to each medoid.
    """
    n = paths.shape[0]
    dist = np.sqrt(((paths[:, None, :] - paths[None, :, :]) ** 2).sum(-1))
    fixed = list(np.argsort(paths.max(axis=1))[::-1][:n_spike])

    # k-means++ initialisation for the free medoids
    medoids = list(fixed)
    candidates = np.setdiff1d(np.arange(n), fixed)
    if not medoids:
        medoids.append(int(rng.choice(candidates)))
    while len(medoids) < n_scenarios:
        d2 = dist[:, medoids].min(axis=1) ** 2
        d2[medoids] = 0
        medoids.append(int(rng.choice(n, p=d2 / d2.sum())))

    for _ in range(max_iter):
        labels = dist[:, medoids].argmin(axis=1)
        new = list(medoids)
        for k in range(len(fixed), n_scenarios):
            members = np.flatnonzero(labels == k)
            if members.size:
                new[k] = int(members[dist[np.ix_(members, members)].sum(axis=1).argmin()])
        if new == medoids:
            break
        medoids = new

    labels = dist[:, medoids].argmin(axis=1)
    weights = np.bincount(labels, minlength=n_scenarios) / n
    return Scenarios(paths=paths[medoids], weights=weights)
