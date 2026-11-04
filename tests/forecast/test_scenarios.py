from statistics import NormalDist

import numpy as np
import pytest
from battery_optimisation.forecast.quantiles import pit

from battery_optimisation.forecast.scenarios import (
    GaussianCopula,
    reduce_scenarios,
    sample_paths,
)

LEVELS = np.linspace(0.01, 0.99, 99)
Z = np.array([NormalDist().inv_cdf(q) for q in LEVELS])
STEPS, SIGMA, RHO = 48, 30.0, 0.8


def ar1(n_days: int, rng) -> np.ndarray:
    """Stationary standard-normal AR(1) noise, one 48-step path per day."""
    eps = np.empty((n_days, STEPS))
    eps[:, 0] = rng.standard_normal(n_days)
    for t in range(1, STEPS):
        eps[:, t] = RHO * eps[:, t - 1] + np.sqrt(1 - RHO**2) * rng.standard_normal(n_days)
    return eps


def lag1(x: np.ndarray) -> float:
    return float(np.mean([np.corrcoef(x[:, t], x[:, t + 1])[0, 1] for t in range(STEPS - 1)]))


def test_copula_reproduces_realised_lag1_correlation() -> None:
    """CLAUDE.md invariant: lag-1 correlation of sampled paths ~ that of realised prices."""
    rng = np.random.default_rng(0)
    mu = 150 + 100 * np.sin(np.arange(STEPS) / STEPS * 2 * np.pi)
    quantiles = mu[:, None] + SIGMA * Z[None, :]  # calibrated marginals, [48, Q]
    realised = mu + SIGMA * ar1(300, rng)  # 300 past days

    u = pit(LEVELS, np.broadcast_to(quantiles, (300, STEPS, LEVELS.size)), realised)
    copula = GaussianCopula.fit(u, shrinkage=0.0)
    paths = sample_paths(LEVELS, quantiles, copula, n=3000, rng=rng)

    realised_lag1 = lag1((realised - mu) / SIGMA)
    sampled_lag1 = lag1((paths - mu) / SIGMA)
    assert sampled_lag1 == pytest.approx(realised_lag1, abs=0.05)

    # Independent sampling (what CLAUDE.md forbids) loses it: the test has teeth.
    independent = sample_paths(LEVELS, quantiles, GaussianCopula.independent(STEPS), 3000, rng)
    assert lag1((independent - mu) / SIGMA) < 0.1


def test_fitted_correlation_is_a_valid_correlation_matrix() -> None:
    rng = np.random.default_rng(1)
    u = rng.uniform(0.01, 0.99, size=(20, STEPS))  # fewer days than steps: rank deficient
    corr = GaussianCopula.fit(u, shrinkage=0.1).corr
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert np.linalg.eigvalsh(corr).min() > 0


def test_reduction_keeps_spike_paths_and_weights() -> None:
    rng = np.random.default_rng(2)
    paths = rng.normal(150, 30, size=(1000, STEPS))
    spikes = [10, 500, 900]
    for k, i in enumerate(spikes):
        paths[i, 20 + k] = 4000.0
    scen = reduce_scenarios(paths, n_scenarios=30, n_spike=3, rng=rng)

    assert scen.paths.shape == (30, STEPS)
    assert scen.weights.sum() == pytest.approx(1.0)
    assert np.all(scen.weights > 0)
    for i in spikes:  # every spike path survives as its own scenario
        assert any(np.array_equal(p, paths[i]) for p in scen.paths)
    for p in scen.paths:  # medoids are actual sampled paths
        assert any(np.array_equal(p, q) for q in paths)
