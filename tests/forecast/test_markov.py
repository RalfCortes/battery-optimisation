import numpy as np
import pytest
from synthetic_forecast import normal_quantiles

from battery_optimisation.config import load_forecast
from battery_optimisation.forecast.markov import empirical_transitions, price_lattice

LEVELS = np.asarray(load_forecast().quantile_levels)
MU = 150 + 50 * np.sin(np.arange(48) / 48 * 2 * np.pi)


def test_transitions_are_distributions_and_states_increase() -> None:
    lat = price_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), states=11, rho=0.8)
    assert lat.prices.shape == (48, 11)
    assert lat.transitions.shape == (11, 11)
    assert lat.transitions.sum(axis=1) == pytest.approx(np.ones(11))
    assert np.all(lat.transitions >= 0)
    assert np.all(np.diff(lat.prices, axis=1) > 0)
    assert lat.weights == pytest.approx(np.full(11, 1 / 11))


def test_state_prices_keep_the_mean() -> None:
    lat = price_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), states=11, rho=0.8)
    assert lat.prices @ lat.weights == pytest.approx(MU, abs=0.5)


def test_no_correlation_is_stagewise_independence() -> None:
    lat = price_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), states=7, rho=0.0)
    assert lat.transitions == pytest.approx(np.full((7, 7), 1 / 7))


def test_persistence_grows_with_correlation() -> None:
    q = normal_quantiles(MU, 40.0, list(LEVELS))
    diag = [np.diag(price_lattice(LEVELS, q, 9, rho).transitions).mean() for rho in (0.3, 0.8)]
    assert diag[1] > diag[0] > 1 / 9
    # the stationary distribution stays (close to) the equal state weights
    lat = price_lattice(LEVELS, q, 9, 0.8)
    assert lat.weights @ lat.transitions == pytest.approx(lat.weights, abs=0.01)


def _skewed(scale: float) -> np.ndarray:
    """[48, Q] right-skewed (lognormal) quantiles, like reserve prices."""
    from scipy.stats import lognorm

    return np.tile(lognorm(s=0.6, scale=scale).ppf(LEVELS), (48, 1))


def test_reserve_factor_states_keep_each_products_mean() -> None:
    """One factor drives every reserve price: state r = factor bin, each product's price = its
    mean given the factor (loading lambda); averaged over the states: the product's mean."""
    from battery_optimisation.forecast.markov import factor_lattice, price_lattice

    q = {"a": _skewed(15.0), "b": _skewed(30.0)}
    lat = factor_lattice(LEVELS, q, {"a": 0.6, "b": 0.8}, states=5, rho=0.75)
    assert lat.prices.shape == (48, 5, 2)
    assert lat.transitions.sum(axis=1) == pytest.approx(np.ones(5))
    assert np.all(np.diff(lat.prices, axis=1) > 0)  # increasing in the factor
    mean = {k: price_lattice(LEVELS, v, 11, 0.0).prices.mean(axis=1) for k, v in q.items()}
    for j, k in enumerate(q):
        assert lat.prices[:, :, j] @ lat.weights == pytest.approx(mean[k], rel=0.01)
    # loading 0: the factor says nothing, every state is the mean
    flat = factor_lattice(LEVELS, q, {"a": 0.0, "b": 0.0}, states=5, rho=0.75)
    assert flat.prices[:, :, 0] == pytest.approx(np.repeat(mean["a"][:, None], 5, 1), rel=0.01)
    # loading 1: the product is the factor, states = the plain Markov lattice's
    full = factor_lattice(LEVELS, q, {"a": 1.0, "b": 1.0}, states=5, rho=0.75)
    assert full.prices[:, :, 1] == pytest.approx(
        price_lattice(LEVELS, q["b"], 5, 0.75).prices, rel=1e-3
    )
    assert full.transitions == pytest.approx(price_lattice(LEVELS, q["b"], 5, 0.75).transitions)


def ar1_pits(n: int, steps: int, rho: float, jump: float = 0.0, seed: int = 0) -> np.ndarray:
    """PITs [n, steps] of a stationary AR(1) in normal scores; with probability ``jump`` a step
    restarts from an independent draw (a jump the AR(1) cannot make)."""
    from scipy.special import ndtr

    rng = np.random.default_rng(seed)
    z = np.empty((n, steps))
    z[:, 0] = rng.standard_normal(n)
    for t in range(1, steps):
        ar = rho * z[:, t - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        z[:, t] = np.where(rng.random(n) < jump, rng.standard_normal(n), ar)
    return ndtr(z)


def tauchen(states: int, rho: float) -> np.ndarray:
    return price_lattice(
        LEVELS, normal_quantiles(MU[:1], 40.0, list(LEVELS)), states, rho
    ).transitions


def test_empirical_transitions_recover_an_ar1() -> None:
    u = ar1_pits(4000, 48, 0.8)
    est = empirical_transitions(u, states=11, rho=0.8, prior_strength=10.0)
    assert est == pytest.approx(tauchen(11, 0.8), abs=0.03)


def test_empirical_transitions_keep_states_equally_likely() -> None:
    u = ar1_pits(300, 48, 0.8, jump=0.1)
    est = empirical_transitions(u, states=11, rho=0.8, prior_strength=50.0)
    assert est.sum(axis=1) == pytest.approx(np.ones(11))
    assert est.sum(axis=0) == pytest.approx(np.ones(11), abs=1e-6)  # uniform stays uniform
    assert np.all(est > 0)


def test_empirical_transitions_see_jumps_tauchen_cannot() -> None:
    u = ar1_pits(4000, 48, 0.8, jump=0.1)
    est = empirical_transitions(u, states=11, rho=0.8, prior_strength=10.0)
    ar = tauchen(11, 0.8)
    assert est[0, -3:].sum() > 0.02 > ar[0, -3:].sum()  # from the lowest state to the top ones


def test_without_data_the_prior_is_tauchen() -> None:
    est = empirical_transitions(np.empty((0, 48)), states=11, rho=0.7, prior_strength=50.0)
    # Tauchen balanced: its columns sum to 0.99-1.01, so balancing moves it a little
    assert est == pytest.approx(tauchen(11, 0.7), abs=0.005)


def test_rows_with_missing_values_are_skipped() -> None:
    u = ar1_pits(500, 48, 0.8)
    holed = u.copy()
    holed[:50, 10] = np.nan
    a = empirical_transitions(u[50:], 11, 0.8, 50.0)
    b = empirical_transitions(holed, 11, 0.8, 50.0)
    assert a.shape == b.shape and np.isfinite(b).all()
