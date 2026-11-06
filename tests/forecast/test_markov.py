import numpy as np
import pytest
from synthetic_forecast import normal_quantiles

from battery_optimisation.config import load_forecast
from battery_optimisation.forecast.markov import price_lattice

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


# --- continuous price state: nodes in score space, hat-function transitions (M9) ---------------


def test_hat_expectations_match_sampling() -> None:
    """E[B_l(Z)] for Z ~ N(m, sd^2), B_l the linear-interpolation weight of node l (clamped at
    the ends): exact rows that sum to 1, equal to Monte Carlo."""
    from battery_optimisation.forecast.markov import hat_expectations

    nodes = np.linspace(-3.0, 3.0, 13)
    means = np.array([-3.5, -1.0, 0.2, 2.9])
    rows = hat_expectations(nodes, means, 0.6)
    assert rows.shape == (4, 13)
    assert rows.sum(axis=1) == pytest.approx(np.ones(4))
    assert np.all(rows >= 0)
    z = np.random.default_rng(0).normal(size=400_000)
    eye = np.eye(nodes.size)
    for m, row in zip(means, rows, strict=True):
        sampled = np.array([np.interp(m + 0.6 * z, nodes, eye[k]).mean() for k in range(13)])
        assert row == pytest.approx(sampled, abs=3e-3)


def test_hat_expectations_without_noise_are_interpolation_weights() -> None:
    from battery_optimisation.forecast.markov import hat_expectations

    nodes = np.linspace(-1.0, 1.0, 5)
    row = hat_expectations(nodes, np.array([0.25]), 0.0)[0]
    assert row == pytest.approx([0, 0, 0.5, 0.5, 0], abs=1e-9)


def test_node_lattice_is_a_markov_chain_with_increasing_prices() -> None:
    from battery_optimisation.forecast.markov import node_lattice

    lat = node_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), 21, 3.0, rho=0.8)
    assert lat.prices.shape == (48, 21)
    assert lat.transitions.shape == (21, 21)
    assert lat.transitions.sum(axis=1) == pytest.approx(np.ones(21))
    assert np.all(lat.transitions >= 0)
    assert np.all(np.diff(lat.prices, axis=1) > 0)
    assert lat.weights.sum() == pytest.approx(1.0)
    assert lat.scores == pytest.approx(np.linspace(-3.0, 3.0, 21))


def test_node_prices_keep_the_mean() -> None:
    """Node price = local mean of the price under its hat (Fact 1: the mean must survive)."""
    from battery_optimisation.forecast.markov import node_lattice

    lat = node_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), 21, 3.0, rho=0.8)
    assert lat.prices @ lat.weights == pytest.approx(MU, abs=0.2)
    skewed = node_lattice(LEVELS, _skewed(30.0), 21, 3.0, rho=0.8)
    exact = price_lattice(LEVELS, _skewed(30.0), 11, 0.0).prices.mean(axis=1)
    assert skewed.prices @ skewed.weights == pytest.approx(exact, rel=0.01)


def test_node_lattice_without_correlation_is_stagewise_independence() -> None:
    from battery_optimisation.forecast.markov import node_lattice

    lat = node_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), 15, 3.0, rho=0.0)
    assert lat.transitions == pytest.approx(np.tile(lat.weights, (15, 1)), abs=1e-3)


def test_node_lattice_is_stationary_and_gives_the_ar1_conditional_mean() -> None:
    """The score distribution stays put across stages, and E[price_{t+1} | z_j] is the AR(1)'s
    (normal prices: mu_{t+1} + sd rho z_j) away from the clamped ends."""
    from battery_optimisation.forecast.markov import node_lattice

    lat = node_lattice(LEVELS, normal_quantiles(MU, 40.0, list(LEVELS)), 21, 3.0, rho=0.8)
    assert lat.weights @ lat.transitions == pytest.approx(lat.weights, abs=3e-3)
    inner = np.abs(lat.scores) <= 2.0
    expected = MU[11] + 40.0 * 0.8 * lat.scores
    assert (lat.transitions @ lat.prices[11])[inner] == pytest.approx(expected[inner], abs=1.0)


def test_node_persistence_grows_with_correlation() -> None:
    from battery_optimisation.forecast.markov import node_lattice

    q = normal_quantiles(MU, 40.0, list(LEVELS))
    diag = [np.diag(node_lattice(LEVELS, q, 15, 3.0, rho).transitions).mean() for rho in (0.3, 0.9)]
    assert diag[1] > diag[0]
