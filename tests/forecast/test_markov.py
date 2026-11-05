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
