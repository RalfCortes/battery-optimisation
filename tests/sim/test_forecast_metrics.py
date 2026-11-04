from statistics import NormalDist

import numpy as np
import pytest

from battery_optimisation.sim.metrics import coverage, crps_quantiles, spike_rates

LEVELS = np.linspace(0.01, 0.99, 99)
STD_NORMAL = np.array([NormalDist().inv_cdf(q) for q in LEVELS])


def analytic_crps_std_normal(y: float) -> float:
    nd = NormalDist()
    return y * (2 * nd.cdf(y) - 1) + 2 * nd.pdf(y) - 1 / np.sqrt(np.pi)


@pytest.mark.parametrize("y", [0.0, 1.0, -2.0])
def test_crps_matches_normal_closed_form(y) -> None:
    got = crps_quantiles(np.array([y]), LEVELS, STD_NORMAL[None, :])[0]
    assert got == pytest.approx(analytic_crps_std_normal(y), rel=0.03)


def test_crps_of_a_point_forecast_is_absolute_error() -> None:
    values = np.full((1, LEVELS.size), 10.0)
    assert crps_quantiles(np.array([13.0]), LEVELS, values)[0] == pytest.approx(3.0, rel=0.01)


def test_coverage() -> None:
    y = np.array([1.0, 5.0, 9.0, 20.0])
    assert coverage(y, np.full(4, 2.0), np.full(4, 10.0)) == 0.5


def test_spike_rates() -> None:
    y = np.array([10.0, 500.0, 600.0, 20.0])
    q90 = np.array([100.0, 450.0, 300.0, 450.0])
    hit, false_alarm = spike_rates(y, q90, threshold=400.0)
    assert (hit, false_alarm) == (0.5, 0.5)


def test_seasonal_naive_scale() -> None:
    from battery_optimisation.sim.metrics import seasonal_naive_mae

    history = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 8.0])  # season 3: diffs 2, 3, 4
    assert seasonal_naive_mae(history, season=3) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="season"):
        seasonal_naive_mae(history[:3], season=3)


def test_mase() -> None:
    from battery_optimisation.sim.metrics import mase

    y = np.array([10.0, 20.0, 30.0])
    assert mase(y, y, scale=5.0) == 0.0
    assert mase(y, y + 5.0, scale=5.0) == pytest.approx(1.0)  # as good as the naive scale
    assert mase(y, y + 2.5, scale=5.0) == pytest.approx(0.5)
