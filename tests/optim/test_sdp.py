import numpy as np
import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.config import load_bidding
from battery_optimisation.optim.milp import solve_schedule
from battery_optimisation.optim.sdp import solve_sdp

BIDDING = load_bidding()


def scenarios(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """[48, n] prices around a daily shape, equal weights."""
    rng = np.random.default_rng(seed)
    base = daily_shape(1)
    prices = base[:, None] + rng.normal(0, 60, (48, n))
    return prices, np.full(n, 1 / n)


def test_value_function_is_concave_and_water_value_non_increasing(battery_cfg, market_cfg):
    """CLAUDE.md invariant."""
    prices, weights = scenarios()
    vf = solve_sdp(prices, weights, Battery(battery_cfg), market_cfg.interval_h, BIDDING)
    for t in range(0, 49, 6):
        wv = vf.water_value(t)
        assert np.all(np.diff(wv) <= 1e-6), f"water value increases with SOC at t={t}"


def test_terminal_value_matches_the_baseline_rule(battery_cfg, market_cfg):
    prices, weights = scenarios()
    vf = solve_sdp(prices, weights, Battery(battery_cfg), market_cfg.interval_h, BIDDING)
    end = vf.values[-1]
    above = vf.soc >= battery_cfg.soc_init
    assert np.all(end[above] == 0.0)
    below = ~above
    expected = -BIDDING.terminal_penalty_sgd_per_mwh * (battery_cfg.soc_init - vf.soc[below])
    assert end[below] == pytest.approx(expected)


def test_deterministic_price_matches_the_milp(battery_cfg, market_cfg):
    """One scenario with weight 1: SDP = perfect-foresight MILP (up to the SOC grid)."""
    prices = daily_shape(1, seed=2)
    battery = Battery(battery_cfg)
    vf = solve_sdp(prices[:, None], np.ones(1), battery, market_cfg.interval_h, BIDDING)
    milp = solve_schedule(make_prices(prices), battery, market_cfg)
    assert vf.at(0, battery_cfg.soc_init) == pytest.approx(milp.objective_sgd, rel=0.02)
    assert vf.at(0, battery_cfg.soc_init) <= milp.objective_sgd + 1e-6  # grid can only lose


def test_higher_future_prices_raise_the_water_value(battery_cfg, market_cfg):
    prices, weights = scenarios()
    battery = Battery(battery_cfg)
    low = solve_sdp(prices, weights, battery, market_cfg.interval_h, BIDDING)
    high = solve_sdp(prices * 2, weights, battery, market_cfg.interval_h, BIDDING)
    mid = len(low.soc) // 2
    assert high.water_value(24)[mid] > low.water_value(24)[mid]
