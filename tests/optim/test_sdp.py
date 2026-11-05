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


def test_stage_limits_keep_soc_in_band_and_power_down(battery_cfg, market_cfg):
    """Reserves held at a stage shrink its power headroom and its SOC band."""
    from battery_optimisation.optim.sdp import StageLimits

    prices = np.tile(np.array([50.0] * 24 + [400.0] * 24)[:, None], (1, 1))  # buy low, sell high
    n = prices.shape[0]
    lo = np.full(n, battery_cfg.soc_min)
    lo[30:] = 60.0  # from stage 30, keep 60 MWh for raise reserves (above the 50 MWh end rule)
    limits = StageLimits(
        max_discharge_mw=np.full(n, 60.0),  # 40 MW held for raise reserves
        max_charge_mw=np.full(n, battery_cfg.p_max_mw),
        soc_low=lo,
        soc_high=np.full(n, battery_cfg.soc_max),
    )
    b = Battery(battery_cfg)
    free = solve_sdp(prices, np.ones(1), b, market_cfg.interval_h, BIDDING)
    held = solve_sdp(prices, np.ones(1), b, market_cfg.interval_h, BIDDING, limits=limits)
    assert held.at(0, battery_cfg.soc_init) < free.at(0, battery_cfg.soc_init)  # costs value
    # below the band at a held stage is heavily penalised
    assert held.at(31, 50.0) < held.at(31, 60.0) - 1000
    for t in range(0, n, 6):
        assert np.all(np.diff(held.water_value(t)) <= 1e-6)  # still concave


def lattice(rho: float, states: int = 9):
    from synthetic_forecast import normal_quantiles

    from battery_optimisation.config import load_forecast
    from battery_optimisation.forecast.markov import price_lattice

    levels = load_forecast().quantile_levels
    q = normal_quantiles(daily_shape(1), 80.0, levels)
    return price_lattice(np.asarray(levels), q, states, rho)


def test_markov_with_independent_transitions_is_the_stagewise_sdp(battery_cfg, market_cfg):
    prices, weights = scenarios(n=8)
    battery, dt = Battery(battery_cfg), market_cfg.interval_h
    plain = solve_sdp(prices, weights, battery, dt, BIDDING)
    markov = solve_sdp(prices, weights, battery, dt, BIDDING, transitions=np.tile(weights, (8, 1)))
    assert markov.values == pytest.approx(plain.values)
    for t in (0, 20, 47):  # the continuation does not depend on the price
        cont = markov.continuation_at(t, np.array([-50.0, 100.0, 900.0]))
        assert cont == pytest.approx(np.broadcast_to(plain.values[t + 1], cont.shape), abs=1e-3)
        assert plain.continuation_at(t, np.array([10.0])) == pytest.approx(
            plain.values[t + 1][None]
        )


def test_markov_continuation_is_concave_in_every_state(battery_cfg, market_cfg):
    """CLAUDE.md invariant, per price state."""
    lat = lattice(0.8)
    vf = solve_sdp(
        lat.prices,
        lat.weights,
        Battery(battery_cfg),
        market_cfg.interval_h,
        BIDDING,
        transitions=lat.transitions,
    )
    for t in range(0, 48, 6):
        cont = vf.continuation_at(t, lat.prices[t])
        slopes = np.diff(cont, axis=1)
        assert np.all(np.diff(slopes, axis=1) <= 1e-3), f"not concave at t={t}"
    assert np.all(np.diff(vf.water_value(0)) <= 1e-6)


def test_persistent_prices_remove_fake_spreads(battery_cfg, market_cfg):
    """Same marginals: independent stages promise spreads that correlated prices do not give."""
    battery, dt = Battery(battery_cfg), market_cfg.interval_h
    value = {}
    for rho in (0.0, 0.9):
        lat = lattice(rho)
        vf = solve_sdp(lat.prices, lat.weights, battery, dt, BIDDING, transitions=lat.transitions)
        value[rho] = vf.at(0, battery_cfg.soc_init)
    assert value[0.9] < 0.8 * value[0.0]
