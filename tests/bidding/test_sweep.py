import numpy as np
import pytest

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.compress import compress
from battery_optimisation.bidding.sweep import offered_net_mw, optimal_net_mw
from battery_optimisation.bidding.water_value import offer_curve
from battery_optimisation.optim.sdp import ValueFunction
from battery_optimisation.sim.clearing import IntervalOffers

DT = 0.5


def concave_vf(battery_cfg) -> ValueFunction:
    soc = np.linspace(battery_cfg.soc_min, battery_cfg.soc_max, 101)
    v = 400 * soc - 2 * soc**2
    return ValueFunction(soc=soc, values=np.stack([v, v]), solve_time_s=0.0)


def prices() -> np.ndarray:
    # avoid exact tranche prices (ties): random, sorted
    return np.sort(np.random.default_rng(0).uniform(-50, 700, 300))


def as_offers(tranches) -> IntervalOffers:
    from datetime import datetime

    return IntervalOffers(interval_start=datetime(2026, 1, 1), tranches=tuple(tranches))


@pytest.mark.parametrize("soc", [10.0, 50.0, 87.3])
def test_uncompressed_curve_is_the_optimal_response(battery_cfg, soc) -> None:
    """The water-value curve must clear exactly the re-solved optimum at every price."""
    vf, b = concave_vf(battery_cfg), Battery(battery_cfg, soc)
    discharge, charge = offer_curve(vf, 0, soc, b, DT)
    grid = prices()
    best = optimal_net_mw(vf, 0, soc, b, DT, grid)
    assert offered_net_mw(as_offers(discharge + charge), grid) == pytest.approx(best, abs=1e-6)


def test_optimal_response_is_a_supply_curve(battery_cfg) -> None:
    vf, b = concave_vf(battery_cfg), Battery(battery_cfg, 50.0)
    best = optimal_net_mw(vf, 0, 50.0, b, DT, prices())
    assert np.all(np.diff(best) >= -1e-9)  # sells more (buys less) as the price rises
    assert best.min() >= -battery_cfg.p_max_mw - 1e-9 and best.max() <= battery_cfg.p_max_mw + 1e-9


def test_compression_gap_is_small(battery_cfg) -> None:
    vf, b = concave_vf(battery_cfg), Battery(battery_cfg, 50.0)
    discharge, charge = offer_curve(vf, 0, 50.0, b, DT)
    offers = as_offers(compress(discharge, 10) + compress(charge, 10))
    grid = prices()
    gap = np.abs(offered_net_mw(offers, grid) - optimal_net_mw(vf, 0, 50.0, b, DT, grid))
    assert gap.mean() < 10.0  # MW, averaged over the price grid


def markov_vf(battery_cfg, market_cfg):
    from synthetic import daily_shape
    from synthetic_forecast import normal_quantiles

    from battery_optimisation.config import load_bidding, load_forecast
    from battery_optimisation.forecast.markov import price_lattice
    from battery_optimisation.optim.sdp import solve_sdp

    levels = load_forecast().quantile_levels
    lat = price_lattice(
        np.asarray(levels), normal_quantiles(daily_shape(1), 80.0, levels), states=9, rho=0.8
    )
    vf = solve_sdp(
        lat.prices,
        lat.weights,
        Battery(battery_cfg),
        market_cfg.interval_h,
        load_bidding(),
        transitions=lat.transitions,
    )
    return vf, lat


@pytest.mark.parametrize("soc", [10.0, 50.0, 87.3])
def test_markov_curve_is_the_optimal_response(battery_cfg, market_cfg, soc) -> None:
    """With price states, the best action depends on what the price says about later prices:
    the curve still clears the re-solved optimum, at every price within the states' range."""
    vf, lat = markov_vf(battery_cfg, market_cfg)
    b, t = Battery(battery_cfg, soc), 30
    discharge, charge = offer_curve(vf, t, soc, b, DT)
    grid = np.sort(np.random.default_rng(1).uniform(lat.prices[t, 0], lat.prices[t, -1], 300))
    best = optimal_net_mw(vf, t, soc, b, DT, grid)
    offered = offered_net_mw(as_offers(discharge + charge), grid)
    assert np.mean(np.abs(offered - best)) < 1.0  # MW: price-grid resolution only
    assert np.all(np.diff(best) >= -1e-9)  # still a supply curve
    d = [tr.price for tr in discharge]
    c = [tr.price for tr in charge]
    assert d == sorted(d) and c == sorted(c, reverse=True)
