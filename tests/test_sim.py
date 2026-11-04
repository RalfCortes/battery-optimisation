import polars as pl
import pytest
from synthetic import FixedStrategy, daily_shape, make_prices

from battery_optimisation.battery import Battery, LimitBreachError
from battery_optimisation.bidding.strategy import ScheduleStrategy
from battery_optimisation.columns import Col
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import IntervalOffers, Market, MarketView, Side, Tranche
from battery_optimisation.sim.metrics import Period
from battery_optimisation.sim.simulator import Simulator


def offers_at(market: Market, i: int, *tranches: tuple[Side, float, float]) -> IntervalOffers:
    """tranches: (side, offer price, MW)."""
    return IntervalOffers(
        interval_start=market.intervals[i],
        tranches=tuple(Tranche(side=s, price=p, quantity_mw=q) for s, p, q in tranches),
    )


# --------------------------------------------------------------------------- market clearing


def test_clearing_rules_at_the_boundary(market_cfg) -> None:
    m = Market(market_cfg, make_prices([100.0, 99.0, 50.0, 51.0]))
    results = [
        m.clear(offers_at(m, 0, (Side.DISCHARGE, 100.0, 10.0))),  # offer == price -> clears
        m.clear(offers_at(m, 1, (Side.DISCHARGE, 100.0, 10.0))),  # offer >  price -> no
        m.clear(offers_at(m, 2, (Side.CHARGE, 50.0, 20.0))),  # bid == price -> clears
        m.clear(offers_at(m, 3, (Side.CHARGE, 50.0, 20.0))),  # bid <  price -> no
    ]
    assert [r.discharge_mw for r in results] == [10.0, 0.0, 0.0, 0.0]
    assert [r.charge_mw for r in results] == [0.0, 0.0, 20.0, 0.0]
    assert [r.price for r in results] == [100.0, 99.0, 50.0, 51.0]


def test_tranches_clear_independently(market_cfg) -> None:
    m = Market(market_cfg, make_prices([120.0]))
    o = offers_at(
        m,
        0,
        (Side.DISCHARGE, 80.0, 10.0),
        (Side.DISCHARGE, 110.0, 20.0),
        (Side.DISCHARGE, 150.0, 30.0),  # above price
    )
    assert m.clear(o).discharge_mw == 30.0


def test_both_sides_clearing_are_netted(market_cfg) -> None:
    m = Market(market_cfg, make_prices([100.0]))
    o = offers_at(m, 0, (Side.DISCHARGE, 50.0, 30.0), (Side.CHARGE, 150.0, 10.0))
    cleared = m.clear(o)
    assert (cleared.discharge_mw, cleared.charge_mw) == (20.0, 0.0)


def test_offer_rules_enforced(market_cfg) -> None:
    m = Market(market_cfg, make_prices([100.0]))
    too_many = offers_at(m, 0, *[(Side.DISCHARGE, 100.0, 1.0)] * (market_cfg.n_tranches + 1))
    with pytest.raises(ValueError, match="tranches"):
        m.clear(too_many)
    with pytest.raises(ValueError, match="floor"):
        m.clear(offers_at(m, 0, (Side.CHARGE, market_cfg.price_cap + 1, 1.0)))
    with pytest.raises(ValueError):
        Tranche(side=Side.CHARGE, price=10.0, quantity_mw=-1.0)


def test_self_schedule_always_clears(market_cfg) -> None:
    m = Market(market_cfg, make_prices([market_cfg.price_cap, market_cfg.price_floor]))
    for i in range(2):
        o = IntervalOffers.self_schedule(m.intervals[i], 30.0, 0.0, market_cfg)
        assert m.clear(o).charge_mw == 30.0
        o = IntervalOffers.self_schedule(m.intervals[i], 0.0, 40.0, market_cfg)
        assert m.clear(o).discharge_mw == 40.0


# --------------------------------------------------------------------------- no look-ahead


def test_view_only_shows_the_past(market_cfg) -> None:
    m = Market(market_cfg, make_prices(daily_shape(1)))
    view = m.view(10)
    assert view.now == m.intervals[10]
    history = view.history()
    assert history.height == 10
    assert history[Col.INTERVAL_START].max() < view.now


def test_simulator_never_shows_a_strategy_the_future(market_cfg, battery_cfg) -> None:
    class Spy:
        def __init__(self) -> None:
            self.seen: list[MarketView] = []

        def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
            self.seen.append(view)
            future = view.history().filter(pl.col(Col.INTERVAL_START) >= view.now)
            assert future.height == 0
            return IntervalOffers(interval_start=view.now)

    m = Market(market_cfg, make_prices(daily_shape(1)))
    spy = Spy()
    dispatch = Simulator(m, Battery(battery_cfg)).run(spy)
    assert [v.now for v in spy.seen] == m.intervals  # once per interval, in order
    assert dispatch.profit_sgd == 0.0


# --------------------------------------------------------------------------- simulator


def test_infeasible_offers_raise(market_cfg, battery_cfg) -> None:
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0] * 3, [100.0] * 3, market_cfg)  # 50 MWh cannot cover it
    with pytest.raises(LimitBreachError, match="soc"):
        Simulator(m, Battery(battery_cfg)).run(strategy)


def test_end_to_end_synthetic_day(market_cfg, battery_cfg) -> None:
    """Perfect foresight -> MILP plan -> self-schedule -> simulator reproduces the MILP."""
    prices = make_prices(daily_shape(1, seed=3))
    strategy = ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg)
    dispatch = Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)

    frame = dispatch.frame
    assert frame.height == 48
    assert dispatch.profit_sgd == pytest.approx(strategy.planned_profit_sgd, rel=1e-9)
    assert dispatch.profit_sgd > 0
    assert frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert frame[Col.SOC_MWH].max() <= battery_cfg.soc_max + 1e-6
    assert frame[Col.SOC_MWH][-1] >= battery_cfg.soc_init - 1e-6


def test_cycles_metric(market_cfg, battery_cfg) -> None:
    """Fill from soc_min to soc_max and empty again: exactly one equivalent full cycle."""
    cfg = battery_cfg.model_copy(update={"soc_init": battery_cfg.soc_min})
    dt, n = market_cfg.interval_h, 4
    charge = cfg.usable_mwh / (cfg.eta_c * dt) / n
    discharge = cfg.usable_mwh * cfg.eta_d / dt / n
    strategy = FixedStrategy([charge] * n + [0.0] * n, [0.0] * n + [discharge] * n, market_cfg)
    m = Market(market_cfg, make_prices([50.0] * n + [200.0] * n))
    dispatch = Simulator(m, Battery(cfg)).run(strategy)

    assert dispatch.total().cycles == pytest.approx(1.0)
    assert len(dispatch.stats(Period.DAY)) == 1
    assert dispatch.frame[Col.SOC_MWH][-1] == pytest.approx(cfg.soc_min)
