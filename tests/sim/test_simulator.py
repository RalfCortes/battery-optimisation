from datetime import timedelta

import polars as pl
import pytest
from synthetic import FixedStrategy, daily_shape, make_prices

from battery_optimisation.battery import Battery, LimitBreachError
from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import IntervalOffers, Market, MarketView, Side, Tranche
from battery_optimisation.sim.simulator import Simulator


def offers_at(market: Market, i: int, *tranches: tuple[Side, float, float]) -> IntervalOffers:
    """tranches: (side, offer price, MW)."""
    return IntervalOffers(
        interval_start=market.intervals[i],
        tranches=tuple(Tranche(side=s, price=p, quantity_mw=q) for s, p, q in tranches),
    )


# --------------------------------------------------------------------------- no look-ahead


def test_view_only_shows_the_past(market_cfg) -> None:
    m = Market(market_cfg, make_prices(daily_shape(1)))
    view = m.view(10)
    assert view.now == m.intervals[10]
    history = view.history()
    assert history.height == 10 - market_cfg.gate_closure_intervals
    assert history[Col.INTERVAL_START].max() < view.decided_at


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


def test_infeasible_offers_raise(market_cfg, battery_cfg) -> None:
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0] * 3, [100.0] * 3, market_cfg)  # 50 MWh cannot cover it
    with pytest.raises(LimitBreachError, match="soc"):
        Simulator(m, Battery(battery_cfg)).run(strategy)


def test_offers_are_fixed_at_gate_closure(market_cfg, battery_cfg) -> None:
    gate = market_cfg.gate_closure_intervals
    assert gate == 2
    dt = market_cfg.interval_h

    class ChargeSpy:
        """Charges 6 MW every interval and records what it knew when deciding."""

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def offers(self, view: MarketView, soc_mwh: float) -> IntervalOffers:
            hist = view.history()
            self.calls.append((view.now, view.decided_at, hist[Col.INTERVAL_START].max(), soc_mwh))
            return IntervalOffers.self_schedule(view.now, 6.0, 0.0, market_cfg)

    m = Market(market_cfg, make_prices([100.0] * 12))
    spy = ChargeSpy()
    Simulator(m, Battery(battery_cfg)).run(spy)

    assert [c[0] for c in spy.calls] == m.intervals  # one offer per interval, in order
    gain = battery_cfg.eta_c * 6.0 * dt
    for j, (now, decided, last_seen, soc) in enumerate(spy.calls):
        assert decided == now - timedelta(minutes=30 * gate)
        assert last_seen is None or last_seen < decided
        # SOC known at the decision: start of interval j - gate (window start for the first ones)
        assert soc == pytest.approx(battery_cfg.soc_init + max(j - gate, 0) * gain)
