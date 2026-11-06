from datetime import timedelta

import polars as pl
import pytest
from synthetic import (
    FixedStrategy,
    daily_shape,
    make_prices,
    without_shortfall_penalty,
    without_soc_enforcement,
)

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
    market_cfg = without_soc_enforcement(market_cfg)  # else the engine cuts them (RC386)
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


def test_shortfall_is_penalised_instead_of_raising(market_cfg, battery_cfg) -> None:
    from battery_optimisation.sim.simulator import ShortfallPolicy

    market_cfg = without_soc_enforcement(market_cfg)
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0] * 3, [100.0] * 3, market_cfg)  # 50 MWh cannot cover it
    dispatch = Simulator(m, Battery(battery_cfg), on_shortfall=ShortfallPolicy.PENALISE).run(
        strategy
    )
    frame = dispatch.frame
    assert frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-9  # physics always respected
    shortfall_mwh = (frame[Col.SHORTFALL_MW] * market_cfg.interval_h).sum()
    assert shortfall_mwh > 0
    assert frame[Col.PENALTY_SGD].sum() == pytest.approx(
        shortfall_mwh * market_cfg.shortfall_penalty.sgd_per_mwh
    )
    delivered = (frame[Col.DISCHARGE_MW] * market_cfg.interval_h).sum()
    assert delivered + shortfall_mwh == pytest.approx(3 * 100.0 * market_cfg.interval_h)
    assert dispatch.total().shortfall_mwh == pytest.approx(shortfall_mwh)


def test_disabled_penalty_books_the_shortfall_at_no_cost(market_cfg, battery_cfg) -> None:
    from battery_optimisation.sim.simulator import ShortfallPolicy

    market_cfg = without_shortfall_penalty(without_soc_enforcement(market_cfg))
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0] * 3, [100.0] * 3, market_cfg)
    dispatch = Simulator(m, Battery(battery_cfg), on_shortfall=ShortfallPolicy.PENALISE).run(
        strategy
    )
    assert dispatch.total().shortfall_mwh > 0
    assert dispatch.frame[Col.PENALTY_SGD].sum() == 0.0


def test_engine_cuts_infeasible_energy_to_the_soc(market_cfg, battery_cfg) -> None:
    """RC386: the engine schedules only what the SoC allows, so there is nothing to penalise."""
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0] * 3, [100.0] * 3, market_cfg)
    dispatch = Simulator(m, Battery(battery_cfg)).run(strategy)  # RAISE: would fail if not cut
    frame = dispatch.frame
    assert frame[Col.SHORTFALL_MW].sum() == 0.0
    assert frame[Col.SOC_CUT_MW].sum() > 0
    gain = battery_cfg.soc_init * battery_cfg.eta_d / market_cfg.interval_h  # MW the SoC allows
    assert frame[Col.DISCHARGE_MW][0] == pytest.approx(min(100.0, gain))
    assert frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-9


def test_dispatch_keeps_every_submitted_offer(market_cfg, battery_cfg, tmp_path) -> None:
    """Every tranche the strategy submitted is kept (offers.parquet) to explain a settlement."""
    m = Market(market_cfg, make_prices([100.0] * 3))
    strategy = FixedStrategy([0.0, 20.0, 0.0], [30.0, 0.0, 0.0], market_cfg)
    dispatch = Simulator(m, Battery(battery_cfg)).run(strategy)
    offers = dispatch.offers
    assert offers.columns == [
        Col.INTERVAL_START,
        Col.PRODUCT,
        Col.SIDE,
        Col.TRANCHE,
        Col.OFFER_PRICE,
        Col.OFFER_MW,
    ]
    assert offers.height == 2
    assert offers.filter(pl.col(Col.SIDE) == "discharge")[Col.OFFER_MW].to_list() == [30.0]
    assert set(offers[Col.PRODUCT].to_list()) == {"energy"}
