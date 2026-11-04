import pytest
from synthetic import make_prices

from battery_optimisation.columns import Col
from battery_optimisation.sim.clearing import IntervalOffers, Market, Side, Tranche


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


def test_history_before_the_window_is_visible_but_not_traded(market_cfg) -> None:
    prices = make_prices([10.0] * 48 + [20.0] * 48)
    before, window = prices.slice(0, 48), prices.slice(48)
    m = Market(market_cfg, window, history=before)
    assert m.intervals == window[Col.INTERVAL_START].to_list()  # only the window is traded
    view = m.view(3)
    gate = market_cfg.gate_closure_intervals
    assert view.history().height == 48 + 3 - gate  # known up to the decision time
    assert view.history()[Col.INTERVAL_START].max() < view.decided_at
