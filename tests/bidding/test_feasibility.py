from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.feasibility import make_feasible, soc_range_after
from battery_optimisation.sim.clearing import IntervalOffers, Side, Tranche

DT = 0.5
T0 = datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Singapore"))


def offers(*tranches: tuple[Side, float, float]) -> IntervalOffers:
    return IntervalOffers(
        interval_start=T0,
        tranches=tuple(Tranche(side=s, price=p, quantity_mw=q) for s, p, q in tranches),
    )


def test_worst_case_soc_after_pending_offers(battery_cfg) -> None:
    b = Battery(battery_cfg, 50.0)
    pending = [offers((Side.DISCHARGE, 100, 20.0), (Side.CHARGE, 10, 30.0))] * 2
    low, high = soc_range_after(50.0, pending, b, DT)
    assert low == pytest.approx(50.0 - 2 * 20.0 * DT / battery_cfg.eta_d)
    assert high == pytest.approx(50.0 + 2 * 30.0 * DT * battery_cfg.eta_c)


def test_tranches_feasible_from_worst_case_soc(battery_cfg) -> None:
    """CLAUDE.md invariant: whatever clears, the battery stays within limits."""
    b = Battery(battery_cfg, 20.0)
    pending = [offers((Side.DISCHARGE, 100, 15.0))]  # may take ~8.1 MWh before our interval
    wanted = offers(
        (Side.DISCHARGE, 200, 10.0),
        (Side.DISCHARGE, 300, 10.0),
        (Side.DISCHARGE, 400, 10.0),
        (Side.CHARGE, 50, 100.0),
    )
    out = make_feasible(wanted, 20.0, pending, b, DT)
    low, high = soc_range_after(20.0, pending, b, DT)
    dis = [t for t in out.tranches if t.side is Side.DISCHARGE]
    cap = (low - battery_cfg.soc_min) * battery_cfg.eta_d / DT  # ~21.9 MW
    assert [t.price for t in dis] == [200, 300, 400]  # curve order kept, the tail is truncated
    assert [t.quantity_mw for t in dis] == pytest.approx([10.0, 10.0, cap - 20.0])
    assert low - cap * DT / battery_cfg.eta_d == pytest.approx(battery_cfg.soc_min)
    charge = [t for t in out.tranches if t.side is Side.CHARGE]
    assert sum(t.quantity_mw for t in charge) <= battery_cfg.p_max_mw + 1e-9
    assert (
        high + sum(t.quantity_mw for t in charge) * DT * battery_cfg.eta_c
        <= battery_cfg.soc_max + 1e-9
    )


def test_nothing_offered_when_no_headroom(battery_cfg) -> None:
    b = Battery(battery_cfg, battery_cfg.soc_min)
    out = make_feasible(offers((Side.DISCHARGE, 100, 50.0)), battery_cfg.soc_min, [], b, DT)
    assert out.tranches == ()


def test_expected_soc_projects_pending_offers_through_scenarios(battery_cfg) -> None:
    from battery_optimisation.bidding.feasibility import expected_soc_after

    b = Battery(battery_cfg, 50.0)
    sell = offers((Side.DISCHARGE, 100.0, 20.0), (Side.DISCHARGE, 300.0, 20.0))
    buy = offers((Side.CHARGE, 50.0, 30.0))
    # scenario prices for the two pending intervals: [interval, scenario]
    prices = [[150.0, 400.0], [40.0, 60.0]]
    weights = [0.5, 0.5]
    soc = expected_soc_after(50.0, [sell, buy], prices, weights, b, DT)
    # interval 1: price 150 -> 20 MW sold, price 400 -> 40 MW sold: expected 30 MW
    # interval 2: price 40 -> 30 MW bought, price 60 -> nothing: expected 15 MW
    loss, gain = DT / battery_cfg.eta_d, battery_cfg.eta_c * DT
    assert soc == pytest.approx(50.0 - 30.0 * loss + 15.0 * gain)


def test_price_taking_tranches_count_as_cleared(battery_cfg) -> None:
    """A charge bid at the cap (or discharge offer at the floor) always clears: the worst case
    for later intervals must count it, otherwise energy we will surely have is not offered."""
    b = Battery(battery_cfg, 10.0)
    floor, cap = -4500.0, 4500.0
    sure_buy = offers((Side.CHARGE, cap, 40.0), (Side.DISCHARGE, 300.0, 5.0))
    low, high = soc_range_after(10.0, [sure_buy], b, DT, floor=floor, cap=cap)
    # lowest: the sure 40 MW buy happens and, at worst, the 5 MW sale too (netted to 35 MW buy)
    assert low == pytest.approx(10.0 + 35.0 * DT * battery_cfg.eta_c)
    assert high == pytest.approx(10.0 + 40.0 * DT * battery_cfg.eta_c)

    sure_sell = offers((Side.DISCHARGE, floor, 10.0))
    low, high = soc_range_after(50.0, [sure_sell], b, DT, floor=floor, cap=cap)
    assert low == high == pytest.approx(50.0 - 10.0 * DT / battery_cfg.eta_d)
