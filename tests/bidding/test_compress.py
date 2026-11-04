import pytest

from battery_optimisation.bidding.compress import compress
from battery_optimisation.sim.clearing import Side, Tranche


def curve(prices: list[float], side: Side = Side.DISCHARGE) -> list[Tranche]:
    return [Tranche(side=side, price=p, quantity_mw=10.0) for p in prices]


def test_breakpoints_at_the_largest_price_jumps() -> None:
    out = compress(curve([10, 11, 12, 50, 51, 90]), k=3)
    assert [t.quantity_mw for t in out] == [30.0, 20.0, 10.0]
    assert [t.price for t in out] == pytest.approx([11.0, 50.5, 90.0])  # MW-weighted means


def test_compression_keeps_quantity_order_and_limit() -> None:
    prices = list(range(100, 0, -3))  # a falling charge curve
    out = compress(curve(prices, Side.CHARGE), k=10)
    assert len(out) == 10
    assert sum(t.quantity_mw for t in out) == pytest.approx(10.0 * len(prices))
    assert all(a.price > b.price for a, b in zip(out, out[1:], strict=False))
    assert all(t.side is Side.CHARGE for t in out)


def test_short_curves_are_untouched() -> None:
    c = curve([1, 2, 3])
    assert compress(c, k=10) == c
    assert compress([], k=10) == []
