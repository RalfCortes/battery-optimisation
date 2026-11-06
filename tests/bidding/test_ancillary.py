"""Reserve offer curves: the reserve MW the SDP's gate-closure choice would hold at each price of
one product (others at their expected value), priced at the opportunity cost of each step."""

import numpy as np
import pytest

from battery_optimisation.bidding.ancillary import curve_tranches, reserve_curve
from battery_optimisation.config import Product

DT = 0.5
# candidate mixes (product 0, product 1) and the expected energy value left beside each
X = np.array([[0.0, 0.0], [20.0, 0.0], [50.0, 0.0], [50.0, 30.0]])
BASE = np.array([100.0, 60.0, -100.0, -200.0])
VALUE = np.array([10.0, 5.0]) * DT  # SGD per MW for the interval at the expected prices
GRID = np.arange(0.0, 100.0 + 1e-9, 0.5)


def test_reserve_curve_holds_more_when_the_price_is_higher() -> None:
    mw = reserve_curve(X, BASE, VALUE, np.zeros(2), 0, GRID, DT)
    assert mw.shape == GRID.shape
    assert np.all(np.diff(mw) >= 0)  # rising
    assert mw[0] == 0.0
    assert mw[-1] == 50.0
    # at the expected price the curve holds what the fixed choice holds (argmax of base + v.x)
    expected = X[np.argmax(BASE + X @ VALUE), 0]
    assert mw[np.searchsorted(GRID, 10.0)] == expected == 20.0


def test_curve_steps_sit_at_the_opportunity_cost() -> None:
    """Holding 20 MW costs 40 of energy value: worth it from 4 SGD/MW/h (40 / (20 x 0.5));
    50 MW costs 160 more for 30 MW more: from 10.67 (first grid price above: 11)."""
    mw = reserve_curve(X, BASE, VALUE, np.zeros(2), 0, GRID, DT)
    tranches = curve_tranches(GRID, mw, Product.REGULATION, k=5)
    assert [t.quantity_mw for t in tranches] == pytest.approx([20.0, 30.0])
    assert [t.price for t in tranches] == pytest.approx([4.0, 11.0], abs=1e-9)
    assert all(t.product is Product.REGULATION for t in tranches)


def test_extra_value_per_mw_shifts_the_curve() -> None:
    """Regulation also pays (or costs) its AGC energy and wear: ``extra`` SGD per MW comes on
    top of price x dt, so a positive extra lowers every step's price."""
    plain = reserve_curve(X, BASE, VALUE, np.zeros(2), 0, GRID, DT)
    extra = reserve_curve(X, BASE, VALUE, np.array([1.0, 0.0]), 0, GRID, DT)
    assert np.all(extra >= plain)
    assert curve_tranches(GRID, extra, Product.REGULATION, 5)[0].price == pytest.approx(2.0)


def test_mw_worth_holding_unpaid_is_offered_at_zero() -> None:
    x = np.array([[0.0, 0.0], [30.0, 0.0]])
    mw = reserve_curve(x, np.array([0.0, 10.0]), VALUE, np.zeros(2), 0, GRID, DT)
    tranches = curve_tranches(GRID, mw, Product.PRIMARY_RESERVE, 5)
    assert tranches[0].price == 0.0 and tranches[0].quantity_mw == 30.0


def test_curve_tranches_compress_to_k_and_keep_the_total() -> None:
    grid = np.linspace(0.0, 50.0, 51)
    mw = np.minimum(np.floor(grid / 2.0) * 3.0, 60.0)  # many small steps
    tranches = curve_tranches(grid, mw, Product.CONTINGENCY_RESERVE, k=5)
    assert len(tranches) <= 5
    assert sum(t.quantity_mw for t in tranches) == pytest.approx(60.0)
    prices = [t.price for t in tranches]
    assert prices == sorted(prices)
