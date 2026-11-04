import numpy as np
import pytest

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.water_value import offer_curve
from battery_optimisation.optim.sdp import ValueFunction
from battery_optimisation.sim.clearing import Side

DT = 0.5


def concave_vf(battery_cfg) -> ValueFunction:
    soc = np.linspace(battery_cfg.soc_min, battery_cfg.soc_max, 101)
    v = 400 * soc - 2 * soc**2  # slope 400 - 4 s: decreasing in SOC
    return ValueFunction(soc=soc, values=np.stack([v, v]), solve_time_s=0.0)


def test_discharge_prices_rise_and_charge_prices_fall(battery_cfg) -> None:
    """CLAUDE.md invariant: discharge offer prices increase with cumulative MW."""
    discharge, charge = offer_curve(
        concave_vf(battery_cfg), 0, 50.0, Battery(battery_cfg, 50.0), DT
    )
    assert all(t.side is Side.DISCHARGE for t in discharge)
    assert all(t.side is Side.CHARGE for t in charge)
    assert np.all(np.diff([t.price for t in discharge]) > 0)
    assert np.all(np.diff([t.price for t in charge]) < 0)
    assert max(t.price for t in charge) < min(t.price for t in discharge)  # never buy to sell


def test_tranche_price_is_the_water_value_formula(battery_cfg) -> None:
    b = Battery(battery_cfg, 50.0)
    discharge, charge = offer_curve(concave_vf(battery_cfg), 0, 50.0, b, DT)
    # first discharge tranche spans SOC 50 -> 49: slope of V there = (V(50) - V(49)) / 1
    slope = (400 * 50 - 2 * 50**2) - (400 * 49 - 2 * 49**2)
    k = battery_cfg.kappa_sgd_per_mwh
    assert discharge[0].price == pytest.approx(slope / battery_cfg.eta_d + k)
    assert discharge[0].quantity_mw == pytest.approx(1.0 * battery_cfg.eta_d / DT)
    up = (400 * 51 - 2 * 51**2) - (400 * 50 - 2 * 50**2)
    assert charge[0].price == pytest.approx(up * battery_cfg.eta_c - k)
    assert charge[0].quantity_mw == pytest.approx(1.0 / (battery_cfg.eta_c * DT))


def test_quantities_cover_the_feasible_range(battery_cfg) -> None:
    vf = concave_vf(battery_cfg)
    for soc in (0.0, 30.3, 99.5):
        b = Battery(battery_cfg, soc)
        discharge, charge = offer_curve(vf, 0, soc, b, DT)
        assert sum(t.quantity_mw for t in discharge) == pytest.approx(b.max_discharge_mw(DT))
        assert sum(t.quantity_mw for t in charge) == pytest.approx(b.max_charge_mw(DT))


def test_trim_to_price_band_drops_offers_that_cannot_clear() -> None:
    from battery_optimisation.bidding.water_value import trim_to_band
    from battery_optimisation.sim.clearing import Tranche

    discharge = [Tranche(side=Side.DISCHARGE, price=p, quantity_mw=5.0) for p in (100, 200, 900)]
    charge = [Tranche(side=Side.CHARGE, price=p, quantity_mw=5.0) for p in (80, 40, -300)]
    d, c = trim_to_band(discharge, charge, low=0.0, high=500.0)
    assert [t.price for t in d] == [100, 200]  # 900 is above every plausible price
    assert [t.price for t in c] == [80, 40]  # -300 is below every plausible price


def test_tranches_beyond_every_plausible_price_become_price_taking() -> None:
    from battery_optimisation.bidding.water_value import price_take_beyond_band
    from battery_optimisation.sim.clearing import Tranche

    discharge = [Tranche(side=Side.DISCHARGE, price=p, quantity_mw=5.0) for p in (50, 150)]
    charge = [Tranche(side=Side.CHARGE, price=p, quantity_mw=5.0) for p in (300, 120)]
    d, c = price_take_beyond_band(discharge, charge, low=100.0, high=200.0, floor=-4500, cap=4500)
    assert [t.price for t in d] == [-4500, 150]  # 50 < every plausible price: sell for sure
    assert [t.price for t in c] == [4500, 120]  # 300 > every plausible price: buy for sure
