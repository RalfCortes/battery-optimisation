"""Storage limits of the market engine (NEMS RC383 / RC386): per-class power, SoC checks, and the
engine cutting schedules to what the SoC allows."""

import pytest
from synthetic import make_prices

from battery_optimisation.battery import Battery, LimitBreachError
from battery_optimisation.config import Product
from battery_optimisation.sim.clearing import (
    AncillaryTranche,
    IntervalOffers,
    Market,
    Side,
    Tranche,
    price_column,
)
from battery_optimisation.sim.storage_rules import (
    check_storage_rules,
    draw_mwh,
    enforce_soc,
    fill_mwh,
    lower_mw,
    raise_mw,
    reserve_draw_mwh,
)

REG, PRI, CON = Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE


@pytest.fixture
def ess(battery_cfg):
    """RC386 worked examples: 100 MWh, 99% efficiencies, SoC range 0-100 MWh."""
    return battery_cfg.model_copy(update={"eta_c": 0.99, "eta_d": 0.99})


def test_power_is_limited_per_reserve_class_not_summed(market_cfg) -> None:
    rules = market_cfg.storage_rules
    # idle: 100 MW primary and 100 MW contingency together (each class <= P)
    assert raise_mw(rules, {PRI: 100.0, CON: 100.0}) == 100.0
    # regulation counts in both classes
    assert raise_mw(rules, {REG: 10.0, PRI: 30.0, CON: 60.0}) == 70.0
    assert lower_mw(rules, {REG: 10.0, PRI: 30.0, CON: 60.0}) == 10.0


def test_discharge_checks_match_the_rc386_examples(market_cfg, ess) -> None:
    rules = market_cfg.storage_rules
    awards = {REG: 10.0, PRI: 30.0, CON: 60.0}
    # C1 35 MWh / 0.99 = 35.35; C3 (5 + 5 + 30) / 0.99 = 40.40 binds
    assert draw_mwh(rules, ess, 0.0, 10.0, awards) == pytest.approx(40.0 / 0.99)
    # reserves alone (no energy): C3 (5 + 30) / 0.99
    assert reserve_draw_mwh(rules, ess, awards) == pytest.approx(35.0 / 0.99)


def test_primary_while_charging_needs_the_first_ten_minutes(market_cfg, battery_cfg) -> None:
    """RC386 constraint 2: 60 MW primary while charging 10 MW needs 10 - 1.67 = 8.3 MWh."""
    ideal = battery_cfg.model_copy(update={"eta_c": 1.0, "eta_d": 1.0})
    assert draw_mwh(market_cfg.storage_rules, ideal, 10.0, 0.0, {PRI: 60.0}) == pytest.approx(
        60 / 6 - 10 / 6
    )


def test_charge_check_counts_regulation_but_not_reserves(market_cfg, ess) -> None:
    """RC386 constraint 4: charging 10 MW + 10 MW regulation fills (5 + 5) x 0.99 MWh."""
    rules = market_cfg.storage_rules
    assert fill_mwh(rules, ess, 10.0, 0.0, {REG: 10.0, CON: 100.0}) == pytest.approx(9.9)


@pytest.mark.parametrize(
    ("soc", "discharge", "awards", "match"),
    [
        (50.0, 100.0, {CON: 10.0}, "power"),  # 100 MW discharging + 10 MW contingency > P
        (12.0, 0.0, {CON: 50.0}, "sustain"),  # 50 MW for 30 min needs 25 MWh / eta_d
        (95.0, 0.0, {REG: 20.0}, "sustain"),  # 20 MW down for 30 min needs 10 MWh x eta_c
    ],
)
def test_check_raises_on_breaches(market_cfg, battery_cfg, soc, discharge, awards, match) -> None:
    with pytest.raises(LimitBreachError, match=match):
        check_storage_rules(market_cfg.storage_rules, battery_cfg, soc, 0.0, discharge, awards)


def _market(market_cfg, usep: float, reg: float = 20.0, pri: float = 8.0, con: float = 30.0):
    import polars as pl

    prices = make_prices([usep]).with_columns(
        pl.lit(reg).alias(price_column(REG, market_cfg)),
        pl.lit(pri).alias(price_column(PRI, market_cfg)),
        pl.lit(con).alias(price_column(CON, market_cfg)),
    )
    return Market(market_cfg, prices), prices[0, "interval_start"]


def test_feasible_schedules_are_left_unchanged(market_cfg, battery_cfg) -> None:
    m, at = _market(market_cfg, 100.0)
    o = IntervalOffers(
        interval_start=at,
        tranches=(Tranche(side=Side.DISCHARGE, price=50.0, quantity_mw=20.0),),
        ancillary=(AncillaryTranche(product=CON, price=0.0, quantity_mw=50.0),),
    )
    cleared = m.clear(o)
    capped = enforce_soc(cleared, o, market_cfg.storage_rules, battery_cfg, 50.0)
    assert capped == cleared


def test_engine_cuts_reserves_to_what_the_soc_sustains(market_cfg, battery_cfg) -> None:
    m, at = _market(market_cfg, 100.0)
    o = IntervalOffers(
        interval_start=at,
        ancillary=(
            AncillaryTranche(product=PRI, price=0.0, quantity_mw=100.0),
            AncillaryTranche(product=CON, price=0.0, quantity_mw=100.0),
        ),
    )
    soc, rules = 20.0, market_cfg.storage_rules
    capped = enforce_soc(m.clear(o), o, rules, battery_cfg, soc)
    check_storage_rules(rules, battery_cfg, soc, 0.0, 0.0, capped.awards)  # no raise
    # contingency pays more (30 vs 8): kept up to its own check C3, 20 MWh x eta_d / 0.5 h
    con = soc * battery_cfg.eta_d / 0.5
    assert capped.awards[CON] == pytest.approx(con, rel=1e-6)
    # primary fills what is left of C1 (primary 10 min, then contingency 20 min)
    assert capped.awards[PRI] == pytest.approx((soc * battery_cfg.eta_d - con / 3) * 6, rel=1e-6)


def test_engine_keeps_price_taking_energy_before_reserves(market_cfg, battery_cfg) -> None:
    m, at = _market(market_cfg, 100.0)
    o = IntervalOffers(
        interval_start=at,
        tranches=(Tranche(side=Side.DISCHARGE, price=market_cfg.price_floor, quantity_mw=100.0),),
        ancillary=(AncillaryTranche(product=CON, price=0.0, quantity_mw=100.0),),
    )
    soc, rules = 20.0, market_cfg.storage_rules
    capped = enforce_soc(m.clear(o), o, rules, battery_cfg, soc)
    check_storage_rules(rules, battery_cfg, soc, 0.0, capped.discharge_mw, capped.awards)
    assert capped.discharge_mw == pytest.approx(soc * battery_cfg.eta_d / 0.5, rel=1e-6)
    assert capped.awards[CON] == pytest.approx(0.0, abs=1e-6)
    Battery(battery_cfg, soc).step(capped.charge_mw, capped.discharge_mw, 0.5)  # deliverable
