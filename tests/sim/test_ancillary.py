"""Ancillary services: clearing, settlement, sustainability (CLAUDE.md invariant)."""

import polars as pl
import pytest
from synthetic import make_prices

from battery_optimisation.battery import Battery, LimitBreachError
from battery_optimisation.columns import Col
from battery_optimisation.config import Product
from battery_optimisation.sim.clearing import (
    AncillaryTranche,
    IntervalOffers,
    Market,
    Side,
    Tranche,
    price_column,
)
from battery_optimisation.sim.simulator import Simulator


def with_ancillary(
    prices: pl.DataFrame, market_cfg, reg: float, pri: float, con: float
) -> pl.DataFrame:
    return prices.with_columns(
        pl.lit(reg).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(pri).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(con).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )


class Fixed:
    def __init__(self, offers_by_index, market_cfg):
        self.offers_by_index, self.i = offers_by_index, 0

    def offers(self, view, soc_mwh):
        o = self.offers_by_index(self.i, view.now)
        self.i += 1
        return o


def test_reserve_group_selects_price_columns(market_cfg) -> None:
    assert price_column(Product.REGULATION, market_cfg) == Col.REGULATION
    assert price_column(Product.PRIMARY_RESERVE, market_cfg) == Col.PRIMARY_A
    assert price_column(Product.CONTINGENCY_RESERVE, market_cfg) == Col.CONTINGENCY_A


def test_ancillary_offers_clear_pay_as_clear_and_settle(market_cfg, battery_cfg) -> None:
    prices = with_ancillary(make_prices([100.0] * 2), market_cfg, reg=20.0, pri=8.0, con=30.0)
    m = Market(market_cfg, prices)

    def offers(i, now):
        return IntervalOffers(
            interval_start=now,
            ancillary=(
                AncillaryTranche(
                    product=Product.REGULATION, price=15.0, quantity_mw=10.0
                ),  # clears
                AncillaryTranche(
                    product=Product.PRIMARY_RESERVE, price=9.0, quantity_mw=10.0
                ),  # no
                AncillaryTranche(product=Product.CONTINGENCY_RESERVE, price=0.0, quantity_mw=20.0),
            ),
        )

    dispatch = Simulator(m, Battery(battery_cfg)).run(Fixed(offers, market_cfg))
    f = dispatch.frame
    dt = market_cfg.interval_h
    assert f[Col.REG_MW].to_list() == [10.0, 10.0]
    assert f[Col.PRI_MW].to_list() == [0.0, 0.0]
    assert f[Col.CON_MW].to_list() == [20.0, 20.0]
    assert f[Col.ANCILLARY_SGD][0] == pytest.approx((20.0 * 10 + 30.0 * 20) * dt)
    u, k = market_cfg.ancillary.regulation_utilisation, battery_cfg.kappa_sgd_per_mwh
    assert f[Col.DEGRADATION_SGD][0] == pytest.approx(k * u * 10.0 * dt)  # regulation throughput
    assert f[Col.SOC_MWH].to_list() == [battery_cfg.soc_init] * 2  # no energy moved
    assert dispatch.total().ancillary_sgd == pytest.approx(f[Col.ANCILLARY_SGD].sum())


def test_awards_are_capped_by_the_product_limit(market_cfg, battery_cfg) -> None:
    capped = market_cfg.model_copy(
        update={
            "ancillary": market_cfg.ancillary.model_copy(
                update={
                    "max_award_mw": {
                        Product.REGULATION: 5.0,
                        Product.PRIMARY_RESERVE: 100.0,
                        Product.CONTINGENCY_RESERVE: 100.0,
                    }
                }
            )
        }
    )
    prices = with_ancillary(make_prices([100.0]), capped, reg=20.0, pri=8.0, con=30.0)
    o = IntervalOffers(
        interval_start=prices[0, Col.INTERVAL_START],
        ancillary=(AncillaryTranche(product=Product.REGULATION, price=0.0, quantity_mw=50.0),),
    )
    assert Market(capped, prices).clear(o).awards[Product.REGULATION] == 5.0


@pytest.mark.parametrize(
    ("soc", "discharge", "con", "match"),
    [
        (50.0, 100.0, 10.0, "power"),  # 100 MW discharging + 10 MW raise reserve > 100 MW
        (12.0, 0.0, 50.0, "sustain"),  # 50 MW for 30 min needs 25 MWh / eta_d of SOC above min
    ],
)
def test_unsustainable_awards_raise(market_cfg, battery_cfg, soc, discharge, con, match) -> None:
    cfg = battery_cfg.model_copy(update={"soc_init": soc})
    prices = with_ancillary(make_prices([100.0]), market_cfg, reg=0.0, pri=0.0, con=30.0)
    tranches = (
        (Tranche(side=Side.DISCHARGE, price=market_cfg.price_floor, quantity_mw=discharge),)
        if discharge
        else ()
    )
    o = IntervalOffers(
        interval_start=prices[0, Col.INTERVAL_START],
        tranches=tranches,
        ancillary=(
            AncillaryTranche(product=Product.CONTINGENCY_RESERVE, price=0.0, quantity_mw=con),
        ),
    )
    with pytest.raises(LimitBreachError, match=match):
        Simulator(Market(market_cfg, prices), Battery(cfg)).run(Fixed(lambda i, now: o, market_cfg))
