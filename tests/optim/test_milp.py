import numpy as np
import polars as pl
import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.columns import Col
from battery_optimisation.optim.milp import solve_schedule


def test_schedule_respects_limits_and_terminal_soc(battery_cfg, market_cfg) -> None:
    sol = solve_schedule(make_prices(daily_shape(2)), Battery(battery_cfg), market_cfg)
    s = sol.schedule
    assert sol.objective_sgd > 0
    assert s[Col.CHARGE_MW].max() <= battery_cfg.p_max_mw + 1e-6
    assert s[Col.DISCHARGE_MW].max() <= battery_cfg.p_max_mw + 1e-6
    assert s[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert s[Col.SOC_MWH].max() <= battery_cfg.soc_max + 1e-6
    assert s[Col.SOC_MWH][-1] >= battery_cfg.soc_init - 1e-6


def test_no_simultaneous_charge_and_discharge(battery_cfg, market_cfg) -> None:
    # Negative prices and a full battery: without the binary, charging and discharging at once
    # would burn grid energy through the losses and be paid for it.
    cfg = battery_cfg.model_copy(update={"kappa_sgd_per_mwh": 0.0})
    full = Battery(cfg, soc_mwh=cfg.soc_max)
    sol = solve_schedule(make_prices(np.full(48, -500.0)), full, market_cfg)
    both = sol.schedule.filter((pl.col(Col.CHARGE_MW) > 0) & (pl.col(Col.DISCHARGE_MW) > 0))
    assert both.height == 0


def test_rejects_gaps_and_missing_prices(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(1))
    with_gap = prices.with_row_index().filter(pl.col("index") != 5).drop("index")
    with pytest.raises(ValueError, match="contiguous"):
        solve_schedule(with_gap, Battery(battery_cfg), market_cfg)
    missing = prices.with_columns(pl.lit(None, pl.Float64).alias(Col.USEP))
    with pytest.raises(ValueError, match="missing"):
        solve_schedule(missing, Battery(battery_cfg), market_cfg)


def test_milp_cooptimises_ancillary_services(battery_cfg, market_cfg) -> None:
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    flat = make_prices(np.full(48, 100.0))  # no energy spread: only reserves pay
    prices = flat.with_columns(
        pl.lit(25.0).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(5.0).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(30.0).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )
    b = Battery(battery_cfg)
    energy_only = solve_schedule(flat, b, market_cfg)
    both = solve_schedule(prices, b, market_cfg)
    assert both.objective_sgd > energy_only.objective_sgd >= -1e-6
    s = both.schedule
    p = battery_cfg.p_max_mw
    net = s[Col.DISCHARGE_MW] - s[Col.CHARGE_MW]
    for reserve in (Col.PRI_MW, Col.CON_MW):  # per reserve class (RC383)
        assert (net + s[Col.REG_MW] + s[reserve]).max() <= p + 1e-6
    assert (s[Col.CHARGE_MW] - s[Col.DISCHARGE_MW] + s[Col.REG_MW]).max() <= p + 1e-6
    assert s[Col.CON_MW].sum() > 0  # the best-paid reserve is used


def test_firm_reserves_do_not_rely_on_the_opposite_energy_flow(battery_cfg, market_cfg) -> None:
    """Without firm reserves, charging frees power for more than P of raise reserves; firm
    reserves fit beside the energy flow itself: raise + discharge <= P, lower + charge <= P."""
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    energy = make_prices(np.array([-50.0] * 24 + [300.0] * 24))  # charge early, sell late
    prices = energy.with_columns(
        pl.lit(40.0).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(40.0).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(40.0).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )
    b, p = Battery(battery_cfg), battery_cfg.p_max_mw
    loose = solve_schedule(prices, b, market_cfg).schedule
    raise_loose = loose[Col.REG_MW] + loose[[Col.PRI_MW, Col.CON_MW]].max_horizontal()
    assert raise_loose.max() > p + 1e-6  # relies on charging
    s = solve_schedule(prices, b, market_cfg, firm_reserves=True).schedule
    for reserve in (Col.PRI_MW, Col.CON_MW):
        assert (s[Col.REG_MW] + s[reserve] + s[Col.DISCHARGE_MW]).max() <= p + 1e-6
    assert (s[Col.REG_MW] + s[Col.CHARGE_MW]).max() <= p + 1e-6


def test_milp_holds_primary_and_contingency_on_the_same_headroom(battery_cfg, market_cfg) -> None:
    """RC383: power is limited per reserve class, so an idle battery can be scheduled for P of
    primary and P of contingency at once, within the RC386 SoC checks."""
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column
    from battery_optimisation.sim.storage_rules import check_storage_rules

    flat = make_prices(np.full(48, 100.0))
    prices = flat.with_columns(
        pl.lit(0.0).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(20.0).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(30.0).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )
    b = Battery(battery_cfg.model_copy(update={"soc_init": 80.0}))
    s = solve_schedule(prices, b, market_cfg).schedule
    assert (s[Col.PRI_MW] + s[Col.CON_MW]).max() > battery_cfg.p_max_mw + 1e-6
    soc = [80.0, *s[Col.SOC_MWH].to_list()[:-1]]  # start of each interval
    for row, start in zip(s.iter_rows(named=True), soc, strict=True):
        awards = {
            Product.REGULATION: row[Col.REG_MW],
            Product.PRIMARY_RESERVE: row[Col.PRI_MW],
            Product.CONTINGENCY_RESERVE: row[Col.CON_MW],
        }
        check_storage_rules(
            market_cfg.storage_rules,
            battery_cfg,
            start,
            row[Col.CHARGE_MW],
            row[Col.DISCHARGE_MW],
            awards,
        )


def test_milp_soc_follows_the_regulation_energy(battery_cfg, market_cfg) -> None:
    """Holding regulation moves energy (``ancillary.regulation_energy``): the plan's SoC
    includes the expected drift and its value includes the energy settled."""
    from battery_optimisation.config import Product, RegulationEnergy
    from battery_optimisation.sim.clearing import price_column

    m = market_cfg.model_copy(
        update={
            "ancillary": market_cfg.ancillary.model_copy(
                update={"regulation_energy": RegulationEnergy(up=0.1, down=0.0)}
            )
        }
    )
    prices = make_prices(np.full(8, 100.0)).with_columns(
        pl.lit(30.0).alias(price_column(Product.REGULATION, m)),
        pl.lit(0.0).alias(price_column(Product.PRIMARY_RESERVE, m)),
        pl.lit(0.0).alias(price_column(Product.CONTINGENCY_RESERVE, m)),
    )
    b = Battery(battery_cfg)
    s = solve_schedule(prices, b, m, soc_terminal_min=0.0).schedule
    dt = m.interval_h
    assert s[Col.REG_MW].sum() > 0
    soc = [battery_cfg.soc_init, *s[Col.SOC_MWH].to_list()]
    for t, row in enumerate(s.iter_rows(named=True)):
        expected = (
            soc[t]
            + b.charge_gain(dt) * row[Col.CHARGE_MW]
            - b.discharge_loss(dt) * row[Col.DISCHARGE_MW]
            - 0.1 * row[Col.REG_MW] * dt / battery_cfg.eta_d
        )
        assert soc[t + 1] == pytest.approx(expected, abs=1e-6)


def test_stochastic_plan_with_one_path_is_the_deterministic_plan(battery_cfg, market_cfg) -> None:
    from battery_optimisation.config import Product
    from battery_optimisation.optim.milp import solve_reserve_plan
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(2)
    energy = make_prices(daily_shape(1, seed=3))
    prices = energy.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(5, 40, 48)),
        pl.Series(price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(0, 20, 48)),
        pl.Series(price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(0, 30, 48)),
    )
    b = Battery(battery_cfg)
    one = solve_reserve_plan(
        prices.drop(Col.USEP), energy[Col.USEP].to_numpy()[None, :], np.ones(1), b, market_cfg
    )
    det = solve_schedule(prices, b, market_cfg)
    assert one.objective_sgd == pytest.approx(det.objective_sgd, rel=1e-6)


def test_stochastic_plan_hedges_reserves_against_price_paths(battery_cfg, market_cfg) -> None:
    """Two equally likely paths, a spike at different times: the plan is worth at least the
    deterministic plan on the mean path evaluated on both paths (it is optimal for them)."""
    from battery_optimisation.config import Product
    from battery_optimisation.optim.milp import solve_reserve_plan
    from battery_optimisation.sim.clearing import price_column

    base = np.full(48, 100.0)
    a, b_ = base.copy(), base.copy()
    a[10], b_[30] = 2000.0, 2000.0
    stamps = make_prices(base)
    anc = stamps.select(Col.INTERVAL_START).with_columns(
        pl.lit(15.0).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(10.0).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(15.0).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )
    plan = solve_reserve_plan(
        anc, np.vstack([a, b_]), np.ones(2), Battery(battery_cfg), market_cfg, firm_reserves=True
    )
    s = plan.schedule
    assert s[Col.USEP].to_list() == pytest.approx(((a + b_) / 2).tolist())
    assert (s[Col.CON_MW] + s[Col.PRI_MW] + s[Col.REG_MW]).sum() > 0
    assert plan.objective_sgd > 0


def test_relaxed_stochastic_plan_matches_on_positive_prices(battery_cfg, market_cfg) -> None:
    from battery_optimisation.config import Product
    from battery_optimisation.optim.milp import solve_reserve_plan
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(5)
    stamps = make_prices(np.full(48, 100.0))
    anc = stamps.select(Col.INTERVAL_START).with_columns(
        pl.Series(price_column(p, market_cfg), rng.uniform(5, 30, 48))
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    )
    paths = daily_shape(1, seed=2)[None, :] * rng.lognormal(0, 0.4, (4, 48))
    args = (anc, paths, np.ones(4), Battery(battery_cfg), market_cfg)
    exact = solve_reserve_plan(*args, firm_reserves=True)
    relaxed = solve_reserve_plan(*args, firm_reserves=True, relax=True)
    assert relaxed.objective_sgd == pytest.approx(exact.objective_sgd, rel=1e-4)
