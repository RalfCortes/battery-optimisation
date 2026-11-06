"""Water-value bidding through the simulator with gate closure."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import daily_shape, make_prices, without_soc_enforcement

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy, WaterValueStrategy
from battery_optimisation.columns import Col
from battery_optimisation.config import load_bidding
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market, Side
from battery_optimisation.sim.simulator import Simulator

DAYS = 3


def classic_bidding():
    """Worst-case feasibility and MILP reserve plan: the setup these tests were built on (the
    defaults, market feasibility and SDP reserves, have their own tests below)."""
    from battery_optimisation.config import FeasibilityMode, ReservePlanner

    b = load_bidding()
    return b.model_copy(
        update={
            "feasibility": b.feasibility.model_copy(update={"mode": FeasibilityMode.WORST_CASE}),
            "reserve_planner": ReservePlanner.MILP,
        }
    )


def scenario_frame(prices: pl.DataFrame, n: int, noise: float, gap: int, seed: int) -> pl.DataFrame:
    """Per delivery day, issued at gate closure: realised + noise (n paths, equal weights)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _, part in prices.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True):
        stamps, y = part[Col.INTERVAL_START], part[Col.USEP].to_numpy()
        issued = stamps[0] - timedelta(minutes=30 * gap)
        for k in range(n):
            frames.append(
                pl.DataFrame(
                    {
                        Col.ISSUED_AT: [issued] * len(y),
                        Col.SCENARIO: [k] * len(y),
                        Col.WEIGHT: [1 / n] * len(y),
                        Col.INTERVAL_START: stamps,
                        Col.USEP_FORECAST: y + rng.normal(0, noise, len(y)),
                    }
                )
            )
    return pl.concat(frames).with_columns(pl.col(Col.SCENARIO).cast(pl.Int32))


def run(strategy, prices, battery_cfg, market_cfg):
    return Simulator(Market(market_cfg, prices), Battery(battery_cfg)).run(strategy)


def test_offer_curves_never_breach_limits(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(DAYS, seed=4))
    scen = scenario_frame(prices, n=20, noise=80.0, gap=market_cfg.gate_closure_intervals, seed=1)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding())
    dispatch = run(strategy, prices, battery_cfg, market_cfg)  # raises on any breach
    assert dispatch.frame.height == DAYS * 48
    assert dispatch.frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert dispatch.profit_sgd > 0
    assert len(strategy.value_functions) == DAYS


def test_perfect_scenarios_approach_the_benchmark(battery_cfg, market_cfg) -> None:
    """One exact scenario: bidding the water value is close to the daily perfect-foresight MILP."""
    prices = make_prices(daily_shape(DAYS, seed=6))
    scen = scenario_frame(prices, n=1, noise=0.0, gap=market_cfg.gate_closure_intervals, seed=0)
    bid = run(
        WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding()),
        prices,
        battery_cfg,
        market_cfg,
    )
    pf = run(
        ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY),
        prices,
        battery_cfg,
        market_cfg,
    )
    assert bid.profit_sgd == pytest.approx(pf.profit_sgd, rel=0.05)


def test_offers_respect_tranche_limit_and_price_bounds(battery_cfg, market_cfg) -> None:
    prices = make_prices(daily_shape(1, seed=2))
    scen = scenario_frame(prices, n=20, noise=200.0, gap=market_cfg.gate_closure_intervals, seed=3)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    for offers in strategy.submitted.values():
        for side in (s for s in {t.side for t in offers.tranches}):
            assert len(offers.on(side)) <= market_cfg.n_tranches
        assert all(
            market_cfg.price_floor <= t.price <= market_cfg.price_cap for t in offers.tranches
        )


def test_bidding_report(battery_cfg, market_cfg) -> None:
    from synthetic_forecast import normal_quantiles, quantile_frame

    from battery_optimisation.bidding.report import evaluate_bidding
    from battery_optimisation.config import load_forecast
    from battery_optimisation.sim.backtest import StrategyName

    prices = make_prices(daily_shape(DAYS, seed=8))
    gap = market_cfg.gate_closure_intervals
    scen = scenario_frame(prices, n=20, noise=60.0, gap=gap, seed=2)
    levels = load_forecast().quantile_levels
    qf = pl.concat(
        quantile_frame(
            part[Col.INTERVAL_START][0] - timedelta(minutes=30 * gap),
            normal_quantiles(part[Col.USEP].to_numpy(), 60.0, levels),
            levels,
            gap=gap,
        )
        for _, part in prices.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True)
    )
    from battery_optimisation.config import PriceModel

    independent = classic_bidding().model_copy(update={"price_model": PriceModel.INDEPENDENT})
    ev = evaluate_bidding(scen, qf, prices, battery_cfg, market_cfg, independent)
    r = ev.report
    by_name = {c.strategy: c for c in r.results}
    assert set(by_name) == {
        StrategyName.PERFECT_FORESIGHT,
        StrategyName.CHRONOS_MEDIAN,
        StrategyName.WATER_VALUE,
    }
    wv, ss = by_name[StrategyName.WATER_VALUE], by_name[StrategyName.CHRONOS_MEDIAN]
    assert r.uplift_sgd == pytest.approx(wv.profit_sgd - ss.profit_sgd)
    assert r.uplift_pct == pytest.approx(r.uplift_sgd / abs(ss.profit_sgd))
    assert sum(m.uplift_sgd for m in r.monthly) == pytest.approx(r.uplift_sgd)
    assert by_name[StrategyName.PERFECT_FORESIGHT].capture_vs_perfect_day == pytest.approx(1.0)
    assert wv.capture_vs_perfect_day <= 1.0 + 1e-9
    assert r.sdp_days == DAYS


def test_sweep_and_comparison_table(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.report import ComparisonInput, comparison_table
    from battery_optimisation.bidding.sweep import sweep
    from battery_optimisation.sim.backtest import StrategyName

    prices = make_prices(daily_shape(2, seed=9))
    scen = scenario_frame(prices, n=20, noise=60.0, gap=market_cfg.gate_closure_intervals, seed=4)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    stats, per_interval = sweep(strategy, np.linspace(-100, 600, 71))
    assert stats.intervals == 2 * 48 == per_interval.height
    assert stats.prices == 71
    assert 0 <= stats.share_matching <= 1
    assert stats.max_gap_mw <= 2 * battery_cfg.p_max_mw

    pf, ss, wv = (
        StrategyName.PERFECT_FORESIGHT,
        StrategyName.CHRONOS_MEDIAN,
        StrategyName.WATER_VALUE,
    )
    table = comparison_table(
        [
            ComparisonInput(
                strategy=pf, horizon=Horizon.FULL, profit_sgd=110.0, cycles_per_day=2.0
            ),
            ComparisonInput(strategy=pf, horizon=Horizon.DAY, profit_sgd=100.0, cycles_per_day=2.0),
            ComparisonInput(strategy=ss, horizon=Horizon.DAY, profit_sgd=40.0, cycles_per_day=1.0),
            ComparisonInput(strategy=wv, horizon=Horizon.DAY, profit_sgd=55.0, cycles_per_day=1.0),
        ],
        baseline=ss,
    )
    row = table.filter(pl.col("strategy") == wv).to_dicts()[0]
    assert row["vs_baseline_sgd"] == pytest.approx(15.0)
    assert row["vs_baseline_pct"] == pytest.approx(0.375)
    assert row["capture_vs_perfect_day"] == pytest.approx(0.55)
    assert row["capture_vs_perfect_full"] == pytest.approx(0.5)


def test_sweep_breakdown_by_pipeline_step(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.sweep import CurveStep, sweep_breakdown

    prices = make_prices(daily_shape(2, seed=9))
    scen = scenario_frame(prices, n=20, noise=60.0, gap=market_cfg.gate_closure_intervals, seed=4)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding())
    run(strategy, prices, battery_cfg, market_cfg)
    steps = sweep_breakdown(strategy, np.linspace(-100, 600, 71))
    assert [s.step for s in steps] == list(CurveStep)
    gaps = {s.step: s.mean_abs_gap_in_band_mw for s in steps}
    assert gaps[CurveStep.RAW] == pytest.approx(0.0, abs=1e-6)  # the curve is the optimum
    stats, _ = __import__("battery_optimisation.bidding.sweep", fromlist=["sweep"]).sweep(
        strategy, np.linspace(-100, 600, 71)
    )
    # replaying every step reproduces exactly what was submitted
    assert gaps[CurveStep.FEASIBLE] == pytest.approx(stats.mean_abs_gap_in_band_mw)


def test_water_value_uses_one_sdp_per_forecast_issue(battery_cfg, market_cfg) -> None:
    """Intraday issues: each decision uses the latest issue (its own SDP); daily issues: one."""
    prices = make_prices(daily_shape(1, seed=6))
    gap = market_cfg.gate_closure_intervals
    issues = []
    for k in range(48):  # one issue per decision, covering the next 48 intervals (clipped)
        part = prices.slice(k)
        frame = scenario_frame(part, n=1, noise=0.0, gap=gap, seed=0)
        first = part[Col.INTERVAL_START][0]
        issues.append(
            frame.with_columns(
                pl.lit(first - timedelta(minutes=30 * gap))
                .dt.cast_time_unit("us")
                .alias(Col.ISSUED_AT)
            )
        )
    scen = pl.concat(issues)
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding())
    bid = run(strategy, prices, battery_cfg, market_cfg)
    assert strategy.n_plans == 48
    pf = run(
        ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY),
        prices,
        battery_cfg,
        market_cfg,
    )
    assert bid.profit_sgd == pytest.approx(pf.profit_sgd, rel=0.05)


def test_probabilistic_feasibility_offers_more_and_reports_shortfall(
    battery_cfg, market_cfg
) -> None:
    from battery_optimisation.config import FeasibilityMode
    from battery_optimisation.sim.simulator import ShortfallPolicy

    market_cfg = without_soc_enforcement(market_cfg)  # shortfalls reach the battery
    prices = make_prices(daily_shape(DAYS, seed=4))
    scen = scenario_frame(prices, n=20, noise=80.0, gap=market_cfg.gate_closure_intervals, seed=1)
    worst_cfg = classic_bidding()
    prob_cfg = worst_cfg.model_copy(
        update={
            "feasibility": worst_cfg.feasibility.model_copy(
                update={"mode": FeasibilityMode.PROBABILISTIC}
            )
        }
    )
    worst = WaterValueStrategy(scen, battery_cfg, market_cfg, worst_cfg)
    run(worst, prices, battery_cfg, market_cfg)
    prob = WaterValueStrategy(scen, battery_cfg, market_cfg, prob_cfg)
    dispatch = Simulator(
        Market(market_cfg, prices), Battery(battery_cfg), on_shortfall=ShortfallPolicy.PENALISE
    ).run(prob)

    def offered(strategy) -> float:
        return sum(t.quantity_mw for o in strategy.submitted.values() for t in o.tranches)

    assert offered(prob) > offered(worst)
    frame = dispatch.frame
    assert frame[Col.SOC_MWH].min() >= battery_cfg.soc_min - 1e-6
    assert frame[Col.SOC_MWH].max() <= battery_cfg.soc_max + 1e-6
    total = dispatch.total()
    assert total.penalty_sgd == pytest.approx(
        total.shortfall_mwh * market_cfg.shortfall_penalty.sgd_per_mwh
    )


def test_water_value_with_ancillary_services(battery_cfg, market_cfg) -> None:
    """Ancillary commitments from a plan, energy curves within the headroom left: no limit or
    sustainability breach (the simulator raises), ancillary revenue earned, more profit than
    energy-only bidding when reserves pay well."""
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(15, 35, base.height)),
        pl.Series(
            price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(5, 15, base.height)
        ),
        pl.Series(
            price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(10, 30, base.height)
        ),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    scen = scenario_frame(
        window.select(Col.INTERVAL_START, Col.USEP),
        n=10,
        noise=40.0,
        gap=market_cfg.gate_closure_intervals,
        seed=2,
    )

    def run_hist(strategy):
        return Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)

    energy_only = run_hist(WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding()))
    both = WaterValueStrategy(scen, battery_cfg, market_cfg, classic_bidding(), ancillary=True)
    dispatch = run_hist(both)
    assert dispatch.total().ancillary_sgd > 0
    assert dispatch.profit_sgd > energy_only.profit_sgd


def test_water_value_reserve_plan_uses_the_given_ancillary_forecaster(
    battery_cfg, market_cfg
) -> None:
    from battery_optimisation.config import Product
    from battery_optimisation.forecast.perfect import PerfectForecaster
    from battery_optimisation.sim.clearing import price_column

    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.lit(25.0).alias(price_column(Product.REGULATION, market_cfg)),
        pl.lit(10.0).alias(price_column(Product.PRIMARY_RESERVE, market_cfg)),
        pl.lit(20.0).alias(price_column(Product.CONTINGENCY_RESERVE, market_cfg)),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    scen = scenario_frame(
        window.select(Col.INTERVAL_START, Col.USEP),
        n=5,
        noise=40.0,
        gap=market_cfg.gate_closure_intervals,
        seed=2,
    )

    class Spy:
        calls = 0

        def __init__(self, inner):
            self.inner = inner

        def forecast(self, view, end):
            Spy.calls += 1
            return self.inner.forecast(view, end)

    spy = Spy(PerfectForecaster(prices))
    strategy = WaterValueStrategy(
        scen, battery_cfg, market_cfg, classic_bidding(), ancillary=True, ancillary_forecaster=spy
    )
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    assert Spy.calls > 0
    assert dispatch.total().ancillary_sgd > 0


@pytest.mark.parametrize("firm", [True, False])
def test_energy_bids_do_not_crowd_out_planned_reserves(battery_cfg, market_cfg, firm) -> None:
    """Energy curves at t keep the SOC the next interval's planned reserves need, so reserves are
    offered as planned (only plan changes between issues may scale them). Unless reserves are
    firm, the charge a plan needs to hold more than P of raise is committed price-taking."""
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import AWARD_COLUMN, price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(15, 35, base.height)),
        pl.Series(
            price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(5, 15, base.height)
        ),
        pl.Series(
            price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(10, 30, base.height)
        ),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    scen = scenario_frame(
        window.select(Col.INTERVAL_START, Col.USEP),
        n=10,
        noise=40.0,
        gap=market_cfg.gate_closure_intervals,
        seed=2,
    )
    strategy = WaterValueStrategy(
        scen, battery_cfg, market_cfg, classic_bidding(), ancillary=True, firm_reserves=firm
    )
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    planned = sum(sum(a.values()) for a in strategy.planned_awards.values())
    awarded = dispatch.frame.select(sum(pl.col(c) for c in AWARD_COLUMN.values())).to_series().sum()
    # hybrid plans count the RC386 charging credit, which worst-case scaling does not
    assert awarded >= (0.97 if firm else 0.94) * planned
    f = dispatch.frame
    # largest reserve class (RC383: power limited per class, regulation counts in both)
    most_raise = (f[Col.REG_MW] + f[[Col.PRI_MW, Col.CON_MW]].max_horizontal()).max()
    if firm:
        assert most_raise <= battery_cfg.p_max_mw + 1e-6
    else:
        assert most_raise > battery_cfg.p_max_mw + 1  # backed by committed charging


def quantiles_for(prices: pl.DataFrame, sigma: float, gap: int) -> pl.DataFrame:
    from synthetic_forecast import normal_quantiles, quantile_frame

    from battery_optimisation.config import load_forecast

    levels = load_forecast().quantile_levels
    return pl.concat(
        quantile_frame(
            part[Col.INTERVAL_START][0] - timedelta(minutes=30 * gap),
            normal_quantiles(part[Col.USEP].to_numpy(), sigma, levels),
            levels,
            gap=gap,
        )
        for _, part in prices.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True)
    )


def markov_bidding(rho: float | None = 0.8):
    from battery_optimisation.config import PriceModel

    bidding = classic_bidding()
    markov = bidding.markov.model_copy(update={"rho": rho})
    return bidding.model_copy(update={"price_model": PriceModel.MARKOV, "markov": markov})


def continuous_bidding(rho: float | None = 0.8):
    from battery_optimisation.config import PriceModel

    return markov_bidding(rho).model_copy(update={"price_model": PriceModel.CONTINUOUS})


def test_continuous_water_value_never_breaches_limits(battery_cfg, market_cfg) -> None:
    """Continuous price state (score nodes, hat transitions) through the same SDP and offers."""
    from battery_optimisation.bidding.stages import stage_source

    prices = make_prices(daily_shape(DAYS, seed=4))
    qf = quantiles_for(prices, 80.0, market_cfg.gate_closure_intervals)
    bidding = continuous_bidding()
    stages = stage_source(pl.DataFrame(), qf, bidding)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding)
    dispatch = run(strategy, prices, battery_cfg, market_cfg)
    assert dispatch.profit_sgd > 0
    issue = next(iter(strategy.value_functions))
    assert strategy.stage_prices(issue).shape == (48, bidding.continuous.nodes)


def test_markov_water_value_never_breaches_limits(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.stages import LatticeStages

    prices = make_prices(daily_shape(DAYS, seed=4))
    qf = quantiles_for(prices, 80.0, market_cfg.gate_closure_intervals)
    bidding = markov_bidding()
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding)
    dispatch = run(strategy, prices, battery_cfg, market_cfg)
    assert dispatch.profit_sgd > 0
    assert len(strategy.value_functions) == DAYS
    issue = next(iter(strategy.value_functions))
    assert strategy.stage_prices(issue).shape == (48, bidding.markov.states)


@pytest.mark.parametrize("continuous", [False, True])
def test_markov_with_exact_quantiles_approaches_the_benchmark(
    battery_cfg, market_cfg, continuous
) -> None:
    from battery_optimisation.bidding.stages import stage_source

    prices = make_prices(daily_shape(DAYS, seed=6))
    qf = quantiles_for(prices, 1e-3, market_cfg.gate_closure_intervals)
    bidding = continuous_bidding() if continuous else markov_bidding()
    stages = stage_source(pl.DataFrame(), qf, bidding)
    bid = run(
        WaterValueStrategy(stages, battery_cfg, market_cfg, bidding),
        prices,
        battery_cfg,
        market_cfg,
    )
    pf = run(
        ScheduleStrategy(PerfectForecaster(prices), battery_cfg, market_cfg, Horizon.DAY),
        prices,
        battery_cfg,
        market_cfg,
    )
    assert bid.profit_sgd == pytest.approx(pf.profit_sgd, rel=0.05)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("rho", [0.8, None])
def test_bidding_report_with_markov_prices(battery_cfg, market_cfg, rho, continuous) -> None:
    """``evaluate_bidding`` builds the lattice from the quantiles; rho from the config, else
    from the walk-forward copula of each month."""
    from datetime import date

    from battery_optimisation.bidding.report import evaluate_bidding

    prices = make_prices(daily_shape(DAYS, seed=8))
    gap = market_cfg.gate_closure_intervals
    qf = quantiles_for(prices, 60.0, gap)
    scen = scenario_frame(prices, n=5, noise=60.0, gap=gap, seed=2)
    ev = evaluate_bidding(
        scen,
        qf,
        prices,
        battery_cfg,
        market_cfg,
        continuous_bidding(rho) if continuous else markov_bidding(rho),
        copula_lag1={date(2026, 1, 1): 0.7},
    )
    assert ev.report.sdp_days == DAYS
    states = ev.water_value.stage_prices(next(iter(ev.water_value.value_functions))).shape[1]
    assert states == (21 if continuous else 11)


def test_markov_water_value_with_ancillary_services(battery_cfg, market_cfg) -> None:
    """Reserve plan + Markov price states: awards stay sustainable (the simulator checks) and
    reserves add profit."""
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(15, 35, base.height)),
        pl.Series(
            price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(5, 15, base.height)
        ),
        pl.Series(
            price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(10, 30, base.height)
        ),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    qf = quantiles_for(
        window.select(Col.INTERVAL_START, Col.USEP), 40.0, market_cfg.gate_closure_intervals
    )
    bidding = markov_bidding()

    def run_hist(ancillary: bool):
        stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
        strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=ancillary)
        return Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)

    energy_only, both = run_hist(False), run_hist(True)
    assert both.total().ancillary_sgd > 0
    assert both.profit_sgd > energy_only.profit_sgd


@pytest.mark.parametrize("mode", ["worst_case", "market"])
def test_sdp_chooses_reserves_inside_the_water_value(battery_cfg, market_cfg, mode) -> None:
    """Reserve planner ``sdp``: the SDP values reserves per SOC and energy action, the offers
    carry the chosen mix and an energy curve that fits beside it; nothing breaches."""
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.config import FeasibilityMode, Product, ReservePlanner
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(15, 35, base.height)),
        pl.Series(
            price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(5, 15, base.height)
        ),
        pl.Series(
            price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(10, 30, base.height)
        ),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    qf = quantiles_for(
        window.select(Col.INTERVAL_START, Col.USEP), 40.0, market_cfg.gate_closure_intervals
    )
    base_cfg = markov_bidding()
    bidding = base_cfg.model_copy(
        update={
            "reserve_planner": ReservePlanner.SDP,
            "feasibility": base_cfg.feasibility.model_copy(update={"mode": FeasibilityMode(mode)}),
        }
    )
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=True)
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    assert dispatch.total().ancillary_sgd > 0
    assert dispatch.total().shortfall_mwh == 0.0
    assert strategy.planned_awards  # chosen at each decision
    for offers in strategy.submitted.values():
        sold = [t.quantity_mw for t in offers.on(Side.DISCHARGE)]
        assert all(q >= 0 for q in sold)


def test_priced_reserve_offers(battery_cfg, market_cfg) -> None:
    """``reserve_offers: curve``: each product is offered as a rising curve priced at its
    opportunity cost (at most K pairs, within the award caps) instead of a fixed MW at 0; the
    engine settles conflicts with energy; nothing breaches and reserves still earn."""
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.config import (
        FeasibilityMode,
        Product,
        ReserveOffers,
        ReservePlanner,
    )
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(Product.REGULATION, market_cfg), rng.uniform(15, 35, base.height)),
        pl.Series(
            price_column(Product.PRIMARY_RESERVE, market_cfg), rng.uniform(5, 15, base.height)
        ),
        pl.Series(
            price_column(Product.CONTINGENCY_RESERVE, market_cfg), rng.uniform(10, 30, base.height)
        ),
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    qf = quantiles_for(
        window.select(Col.INTERVAL_START, Col.USEP), 40.0, market_cfg.gate_closure_intervals
    )
    base_cfg = markov_bidding()
    bidding = base_cfg.model_copy(
        update={
            "reserve_planner": ReservePlanner.SDP,
            "reserve_offers": ReserveOffers.CURVE,
            "feasibility": base_cfg.feasibility.model_copy(update={"mode": FeasibilityMode.MARKET}),
        }
    )
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=True)
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    assert dispatch.total().ancillary_sgd > 0
    assert dispatch.total().shortfall_mwh == 0.0
    priced = 0
    for offers in strategy.submitted.values():
        for p in {t.product for t in offers.ancillary}:
            mine = [t for t in offers.ancillary if t.product is p]
            assert len(mine) <= market_cfg.n_tranches
            assert [t.price for t in mine] == sorted(t.price for t in mine)
            assert sum(t.quantity_mw for t in mine) <= market_cfg.ancillary.max_award_mw[p] + 1e-6
            priced += sum(t.price > 0 for t in mine)
    assert priced > 0  # not everything at 0: the curves carry prices


def test_priced_reserve_offers_need_market_feasibility(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.config import FeasibilityMode, ReserveOffers, ReservePlanner

    base_cfg = markov_bidding()
    bidding = base_cfg.model_copy(
        update={
            "reserve_planner": ReservePlanner.SDP,
            "reserve_offers": ReserveOffers.CURVE,
            "feasibility": base_cfg.feasibility.model_copy(
                update={"mode": FeasibilityMode.WORST_CASE}
            ),
        }
    )
    prices = make_prices(daily_shape(2, seed=4))
    qf = quantiles_for(prices, 40.0, market_cfg.gate_closure_intervals)
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    with pytest.raises(ValueError, match="feasibility"):
        WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=True)


def test_stochastic_reserve_plan_runs_without_breaches(battery_cfg, market_cfg) -> None:
    """Reserve planner ``stochastic_milp``: reserves planned against sampled Markov price paths;
    the awards are sustainable (the simulator checks) and earn ancillary revenue."""
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.config import Product, ReservePlanner
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(DAYS + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(p, market_cfg), rng.uniform(5, 30, base.height))
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    qf = quantiles_for(
        window.select(Col.INTERVAL_START, Col.USEP), 40.0, market_cfg.gate_closure_intervals
    )
    bidding = markov_bidding().model_copy(
        update={"reserve_planner": ReservePlanner.STOCHASTIC_MILP, "stochastic_paths": 5}
    )
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=True)
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    assert dispatch.total().ancillary_sgd > 0
    issue = next(iter(strategy.value_functions))
    paths = strategy._price_paths(issue)
    assert paths.shape == (5, 48)
    assert set(np.unique(paths[:, 0])) <= set(strategy.stage_prices(issue)[0])


def test_reserve_reward_pulls_the_soc_to_the_middle(battery_cfg, market_cfg) -> None:
    """Over one stage with no energy spread, regulation (up and down for 30 min, RC386) is worth
    most at a mid SOC; raise reserves only need stored energy, so none at an empty battery."""
    from battery_optimisation.optim.reserves import pair_corners
    from battery_optimisation.optim.sdp import solve_sdp

    bidding = markov_bidding()
    dt = market_cfg.interval_h
    corners_ = pair_corners(market_cfg, battery_cfg, bidding.soc_levels, dt, True)
    reg_only = corners_.rewards(np.array([[20.0 * dt, 0.0, 0.0]]))
    reserves = corners_.rewards(np.array([[0.0, 10.0 * dt, 20.0 * dt]]))

    def values(r):
        return solve_sdp(
            np.zeros((1, 1)),
            np.ones(1),
            Battery(battery_cfg),
            dt,
            bidding,
            soc_target=0.0,
            transitions=np.ones((1, 1)),
            reserve_reward=r,
        ).values[0]

    gain = values(reg_only) - values(None)
    mid = len(gain) // 2
    assert gain[mid] > gain[0] + 1.0 and gain[mid] > gain[-1] + 1.0
    gain = values(reserves) - values(None)
    assert gain[0] == pytest.approx(0.0, abs=1e-6) and gain[mid] > 1.0


@pytest.mark.parametrize("ancillary", [False, True])
def test_market_feasibility_leaves_the_soc_to_the_engine(battery_cfg, market_cfg, ancillary):
    """Feasibility ``market``: offers capped from the *expected* SOC only (what can reasonably
    be delivered, Ch. 6 5.2B.5-6), not the worst case: more is offered; the engine cuts the
    schedule to the real SoC (RC386), so nothing breaches or is penalised."""
    from battery_optimisation.config import FeasibilityMode, Product
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(1)
    base = make_prices(daily_shape(DAYS + 1, seed=5))
    prices = base.with_columns(
        pl.Series(price_column(p, market_cfg), rng.uniform(5, 30, base.height))
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    )
    history, window = prices.slice(0, 48), prices.slice(48)
    if not ancillary:
        window = window.select(Col.INTERVAL_START, Col.USEP)
    scen = scenario_frame(
        window.select(Col.INTERVAL_START, Col.USEP),
        n=10,
        noise=60.0,
        gap=market_cfg.gate_closure_intervals,
        seed=3,
    )
    worst_cfg = classic_bidding()
    market_mode = worst_cfg.model_copy(
        update={
            "feasibility": worst_cfg.feasibility.model_copy(update={"mode": FeasibilityMode.MARKET})
        }
    )
    offered = {}
    b, dt = Battery(battery_cfg), market_cfg.interval_h
    for name, cfg in (("worst", worst_cfg), ("market", market_mode)):
        strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, cfg, ancillary=ancillary)
        dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(
            strategy
        )  # RAISE: any breach fails the test
        offered[name] = sum(t.quantity_mw for o in strategy.submitted.values() for t in o.tranches)
        assert dispatch.total().shortfall_mwh == 0.0
        if name == "market":
            for at, offers in strategy.submitted.items():
                soc = strategy.reference_soc[at]  # expected SOC at the start of the interval
                sold = sum(t.quantity_mw for t in offers.on(Side.DISCHARGE))
                bought = sum(t.quantity_mw for t in offers.on(Side.CHARGE))
                assert sold * b.discharge_loss(dt) <= soc - battery_cfg.soc_min + 0.1
                assert bought * b.charge_gain(dt) <= battery_cfg.soc_max - soc + 0.1
    assert offered["market"] > offered["worst"]


def test_water_value_keeps_a_decision_trace(battery_cfg, market_cfg) -> None:
    """Per offered interval: issue, stage, decision and expected SOC, price states, planned
    reserves (decisions.parquet) to explain a settlement."""
    from battery_optimisation.bidding.stages import LatticeStages

    prices = make_prices(daily_shape(2, seed=4))
    qf = quantiles_for(prices, 40.0, market_cfg.gate_closure_intervals)
    bidding = markov_bidding()
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    strategy = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding)
    run(strategy, prices, battery_cfg, market_cfg)
    trace = strategy.decisions()
    assert trace.height == len(strategy.submitted)
    assert {Col.ISSUED_AT, Col.STAGE, Col.DECISION_SOC_MWH, Col.EXPECTED_SOC_MWH} <= set(
        trace.columns
    )
    assert trace[Col.STATE_PRICES].list.len().unique().to_list() == [bidding.markov.states]


def _reserve_window(market_cfg, days: int = 2):
    from battery_optimisation.config import Product
    from battery_optimisation.sim.clearing import price_column

    rng = np.random.default_rng(0)
    base = make_prices(daily_shape(days + 1, seed=4))
    prices = base.with_columns(
        pl.Series(price_column(p, market_cfg), rng.uniform(lo, hi, base.height))
        for p, lo, hi in (
            (Product.REGULATION, 15, 35),
            (Product.PRIMARY_RESERVE, 5, 15),
            (Product.CONTINGENCY_RESERVE, 10, 30),
        )
    )
    return prices.slice(0, 48), prices.slice(48)


@pytest.mark.parametrize("record", [True, False])
def test_strategies_record_the_plan_of_every_decision(battery_cfg, market_cfg, record) -> None:
    """What each strategy planned at each decision (troubleshooting): the self-schedule's MILP
    plan, the water value's SDP policy rolled forward (expected SoC with a 10-90% band and the
    reserves it would hold); ``record_plans`` off records nothing."""
    from battery_optimisation.bidding.stages import LatticeStages
    from battery_optimisation.bidding.strategy import PLAN_COLUMNS
    from battery_optimisation.config import ReservePlanner

    history, window = _reserve_window(market_cfg)
    qf = quantiles_for(
        window.select(Col.INTERVAL_START, Col.USEP), 40.0, market_cfg.gate_closure_intervals
    )
    bidding = markov_bidding().model_copy(
        update={"reserve_planner": ReservePlanner.SDP, "record_plans": record}
    )
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    wv = WaterValueStrategy(stages, battery_cfg, market_cfg, bidding, ancillary=True)
    ss = ScheduleStrategy(
        PerfectForecaster(window), battery_cfg, market_cfg, Horizon.ROLLING, record_plans=record
    )
    gate = timedelta(hours=market_cfg.interval_h * market_cfg.gate_closure_intervals)
    for strategy in (wv, ss):
        Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
        plans = strategy.plans()
        assert plans.columns == list(PLAN_COLUMNS)
        if not record:
            assert plans.is_empty()
            continue
        firsts = plans.group_by(Col.TARGET_START).agg(pl.all().first()).sort(Col.TARGET_START)
        assert firsts.height == window.height
        assert (firsts[Col.INTERVAL_START] == firsts[Col.TARGET_START]).all()
        assert (firsts[Col.DECIDED_AT] == firsts[Col.TARGET_START] - gate).all()
        assert (
            plans[Col.SOC_MWH]
            .is_between(battery_cfg.soc_min - 1e-6, battery_cfg.soc_max + 1e-6)
            .all()
        )
        assert (plans[Col.NET_MW].abs() <= battery_cfg.p_max_mw + 1e-6).all()
    wv_plans = wv.plans() if record else None
    if record:
        # a band (the mean of a skewed distribution may lie outside it)
        assert (wv_plans[Col.SOC_P10_MWH] <= wv_plans[Col.SOC_P90_MWH]).all()
        assert (wv_plans[Col.SOC_P90_MWH] > wv_plans[Col.SOC_P10_MWH]).any()
        assert wv_plans[Col.CON_MW].sum() > 0  # reserves held along the plan
        # the self-schedule's plan for the offered interval is what it offered
        first = ss.plans().group_by(Col.TARGET_START).agg(pl.col(Col.NET_MW).first())
        offered = {at: d - c for at, (c, d) in ss.committed.items()}
        for at, net in first.iter_rows():
            assert net == pytest.approx(offered[at], abs=0.1)


def test_water_value_with_reserve_price_states(battery_cfg, market_cfg) -> None:
    """``reserve_prices: markov``: the SDP's state is (energy, reserve factor), the reserves
    are chosen in expectation over both, nothing breaches, plans roll over the joint states."""
    from synthetic_forecast import normal_quantiles, quantile_frame

    from battery_optimisation.bidding.stages import LatticeStages, ReserveLattices
    from battery_optimisation.config import (
        AncillaryPrice,
        Product,
        ReservePlanner,
        ReservePrices,
        load_forecast,
    )
    from battery_optimisation.sim.clearing import price_column

    history, window = _reserve_window(market_cfg)
    gap = market_cfg.gate_closure_intervals
    qf = quantiles_for(window.select(Col.INTERVAL_START, Col.USEP), 40.0, gap)
    levels = load_forecast().quantile_levels
    frames = []
    for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE):
        col = price_column(p, market_cfg)
        for _, part in window.group_by(pl.col(Col.INTERVAL_START).dt.date(), maintain_order=True):
            issued = part[Col.INTERVAL_START][0] - timedelta(minutes=30 * gap)
            q = normal_quantiles(part[col].to_numpy(), 6.0, levels)
            frames.append(
                quantile_frame(issued, q, levels, gap=gap).with_columns(
                    pl.lit(str(col)).alias(Col.TARGET)
                )
            )
    anc = pl.concat(frames)
    bidding = markov_bidding().model_copy(
        update={
            "reserve_planner": ReservePlanner.SDP,
            "reserve_prices": ReservePrices.MARKOV,
            "ancillary_price": AncillaryPrice.MEAN,
        }
    )
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
    lattices = ReserveLattices(
        anc,
        market_cfg,
        bidding.reserve_markov,
        (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE),
    )
    strategy = WaterValueStrategy(
        stages, battery_cfg, market_cfg, bidding, ancillary=True, reserve_lattices=lattices
    )
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    assert dispatch.total().ancillary_sgd > 0
    assert strategy.reserve_lattice_fallbacks == 0
    vf = strategy.value_functions[max(strategy.value_functions)]
    k, r = bidding.markov.states, bidding.reserve_markov.states
    assert vf.joint_occupancy.shape[1] == k * r
    plans = strategy.plans()
    assert plans[Col.CON_MW].sum() > 0
    assert (plans[Col.SOC_P10_MWH] <= plans[Col.SOC_P90_MWH]).all()
    # older issues dropped their policy (memory), the latest keep it
    kept = [i for i, v in strategy.value_functions.items() if v.policy is not None]
    assert 0 < len(kept) <= WaterValueStrategy.POLICIES_KEPT
