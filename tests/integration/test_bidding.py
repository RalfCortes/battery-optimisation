"""Water-value bidding through the simulator with gate closure."""

from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import daily_shape, make_prices

from battery_optimisation.battery import Battery
from battery_optimisation.bidding.strategy import Horizon, ScheduleStrategy, WaterValueStrategy
from battery_optimisation.columns import Col
from battery_optimisation.config import load_bidding
from battery_optimisation.forecast.perfect import PerfectForecaster
from battery_optimisation.sim.clearing import Market
from battery_optimisation.sim.simulator import Simulator

DAYS = 3


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
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
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
        WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding()),
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
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
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

    independent = load_bidding().model_copy(update={"price_model": PriceModel.INDEPENDENT})
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
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
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
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
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
    strategy = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding())
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

    prices = make_prices(daily_shape(DAYS, seed=4))
    scen = scenario_frame(prices, n=20, noise=80.0, gap=market_cfg.gate_closure_intervals, seed=1)
    worst_cfg = load_bidding()
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
        total.shortfall_mwh * market_cfg.shortfall_penalty_sgd_per_mwh
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

    energy_only = run_hist(WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding()))
    both = WaterValueStrategy(scen, battery_cfg, market_cfg, load_bidding(), ancillary=True)
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
        scen, battery_cfg, market_cfg, load_bidding(), ancillary=True, ancillary_forecaster=spy
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
        scen, battery_cfg, market_cfg, load_bidding(), ancillary=True, firm_reserves=firm
    )
    dispatch = Simulator(Market(market_cfg, window, history), Battery(battery_cfg)).run(strategy)
    planned = sum(sum(a.values()) for a in strategy.planned_awards.values())
    awarded = dispatch.frame.select(sum(pl.col(c) for c in AWARD_COLUMN.values())).to_series().sum()
    assert awarded >= (0.97 if firm else 0.95) * planned
    raise_cols = [
        AWARD_COLUMN[p]
        for p in (Product.REGULATION, Product.PRIMARY_RESERVE, Product.CONTINGENCY_RESERVE)
    ]
    most_raise = dispatch.frame.select(sum(pl.col(c) for c in raise_cols)).to_series().max()
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

    bidding = load_bidding()
    markov = bidding.markov.model_copy(update={"rho": rho})
    return bidding.model_copy(update={"price_model": PriceModel.MARKOV, "markov": markov})


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


def test_markov_with_exact_quantiles_approaches_the_benchmark(battery_cfg, market_cfg) -> None:
    from battery_optimisation.bidding.stages import LatticeStages

    prices = make_prices(daily_shape(DAYS, seed=6))
    qf = quantiles_for(prices, 1e-3, market_cfg.gate_closure_intervals)
    bidding = markov_bidding()
    stages = LatticeStages(qf, bidding.markov.states, lambda _: 0.8)
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


@pytest.mark.parametrize("rho", [0.8, None])
def test_bidding_report_with_markov_prices(battery_cfg, market_cfg, rho) -> None:
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
        markov_bidding(rho),
        copula_lag1={date(2026, 1, 1): 0.7},
    )
    assert ev.report.sdp_days == DAYS
    assert ev.water_value.stage_prices(next(iter(ev.water_value.value_functions))).shape[1] == 11


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
