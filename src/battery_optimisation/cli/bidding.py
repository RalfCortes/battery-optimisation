"""Bidding commands: water-value offer curves vs the self-schedule baseline.

Examples::

    uv run bidding run                         # on the latest forecast run (runs/forecast_*)
    uv run bidding run --forecast-run runs/forecast_chronos2_...
    uv run limited --mem-gb 100 --cpus 8 -- uv run bidding run   # under resource limits
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from battery_optimisation.bidding.report import (
    ComparisonInput,
    comparison_table,
    evaluate_bidding,
    save_bidding,
    to_markdown,
)
from battery_optimisation.bidding.sweep import sweep, sweep_breakdown
from battery_optimisation.columns import Col
from battery_optimisation.config import (
    BiddingConfig,
    FeasibilityMode,
    load_battery,
    load_bidding,
    load_market,
)
from battery_optimisation.forecast.report import (
    list_forecast_runs,
    load_forecast_run,
    schedule_horizon,
)
from battery_optimisation.logs import configure_logging
from battery_optimisation.sim.backtest import StrategyName, load_prices

app = typer.Typer(
    help="Water-value bidding backtests.", no_args_is_help=True, pretty_exceptions_enable=False
)


@app.callback()
def _root() -> None:
    """Water-value bidding backtests."""


def bidding_tag(bidding: BiddingConfig, ancillary: bool) -> str:
    """Output folder suffix: feasibility rule, then ``_as`` (+ ``_hybrid``) with ancillary."""
    f = bidding.feasibility
    tag = (
        "worst_case"
        if f.mode is FeasibilityMode.WORST_CASE
        else f"prob_q{f.soc_quantile:g}_m{f.margin_pct:g}"
    )
    if ancillary:
        tag += "_as" + ("" if bidding.firm_reserves else "_hybrid")
    return tag


@app.command()
def run(
    forecast_run: Annotated[
        Path | None, typer.Option(help="Forecast run folder; default: the latest.")
    ] = None,
    feasibility: Annotated[FeasibilityMode | None, typer.Option(help="Default: config.")] = None,
    margin_pct: Annotated[
        float | None, typer.Option(help="Probabilistic margin, % usable.")
    ] = None,
    soc_quantile: Annotated[float | None, typer.Option(help="Probabilistic SOC quantile.")] = None,
    ancillary: Annotated[
        bool, typer.Option(help="Regulation and reserves too (plan + energy curves).")
    ] = False,
    firm_reserves: Annotated[
        bool | None,
        typer.Option(help="Ancillary: reserves beside any energy flow. Default: config."),
    ] = None,
) -> None:
    """Water-value offers vs self-schedule, price sweep check, comparison of all strategies."""
    configure_logging()
    data = load_bidding().model_dump()  # overrides re-validated like the YAML
    for key, value in (
        ("mode", feasibility),
        ("margin_pct", margin_pct),
        ("soc_quantile", soc_quantile),
    ):
        if value is not None:
            data["feasibility"][key] = value
    if firm_reserves is not None:
        data["firm_reserves"] = firm_reserves
    bidding = BiddingConfig.model_validate(data)
    fr = load_forecast_run(forecast_run or list_forecast_runs()[0])
    market = load_market()
    start = fr.report.test_start
    prices = load_prices(start - timedelta(days=3), fr.report.test_end, market, ancillary)
    realised = prices.filter(pl.col(Col.INTERVAL_START).dt.date() >= start)
    history = prices.filter(pl.col(Col.INTERVAL_START).dt.date() < start)
    horizon = schedule_horizon(fr.report.config, market)
    ev = evaluate_bidding(
        fr.scenarios,
        fr.quantiles,
        realised,
        load_battery(),
        market,
        bidding,
        horizon,
        history=history,
        ancillary=ancillary,
        ancillary_quantiles=fr.ancillary_quantiles,
    )
    if ancillary:
        source = "Chronos-2" if fr.ancillary_quantiles is not None else "persistence"
        typer.echo(f"Ancillary prices forecast by {source}")
    grid = np.asarray(bidding.sweep.prices)
    stats, per_interval = sweep(ev.water_value, grid)
    steps = sweep_breakdown(ev.water_value, grid)
    ev = type(ev)(
        report=ev.report.model_copy(update={"sweep": stats, "sweep_steps": steps}),
        runs=ev.runs,
        water_value=ev.water_value,
    )
    tag = bidding_tag(bidding, ancillary)
    out = save_bidding(ev, fr.path / f"bidding_{tag}")
    per_interval.write_parquet(out / "sweep.parquet")

    # every strategy over the same window: forecast-run backtests + the bidding run
    # (with ancillary services: this run's own benchmark and strategies only)
    rows = (
        []
        if ancillary
        else [
            ComparisonInput(
                strategy=c.strategy,
                horizon=c.horizon,
                profit_sgd=c.profit_sgd,
                cycles_per_day=c.cycles_per_day,
            )
            for c in fr.report.capture
        ]
    )
    if ancillary:
        rows += [
            ComparisonInput(
                strategy=name,
                horizon=summary.horizon,
                profit_sgd=summary.total.profit_sgd,
                cycles_per_day=summary.cycles_per_day,
            )
            for name, (summary, _) in ev.runs.items()
            if name is not StrategyName.WATER_VALUE
        ]
    wv = ev.runs[StrategyName.WATER_VALUE][0]
    rows.append(
        ComparisonInput(
            strategy=StrategyName.WATER_VALUE,
            horizon=horizon,
            profit_sgd=wv.total.profit_sgd,
            cycles_per_day=wv.cycles_per_day,
        )
    )
    table = comparison_table(rows, baseline=StrategyName.CHRONOS_MEDIAN)
    markdown = to_markdown(table)
    (out / "comparison.md").write_text(markdown)
    table.write_csv(out / "comparison.csv")

    r = ev.report
    typer.echo(f"\nBidding backtest {r.start} -> {r.end} (forecast run {fr.path.name})")
    typer.echo(
        f"gate closure {r.gate_closure_intervals} intervals, K = {r.n_tranches} tranches/side, "
        f"{r.sdp_days} daily SDPs in {r.sdp_solve_time_s:.0f} s\n"
    )
    typer.echo(markdown)
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, thousands_separator=True):
        typer.echo(
            pl.DataFrame([m.model_dump() for m in r.monthly]).select(
                "period_start",
                *[
                    pl.col(c).round(0).cast(pl.Int64)
                    for c in ("self_schedule_sgd", "water_value_sgd", "uplift_sgd")
                ],
            )
        )
    typer.echo(
        f"\nBidding uplift {r.uplift_sgd:,.0f} SGD ({r.uplift_pct:+.1%} vs self-schedule)"
        f"\nFeasibility {r.feasibility}: shortfall {r.shortfall_mwh:,.1f} MWh, "
        f"penalty {r.penalty_sgd:,.0f} SGD (in profit)"
        f"\nOffers: {r.mean_tranches_per_interval:.1f} tranches/interval, "
        f"{r.price_taking_share:.0%} of offered MW price-taking"
        f"\nPrice sweep ({stats.prices} prices {stats.price_min:g}..{stats.price_max:g}, "
        f"{stats.intervals} intervals): mean |offered - optimal| {stats.mean_abs_gap_mw:.1f} MW "
        f"(within scenario band {stats.mean_abs_gap_in_band_mw:.1f} MW), "
        f"{stats.share_matching:.0%} within 1 MW, max {stats.max_gap_mw:.0f} MW"
    )
    typer.echo("\nGap to the optimal response after each step (mean MW, all prices / in band):")
    for s in steps:
        typer.echo(f"  {s.step:<13} {s.mean_abs_gap_mw:6.1f} / {s.mean_abs_gap_in_band_mw:6.1f}")
    typer.echo(f"\nSaved to {out}")


if __name__ == "__main__":
    app()
