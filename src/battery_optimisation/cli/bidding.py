"""Bidding commands: water-value offer curves vs the self-schedule baseline.

Examples::

    uv run bidding run                         # on the latest forecast run (runs/forecast_*)
    uv run bidding run --forecast-run runs/forecast_chronos2_...
    uv run limited --mem-gb 100 --cpus 8 -- uv run bidding run   # under resource limits
"""

from __future__ import annotations

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
from battery_optimisation.bidding.strategy import Horizon
from battery_optimisation.bidding.sweep import sweep, sweep_breakdown
from battery_optimisation.columns import Col
from battery_optimisation.config import load_battery, load_bidding, load_market
from battery_optimisation.data.nems import NemsSeries, load_nems
from battery_optimisation.forecast.report import list_forecast_runs, load_forecast_run
from battery_optimisation.logs import configure_logging
from battery_optimisation.sim.backtest import StrategyName

app = typer.Typer(
    help="Water-value bidding backtests.", no_args_is_help=True, pretty_exceptions_enable=False
)


@app.callback()
def _root() -> None:
    """Water-value bidding backtests."""


@app.command()
def run(
    forecast_run: Annotated[
        Path | None, typer.Option(help="Forecast run folder; default: the latest.")
    ] = None,
) -> None:
    """Water-value offers vs self-schedule, price sweep check, comparison of all strategies."""
    configure_logging()
    bidding = load_bidding()
    fr = load_forecast_run(forecast_run or list_forecast_runs()[0])
    realised = load_nems(fr.report.test_start, fr.report.test_end, series=[NemsSeries.USEP]).select(
        Col.INTERVAL_START, Col.USEP
    )
    ev = evaluate_bidding(
        fr.scenarios, fr.quantiles, realised, load_battery(), load_market(), bidding
    )
    grid = np.asarray(bidding.sweep.prices)
    stats, per_interval = sweep(ev.water_value, grid)
    steps = sweep_breakdown(ev.water_value, grid)
    ev = type(ev)(
        report=ev.report.model_copy(update={"sweep": stats, "sweep_steps": steps}),
        runs=ev.runs,
        water_value=ev.water_value,
    )
    out = save_bidding(ev, fr.path / "bidding")
    per_interval.write_parquet(out / "sweep.parquet")

    # every strategy over the same window: forecast-run backtests + the bidding run
    rows = [
        ComparisonInput(
            strategy=c.strategy,
            horizon=c.horizon,
            profit_sgd=c.profit_sgd,
            cycles_per_day=c.cycles_per_day,
        )
        for c in fr.report.capture
    ]
    wv = ev.runs[StrategyName.WATER_VALUE][0]
    rows.append(
        ComparisonInput(
            strategy=StrategyName.WATER_VALUE,
            horizon=Horizon.DAY,
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
