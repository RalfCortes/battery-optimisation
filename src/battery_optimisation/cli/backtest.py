"""Backtest commands.

Examples::

    uv run backtest perfect-foresight                     # last 12 months of USEP history
    uv run backtest perfect-foresight --months 3 --horizon day
    uv run backtest perfect-foresight --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from battery_optimisation.bidding.strategy import Horizon
from battery_optimisation.columns import Col
from battery_optimisation.config import load_battery, load_market
from battery_optimisation.logs import configure_logging
from battery_optimisation.sim import backtest

app = typer.Typer(help="Backtests through the clearing simulator.", no_args_is_help=True)

DateOpt = typer.Option(formats=["%Y-%m-%d"], help="YYYY-MM-DD")


@app.callback()
def _root() -> None:
    """Backtests through the clearing simulator."""


@app.command("perfect-foresight")
def perfect_foresight(
    months: Annotated[int, typer.Option(help="Window length, ending at --end.")] = 12,
    start: Annotated[datetime | None, DateOpt] = None,
    end: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Default: last day of USEP history."),
    ] = None,
    horizon: Annotated[
        Horizon, typer.Option(help="full = one MILP (true optimum); day = daily MILPs.")
    ] = Horizon.FULL,
    runs_dir: Annotated[Path, typer.Option()] = backtest.DEFAULT_RUNS_DIR,
    ancillary: Annotated[
        bool, typer.Option(help="Co-optimise regulation and reserves with energy.")
    ] = False,
) -> None:
    """Perfect-foresight benchmark on realised USEP; saves summary.json + dispatch.parquet."""
    configure_logging()
    battery, market = load_battery(), load_market()
    window_start, window_end = backtest.last_months_window(months, end.date() if end else None)
    if start:
        window_start = start.date()

    summary, dispatch = backtest.run_perfect_foresight(
        window_start, window_end, battery, market, horizon, ancillary
    )
    run_dir = backtest.save_run(summary, dispatch, runs_dir)

    monthly = pl.DataFrame([m.model_dump() for m in summary.monthly]).select(
        Col.PERIOD_START,
        pl.col(Col.PROFIT_SGD, Col.REVENUE_SGD, Col.DEGRADATION_SGD).round(0).cast(pl.Int64),
        pl.col(Col.CYCLES).round(1),
        pl.col(Col.MEAN_USEP).round(1),
    )
    daily = pl.DataFrame([d.model_dump() for d in summary.daily])
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, thousands_separator=True):
        typer.echo(f"\nPerfect foresight ({summary.horizon}) {summary.start} -> {summary.end}")
        rte = battery.round_trip_efficiency
        typer.echo(f"{battery.p_max_mw:g} MW / {battery.e_max_mwh:g} MWh, RTE {rte:.0%}\n")
        typer.echo(monthly)
        typer.echo(
            daily.select(
                pl.col(Col.PROFIT_SGD).quantile(q).round(0).cast(pl.Int64).alias(name)
                for q, name in [
                    (0.0, "min"),
                    (0.1, "p10"),
                    (0.5, "median"),
                    (0.9, "p90"),
                    (1.0, "max"),
                ]
            )
            .with_columns(pl.lit("daily profit SGD").alias(""))
            .select("", pl.exclude(""))
        )
    t = summary.total
    typer.echo(
        f"\nTotal profit   {t.profit_sgd:>14,.0f} SGD  ({summary.profit_per_day_sgd:,.0f} /day)"
        f"\nRevenue        {t.revenue_sgd:>14,.0f} SGD"
        f"\nAncillary      {t.ancillary_sgd:>14,.0f} SGD"
        f"\nDegradation    {t.degradation_sgd:>14,.0f} SGD"
        f"\nCycles         {t.cycles:>14,.1f}      ({summary.cycles_per_day:.2f} /day)"
        f"\nMILP solves    {summary.plans:>14,d}      ({summary.solve_time_s:.1f} s)"
        f"\nSimulation     {summary.simulate_time_s:>14.1f} s"
        f"\n\nSaved to {run_dir}"
    )


if __name__ == "__main__":
    app()
