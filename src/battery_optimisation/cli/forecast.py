"""Probabilistic forecasting commands.

Examples::

    uv run forecast run                                    # last 12 months, config defaults
    uv run forecast run --weather-model ecmwf_ifs --start 2026-02-01
    uv run forecast run --covariates calendar --covariates regime   # ablation
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from battery_optimisation.columns import Col
from battery_optimisation.config import (
    Covariate,
    ForecastConfig,
    load_battery,
    load_forecast,
    load_market,
)
from battery_optimisation.data.nems import NemsSeries, load_nems
from battery_optimisation.forecast import walkforward
from battery_optimisation.forecast.chronos import ChronosModel
from battery_optimisation.forecast.report import (
    DEFAULT_RUNS_DIR,
    ForecastRunFiles,
    evaluate,
    list_forecast_runs,
    load_forecast_run,
    save_ancillary_quantiles,
    save_forecast_run,
)
from battery_optimisation.logs import configure_logging
from battery_optimisation.sim import backtest
from battery_optimisation.sim.clearing import price_column

app = typer.Typer(
    help="Chronos-2 forecasts, scenarios, evaluation.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
DateOpt = typer.Option(formats=["%Y-%m-%d"], help="YYYY-MM-DD")


@app.callback()
def _root() -> None:
    """Chronos-2 forecasts, scenarios, evaluation."""


@app.command()
def run(
    months: Annotated[int, typer.Option(help="Test window length, ending at --end.")] = 12,
    start: Annotated[datetime | None, DateOpt] = None,
    end: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="Default: last day of USEP history."),
    ] = None,
    weather_model: Annotated[
        str | None, typer.Option(help="Weather provider (must be in archive_models).")
    ] = None,
    covariates: Annotated[
        list[Covariate] | None, typer.Option(help="Repeatable; default: config.")
    ] = None,
    runs_dir: Annotated[Path, typer.Option()] = backtest.DEFAULT_RUNS_DIR,
    reuse_quantiles: Annotated[
        bool, typer.Option(help="Reuse the run folder's saved quantiles (skip Chronos-2).")
    ] = False,
    issue_every: Annotated[
        int | None, typer.Option(help="Issue cadence in intervals (48 = daily). Default: config.")
    ] = None,
) -> None:
    """Walk-forward Chronos-2 quantiles -> monthly copula scenarios -> stats and capture rates."""
    configure_logging()
    data = load_forecast().model_dump()  # overrides are re-validated like the YAML
    if weather_model:
        data["weather"]["model"] = weather_model
    if covariates:
        data["covariates"] = covariates
    if issue_every is not None:
        data["issue_every_intervals"] = issue_every
    cfg = ForecastConfig.model_validate(data)
    battery, market = load_battery(), load_market()

    test_start, test_end = backtest.last_months_window(months, end.date() if end else None)
    if start:
        test_start = start.date()
    warmup = test_start - timedelta(days=cfg.scenarios.calibration_days)
    covs = "-".join(str(c) for c in cfg.covariates)
    cadence = "" if cfg.issue_every_intervals >= 48 else f"_every{cfg.issue_every_intervals}"
    name = f"forecast_chronos2_{covs}_{cfg.weather.model}{cadence}_{test_start}_{test_end}"
    run_dir = Path(runs_dir) / name

    features, trajectories, weather_report = walkforward.build_features(warmup, test_end, cfg)
    issues = walkforward.issue_times(warmup, test_end, cfg)
    saved = run_dir / ForecastRunFiles.QUANTILES
    t0 = datetime.now()
    if reuse_quantiles and saved.exists():
        qf = pl.read_parquet(saved)
    else:
        qf = ChronosModel(cfg).forecast_quantiles(features, issues, trajectories)
        run_dir.mkdir(parents=True, exist_ok=True)
        qf.write_parquet(saved)  # kept if the evaluation fails (--reuse-quantiles)
    chronos_seconds = (datetime.now() - t0).total_seconds()

    realised = load_nems(warmup - timedelta(days=1), test_end, series=[NemsSeries.USEP]).select(
        Col.INTERVAL_START, Col.USEP
    )
    ev = evaluate(qf, realised, test_start, test_end, cfg, battery, market)
    report = ev.report.model_copy(
        update={"chronos_seconds": chronos_seconds, "weather": weather_report}
    )
    save_forecast_run(ev, qf, run_dir, report)

    _print(report)
    typer.echo(f"\nSaved to {run_dir}")


@app.command()
def ancillary(
    forecast_run: Annotated[
        Path | None, typer.Option(help="Forecast run folder; default: the latest.")
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where forecast runs live.")] = DEFAULT_RUNS_DIR,
) -> None:
    """Add Chronos-2 forecasts of the ancillary prices to a forecast run (same issues and
    covariates as its USEP forecasts); ``bidding run --ancillary`` then uses them."""
    configure_logging()
    run_dir = forecast_run or list_forecast_runs(runs_dir)[0]
    fr = load_forecast_run(run_dir)
    cfg, market = fr.report.config, load_market()
    start, end = fr.report.test_start, fr.report.test_end
    features, trajectories, _ = walkforward.build_features(start, end, cfg)
    cols = [price_column(p, market) for p in market.ancillary_products]
    first = features[Col.INTERVAL_START].min().date()
    prices = backtest.load_prices(first, end + timedelta(days=2), market, ancillary=True)
    features = features.join(
        prices.select(Col.INTERVAL_START, *cols), on=Col.INTERVAL_START, how="left"
    ).with_columns(pl.col(c).fill_null(strategy="forward") for c in cols)
    issues = walkforward.issue_times(start, end, cfg)
    aq = walkforward.ancillary_quantiles(ChronosModel(cfg), features, issues, trajectories, market)
    path = save_ancillary_quantiles(aq, run_dir)
    median = aq.filter((pl.col(Col.QUANTILE) - 0.5).abs() < 1e-9).join(
        prices, on=Col.INTERVAL_START
    )
    for c in cols:
        part = median.filter(pl.col(Col.TARGET) == str(c))
        mae = (part[Col.USEP_FORECAST] - part[c]).abs().mean()
        typer.echo(f"{c}: MAE(median) {mae:.2f} SGD/MWh over {part.height} forecasts")
    typer.echo(f"Saved to {path}")


def _print(report) -> None:
    def frame(rows, cols):
        return pl.DataFrame([r.model_dump() for r in rows]).select(
            pl.col(c).round(3)
            if c
            not in ("period_start", "strategy", "horizon", "run", "intervals", "calibration_issues")
            else pl.col(c)
            for c in cols
        )

    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_width_chars=160):
        o, c = report.overall, report.config
        covs = [str(x) for x in c.covariates]
        typer.echo(
            f"\nChronos-2 {report.test_start} -> {report.test_end}: {report.issues} issues "
            f"(every {c.issue_every_intervals} intervals), "
            f"covariates {covs}, weather {c.weather.model}"
        )
        typer.echo(
            f"CRPS {o.crps:.1f}  MAE(median) {o.mae_median:.1f}  MASE {o.mase:.2f}  "
            f"coverage 10-90 {o.coverage_10_90:.1%}  "
            f"spikes (>= {report.spike_threshold:.0f}): hit {o.spike_hit_rate:.1%}, "
            f"false alarm {o.spike_false_alarm:.1%}\n"
        )
        typer.echo(
            frame(
                report.monthly,
                [
                    "period_start",
                    "intervals",
                    "crps",
                    "mae_median",
                    "mase",
                    "coverage_10_90",
                    "spike_hit_rate",
                    "spike_false_alarm",
                ],
            )
        )
        typer.echo(
            frame(
                report.scenarios,
                ["period_start", "calibration_issues", "copula_lag1", "realised_lag1"],
            )
        )
        typer.echo(
            frame(
                report.capture,
                [
                    "strategy",
                    "horizon",
                    "profit_sgd",
                    "cycles_per_day",
                    "capture_vs_perfect_full",
                    "capture_vs_perfect_day",
                ],
            )
        )


if __name__ == "__main__":
    app()
