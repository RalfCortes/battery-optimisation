"""Weather forecast commands (settings: ``weather`` section of config/forecast.yaml).

Examples::
    uv run weather download          # cron: last 2 days, all runs, all providers
    uv run weather download --start 2025-06-01 --needed-only --model ecmwf_ifs   # backfill
    uv run weather download --start 2025-06-01 --needed-only --model ecmwf_ifs   # backfill
    uv run weather process --start 2025-10-01 --end 2025-10-31   # features + transform report
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from battery_optimisation.config import load_forecast
from battery_optimisation.data import weather
from battery_optimisation.logs import configure_logging

app = typer.Typer(
    help="As-of weather forecasts (Open-Meteo single runs).",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
DateOpt = typer.Option(formats=["%Y-%m-%d"], help="YYYY-MM-DD")


@app.callback()
def _root() -> None:
    """As-of weather forecasts (Open-Meteo single runs)."""


@app.command()
def download(
    start: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Default: 2 days ago.")
    ] = None,
    end: Annotated[datetime | None, DateOpt] = None,
    model: Annotated[
        list[str] | None, typer.Option(help="Repeatable; default: all archive_models.")
    ] = None,
    needed_only: Annotated[
        bool, typer.Option(help="Only the runs the daily forecast features use.")
    ] = False,
    raw_dir: Annotated[Path, typer.Option()] = weather.DEFAULT_RAW_DIR,
    overwrite: Annotated[bool, typer.Option(help="Re-fetch runs already on disk.")] = False,
    json_logs: Annotated[bool, typer.Option(help="One JSON object per log line.")] = False,
) -> None:
    """Download model runs (one file per run); resumable."""
    configure_logging(json=json_logs)
    fcfg = load_forecast()
    cfg = fcfg.weather
    first = start.date() if start else date.today() - timedelta(days=2)
    last = end.date() if end else date.today()
    runs = (
        weather.needed_runs(first, last, fcfg)
        if needed_only
        else weather.all_runs(first, last, cfg)
    )
    failed = []
    for name in model or cfg.archive_models:
        try:
            written = weather.download_runs(runs, cfg, name, raw_dir, overwrite)
            typer.echo(f"{name}: {len(written)} new run(s) of {len(runs)} requested")
        except Exception:  # one provider failing must not stop the others
            weather.log.exception("weather.download_failed", model=name)
            failed.append(name)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def process(
    start: Annotated[datetime, DateOpt],
    end: Annotated[datetime, DateOpt],
    raw_dir: Annotated[Path, typer.Option()] = weather.DEFAULT_RAW_DIR,
    processed_dir: Annotated[Path, typer.Option()] = weather.DEFAULT_PROCESSED_DIR,
) -> None:
    """As-of 30-min features of the configured model; transforms logged and saved to report.json."""
    configure_logging()
    features, _ = weather.process(load_forecast(), start.date(), end.date(), raw_dir, processed_dir)
    typer.echo(f"{features.height} intervals x {features.width - 2} features -> {processed_dir}")


if __name__ == "__main__":
    app()
