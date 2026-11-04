"""NEMS market data commands.

Examples::

    uv run nems download                                   # full 5-year history
    uv run nems download --start 2024-01-01 --series usep
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from battery_optimisation.data import nems
from battery_optimisation.logs import configure_logging

app = typer.Typer(help="EMC NEMS market data.", no_args_is_help=True)

Series = StrEnum("Series", {name: name for name in nems.SERIES})
DateOpt = typer.Option(formats=["%Y-%m-%d"], help="YYYY-MM-DD")


@app.callback()
def _root() -> None:
    """EMC NEMS market data."""


@app.command()
def download(
    start: Annotated[datetime | None, DateOpt] = None,
    end: Annotated[datetime | None, DateOpt] = None,
    series: Annotated[list[Series] | None, typer.Option(help="Repeatable; default all.")] = None,
    raw_dir: Annotated[Path, typer.Option()] = nems.DEFAULT_RAW_DIR,
    overwrite: Annotated[bool, typer.Option(help="Re-fetch months already on disk.")] = False,
) -> None:
    """Download USEP, reserve and regulation history to monthly parquet files."""
    configure_logging()
    names = [s.value for s in series] if series else list(nems.SERIES)
    written = nems.download(
        names,
        start.date() if start else None,
        end.date() if end else None,
        raw_dir,
        overwrite,
    )
    typer.echo(f"wrote {len(written)} file(s) under {raw_dir}")


if __name__ == "__main__":
    app()
