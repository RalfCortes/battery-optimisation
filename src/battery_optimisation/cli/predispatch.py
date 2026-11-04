"""EMC pre-dispatch archiving commands (meant to run from cron; see crontab/).

Examples::

    uv run predispatch fetch                 # archive the current run of all feeds
    uv run predispatch fetch --feed energy --json-logs
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from battery_optimisation.data import predispatch
from battery_optimisation.data.predispatch import Feed
from battery_optimisation.logs import configure_logging

# Plain tracebacks: cron logs are JSON lines, not a terminal.
app = typer.Typer(
    help="EMC pre-dispatch forecast archive.", no_args_is_help=True, pretty_exceptions_enable=False
)


@app.callback()
def _root() -> None:
    """EMC pre-dispatch forecast archive."""


@app.command()
def fetch(
    feed: Annotated[list[Feed] | None, typer.Option(help="Repeatable; default all.")] = None,
    raw_dir: Annotated[Path, typer.Option()] = predispatch.DEFAULT_RAW_DIR,
    overwrite: Annotated[bool, typer.Option(help="Rewrite runs already on disk.")] = False,
    json_logs: Annotated[bool, typer.Option(help="One JSON object per log line.")] = False,
) -> None:
    """Fetch the current pre-dispatch run and append it to the archive."""
    configure_logging(json=json_logs)
    try:
        predispatch.archive(feed or tuple(Feed), raw_dir, overwrite)
    except RuntimeError:
        raise typer.Exit(code=1) from None  # details already logged per feed


if __name__ == "__main__":
    app()
