"""Run any command under CPU and RAM limits (see ``battery_optimisation.resources``).

Examples::

    uv run limited --mem-gb 100 --cpus 8 -- uv run jupyter execute --inplace notebooks/*.ipynb
    uv run limited --cpu-list 0-7 -- uv run bidding run
"""

from __future__ import annotations

from typing import Annotated

import typer

from battery_optimisation.logs import configure_logging
from battery_optimisation.resources import Limits, parse_cpus, run_limited

app = typer.Typer(help=__doc__.splitlines()[0], pretty_exceptions_enable=False)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    cmd: Annotated[list[str], typer.Argument(help="Command to run (after --).")],
    mem_gb: Annotated[float, typer.Option(help="RAM cap for the whole process tree (RSS).")] = 100,
    cpus: Annotated[int, typer.Option(help="Number of cores (least busy are picked).")] = 8,
    cpu_list: Annotated[str | None, typer.Option(help="Explicit cores, e.g. 0-7.")] = None,
) -> None:
    configure_logging()
    limits = Limits(mem_gb=mem_gb, cpus=parse_cpus(cpu_list) if cpu_list else cpus)
    result = run_limited(cmd, limits)
    if result.killed_for_memory:
        typer.echo(f"killed: RAM above {mem_gb} GB", err=True)
    raise typer.Exit(code=result.returncode if result.returncode >= 0 else 1)


if __name__ == "__main__":
    app()
