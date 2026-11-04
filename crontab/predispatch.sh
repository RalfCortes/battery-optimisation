#!/usr/bin/env bash
# Archive the current EMC pre-dispatch run (see src/battery_optimisation/data/predispatch.py).
# Called from cron via predispatch.cron. Safe to run by hand or twice in a row.

# cron starts with an empty environment: load your shell's proxy and CA settings from ~/.bashrc.
# (Before `set -eu`: .bashrc is not written to run under strict mode.)
source "$HOME/.bashrc" >/dev/null 2>&1 || true

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
cd "$REPO"

# flock -n: skip this run if the previous one is still going (e.g. slow server).
exec flock -n "$LOG_DIR/.predispatch.lock" \
    "$UV" run --frozen --quiet predispatch fetch --json-logs \
    >> "$LOG_DIR/predispatch.log" 2>&1
