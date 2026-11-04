#!/usr/bin/env bash
# Archive day-ahead weather forecasts for every provider in config/forecast.yaml (weather section).
# Fetches every run of the last 2 days (resumable: runs on disk are skipped).
set +u
source "$HOME/.bashrc" >/dev/null 2>&1 || true   # proxy + CA settings (cron has no environment)
set -euo pipefail
unset UV_INDEX                                     # never resolve packages from that feed

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
cd "$REPO"

exec flock -n "$LOG_DIR/.weather.lock" \
    "$UV" run --frozen --quiet weather download --json-logs \
    >> "$LOG_DIR/weather.log" 2>&1
