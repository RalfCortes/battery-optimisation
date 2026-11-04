"""Delivery days, forecast issue times and gate closure (Asia/Singapore, 30-min intervals).

A delivery day D runs from D 00:00 to D 23:30 (interval starts). Offers for interval t are fixed
``gate_closure_intervals`` before t. The forecast for day D is therefore issued
``issue_lead_intervals`` before D 00:00 (normally = gate closure of D's first interval), using
data strictly before the issue time.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from battery_optimisation.config import ForecastConfig
from battery_optimisation.data.nems import TZ

INTERVAL = timedelta(minutes=30)
SGT = ZoneInfo(TZ)


def day_start(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time(), SGT)


def issue_time(delivery: date, cfg: ForecastConfig) -> datetime:
    """When the forecast for delivery day ``delivery`` is made."""
    return day_start(delivery) - cfg.issue_lead_intervals * INTERVAL


def delivery_day(issued_at: datetime, cfg: ForecastConfig) -> date:
    return (issued_at.astimezone(SGT) + cfg.issue_lead_intervals * INTERVAL).date()
