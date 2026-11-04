from datetime import date, datetime
from zoneinfo import ZoneInfo

from battery_optimisation.config import load_forecast
from battery_optimisation.timing import day_start, delivery_day, issue_time

SGT = ZoneInfo("Asia/Singapore")
CFG = load_forecast()  # issue_lead_intervals = 2


def test_day_start_is_local_midnight() -> None:
    assert day_start(date(2026, 1, 2)) == datetime(2026, 1, 2, tzinfo=SGT)


def test_forecast_for_a_day_is_issued_at_its_first_gate_closure() -> None:
    assert issue_time(date(2026, 1, 2), CFG) == datetime(2026, 1, 1, 23, 0, tzinfo=SGT)
    assert delivery_day(datetime(2026, 1, 1, 23, 0, tzinfo=SGT), CFG) == date(2026, 1, 2)
    no_lead = CFG.model_copy(update={"issue_lead_intervals": 0})
    assert issue_time(date(2026, 1, 2), no_lead) == datetime(2026, 1, 2, tzinfo=SGT)
