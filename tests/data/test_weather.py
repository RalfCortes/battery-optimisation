import json
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from battery_optimisation.columns import Col
from battery_optimisation.config import GapFill, load_forecast
from battery_optimisation.data import weather
from battery_optimisation.data.weather import Transform, WeatherVar, feature_name

SGT = ZoneInfo("Asia/Singapore")
FCFG = load_forecast()
CFG = FCFG.weather
P0 = CFG.points[0]


def one_point(cfg=CFG):
    grid = cfg.grid.model_copy(update={"latitudes": [P0[0]], "longitudes": [P0[1]]})
    return cfg.model_copy(update={"grid": grid})


def utc(y, m, d, h) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


def sgt(y, m, d, h=0, mi=0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=SGT)


# --------------------------------------------------------------------------- as-of rule


def test_asof_run_respects_publication_delay() -> None:
    issue = sgt(2026, 1, 2)  # = 2026-01-01 16:00 UTC
    assert weather.asof_run(issue, CFG) == utc(2026, 1, 1, 6)  # 06Z + 8 h = 14Z <= 16Z; 12Z is not
    quick = CFG.model_copy(update={"availability_delay_h": 4, "run_hours_utc": [0, 6, 12, 18]})
    assert weather.asof_run(issue, quick) == utc(2026, 1, 1, 12)  # 12Z + 4 h = 16Z: just usable
    # 16:52 SGT = 08:52Z; minus 8 h = 00:52Z -> the 00Z run of the same day
    every6h = CFG.model_copy(update={"run_hours_utc": [0, 6, 12, 18]})
    assert weather.asof_run(sgt(2026, 1, 1, 16, 52), every6h) == utc(2026, 1, 1, 0)
    assert weather.asof_run(sgt(2026, 1, 1, 16, 52), CFG) == utc(2025, 12, 31, 18)  # 06Z/18Z only


def test_needed_runs_one_per_issue_day() -> None:
    runs = weather.needed_runs(date(2026, 1, 1), date(2026, 1, 3), FCFG)
    assert runs == [utc(2025, 12, 31, 6), utc(2026, 1, 1, 6), utc(2026, 1, 2, 6)]


def test_request_for_a_run() -> None:
    params = weather.request_params(utc(2026, 1, 1, 6), CFG, model="icon_global")
    assert params["run"] == "2026-01-01T06:00"
    assert params["models"] == "icon_global"
    assert params["minutely_15"].split(",") == [str(v) for v in CFG.variables]
    assert params["forecast_days"] == str(CFG.forecast_days)
    assert len(params["latitude"].split(",")) == len(CFG.points)


# --------------------------------------------------------------------------- parse / process


def run_payload(run: datetime, value_of, points=(P0,)) -> str:
    """Single-run response, 15-min steps for 2 days from the run, local (SGT) stamps."""
    times = [run.astimezone(SGT) + timedelta(minutes=15 * i) for i in range(2 * 96)]
    locs = []
    for p in points:
        series = {str(v): [value_of(v, t) for t in times] for v in WeatherVar}
        locs.append(
            {
                "latitude": p[0] + 0.01,
                "longitude": p[1],
                "minutely_15": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], **series},
            }
        )
    return json.dumps(locs)


def raw_runs(runs_values: dict[datetime, float], cfg=None) -> pl.DataFrame:
    cfg = cfg or one_point()
    frames = []
    for run, base in runs_values.items():
        text = run_payload(run, lambda v, t, b=base: b + t.hour + t.minute / 60)
        frames.append(weather.parse_run(text, run, cfg.points, cfg))
    return pl.concat(frames)


def test_parse_run_labels_run_and_points() -> None:
    df = raw_runs({utc(2026, 1, 1, 6): 0.0})
    assert df[Col.RUN_TIME].unique().to_list() == [utc(2026, 1, 1, 6)]
    assert df[Col.LATITUDE].unique().to_list() == [P0[0]]
    assert df[Col.INTERVAL_START][0] == sgt(2026, 1, 1, 14)


def test_each_day_uses_its_asof_run_only() -> None:
    """Day D's features come from the run usable at D's issue time; later runs never leak in."""
    cfg = one_point()
    raw = raw_runs({utc(2026, 1, 1, 6): 0.0, utc(2026, 1, 1, 12): 1000.0}, cfg)  # 12Z: too late
    wide, report = weather.to_features(
        raw, FCFG.model_copy(update={"weather": cfg}), date(2026, 1, 2), date(2026, 1, 2)
    )
    temp = wide[feature_name(WeatherVar.TEMPERATURE, P0)]
    assert wide.height == 48
    assert temp.max() < 1000  # nothing from the 12Z run
    # interval [00:00, 00:30) = steps stamped 00:15 and 00:30 of the 06Z run -> 0.25, 0.5
    assert temp[0] == pytest.approx(0.375)
    assert wide[Col.WEATHER_LEAD_H].to_list()[:2] == [10.0, 10.5]  # 00:00 SGT = 16Z, run 06Z
    asof = report.step(Transform.ASOF)
    assert asof.runs_used == [utc(2026, 1, 1, 6)]
    assert asof.days_without_run == []


def test_missing_run_falls_back_and_uncovered_days_are_reported() -> None:
    cfg = one_point()
    raw = raw_runs({utc(2026, 1, 1, 6): 0.0}, cfg)  # Jan 3's and Jan 4's runs are missing
    fcfg = FCFG.model_copy(update={"weather": cfg})
    wide, report = weather.to_features(raw, fcfg, date(2026, 1, 2), date(2026, 1, 4))
    col = feature_name(WeatherVar.TEMPERATURE, P0)
    day3 = wide.filter(pl.col(Col.INTERVAL_START).dt.date() == date(2026, 1, 3))[col]
    assert day3.null_count() < 48  # older run (06Z Jan 1) covers Jan 3 until 14:00
    asof = report.step(Transform.ASOF)
    assert asof.days_on_older_run == 2
    assert asof.days_without_run == [date(2026, 1, 4)]  # beyond the older run's 48 h


def test_gap_fill_is_limited_and_reported() -> None:
    cfg = one_point()
    run = utc(2026, 1, 1, 6)

    def value(v, t):
        humid_gap = v is WeatherVar.HUMIDITY and t.day == 2 and t.hour in (2, 3)  # 8 steps
        temp_gap = v is WeatherVar.TEMPERATURE and (t.day, t.hour, t.minute) == (2, 0, 30)  # 1
        return None if humid_gap or temp_gap else 1.0

    text = run_payload(run, value)
    raw = weather.parse_run(text, run, cfg.points, cfg)
    wide, report = weather.to_features(
        raw, FCFG.model_copy(update={"weather": cfg}), date(2026, 1, 2), date(2026, 1, 2)
    )
    assert wide[feature_name(WeatherVar.TEMPERATURE, P0)].null_count() == 0  # 1 step: filled
    assert wide[feature_name(WeatherVar.HUMIDITY, P0)].null_count() > 0  # 8 > max 4: kept
    fill = report.step(Transform.GAP_FILL)
    assert fill.values_filled[WeatherVar.TEMPERATURE] == 1
    assert CFG.gap_fill.method is GapFill.LINEAR


def test_report_lists_every_transform_in_order() -> None:
    cfg = one_point()
    raw = raw_runs({utc(2026, 1, 1, 6): 0.0}, cfg)
    _, report = weather.to_features(
        raw, FCFG.model_copy(update={"weather": cfg}), date(2026, 1, 2), date(2026, 1, 2)
    )
    assert [s.name for s in report.steps] == [
        Transform.RAW,
        Transform.REGRID,
        Transform.GAP_FILL,
        Transform.AGGREGATE,
        Transform.ASOF,
        Transform.PIVOT,
    ]


def test_feature_names_follow_config() -> None:
    assert len(weather.feature_names(CFG)) == len(CFG.variables) * len(CFG.points) == 45


def test_raw_runs_are_kept_per_provider(tmp_path) -> None:
    path = weather.run_path(utc(2026, 1, 1, 6), "icon_global", tmp_path)
    assert path == tmp_path / "icon_global" / "2026-01" / "20260101T06.parquet"


def test_trajectory_for_an_issue_comes_from_its_asof_run() -> None:
    """Intraday issues: the horizon weather is the run usable at the issue time itself."""
    cfg = one_point()
    fcfg = FCFG.model_copy(update={"weather": cfg})
    runs = {utc(2026, 1, 1, 6): 0.0, utc(2026, 1, 1, 18): 1000.0}
    wx = weather.WeatherTrajectories.from_raw(raw_runs(runs, cfg), fcfg)
    stamps = [sgt(2026, 1, 2, 12), sgt(2026, 1, 2, 12, 30)]
    early = wx.for_issue(sgt(2026, 1, 2, 9), stamps)  # 01Z - 8 h -> 06Z Jan 1
    late = wx.for_issue(sgt(2026, 1, 2, 10), stamps)  # 02Z - 8 h -> 18Z Jan 1
    col = feature_name(WeatherVar.TEMPERATURE, P0)
    assert early[col].max() < 1000 <= late[col].min()
    assert early[Col.INTERVAL_START].to_list() == stamps
    with pytest.raises(ValueError, match="run"):
        wx.for_issue(sgt(2026, 1, 1, 12), stamps)  # usable run (18Z Dec 31) not downloaded


def test_missing_run_falls_back_to_an_older_usable_one() -> None:
    cfg = one_point()
    fcfg = FCFG.model_copy(update={"weather": cfg})
    wx = weather.WeatherTrajectories.from_raw(raw_runs({utc(2026, 1, 1, 6): 0.0}, cfg), fcfg)
    stamps = [sgt(2026, 1, 2, 12)]
    # at 10:00 the 18Z run is the as-of one, but it is missing: the 06Z run is used
    out = wx.for_issue(sgt(2026, 1, 2, 10), stamps)
    assert out[feature_name(WeatherVar.TEMPERATURE, P0)][0] < 1000
    assert wx.fallbacks == 1
    # never a run that was not yet usable: 06Z Jan 2 is usable from 22:00 SGT only
    later = weather.WeatherTrajectories.from_raw(raw_runs({utc(2026, 1, 2, 6): 5.0}, cfg), fcfg)
    with pytest.raises(ValueError, match="run"):
        later.for_issue(sgt(2026, 1, 2, 21), stamps)


def test_download_backs_off_when_rate_limited(tmp_path, monkeypatch) -> None:
    import requests

    calls, sleeps = [], []

    class Resp:
        def __init__(self, status: int) -> None:
            self.status_code, self.text = (
                status,
                run_payload(utc(2026, 1, 1, 6), lambda v, t: 1.0, CFG.points),
            )

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    statuses = iter([429, 429, 200])

    def fake_get(self, url, params=None, timeout=None):
        calls.append(params["run"])
        return Resp(next(statuses))

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: sleeps.append(s))
    written = weather.download_runs([utc(2026, 1, 1, 6)], CFG, "ecmwf_ifs", tmp_path)
    assert len(written) == 1 and len(calls) == 3
    assert sleeps[0] < sleeps[1]  # backoff grows


def test_download_skips_a_run_the_server_fails_on(tmp_path, monkeypatch) -> None:
    import requests

    class Resp:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.text = run_payload(utc(2026, 1, 1, 6), lambda v, t: 1.0, CFG.points)

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    def fake_get(self, url, params=None, timeout=None):
        return Resp(500 if params["run"] == "2026-01-01T06:00" else 200)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    written = weather.download_runs(
        [utc(2026, 1, 1, 6), utc(2026, 1, 1, 18)], CFG, "ecmwf_ifs", tmp_path
    )
    assert [p.name for p in written] == ["20260101T18.parquet"]  # the failing run is skipped


def test_blank_runs_count_as_missing() -> None:
    """A run that came back (mostly) empty must not be used: fall back to the previous one."""
    cfg = one_point()
    fcfg = FCFG.model_copy(update={"weather": cfg})
    good = raw_runs({utc(2026, 1, 1, 6): 0.0}, cfg)
    blank = raw_runs({utc(2026, 1, 1, 18): 0.0}, cfg).with_columns(
        pl.lit(None, pl.Float64).alias(c) for c in [weather.COLUMN[v] for v in cfg.variables]
    )
    raw = pl.concat([good, blank])
    wx = weather.WeatherTrajectories.from_raw(raw, fcfg)
    out = wx.for_issue(sgt(2026, 1, 2, 10), [sgt(2026, 1, 2, 12)])  # as-of run = the blank 18Z
    assert out[feature_name(WeatherVar.TEMPERATURE, P0)][0] is not None
    assert wx.fallbacks == 1
    daily, report = weather.to_features(raw, fcfg, date(2026, 1, 3), date(2026, 1, 3))
    # Jan 3's as-of run is 06Z Jan 2 (not downloaded): falls back to the latest complete run
    assert report.step(Transform.ASOF).days_without_run == []
