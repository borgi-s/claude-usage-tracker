from datetime import datetime, date, timezone
import polars as pl
import metrics

# NOTE: use real tz-aware datetime objects — `pl.datetime(...)` inside a dict is a
# Polars EXPRESSION, not a datetime, and yields an Object column that breaks `.dt.*`.

def _dt(day, hour):
    return datetime(2026, 5, day, hour, 0, tzinfo=timezone.utc)


BASE = [
    {"ts": _dt(23, 10), "is_subagent": False, "dollar_cost": 1.0, "machine": "laptop"},
    {"ts": _dt(23, 11), "is_subagent": True,  "dollar_cost": 2.0, "machine": "server"},
    {"ts": _dt(24, 10), "is_subagent": False, "dollar_cost": 4.0, "machine": "laptop"},
]


def test_by_subagent():
    out = metrics.daily_stacked(pl.DataFrame(BASE), by="is_subagent")
    row = out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)
    assert row["main"] == 1.0 and row["subagent"] == 2.0


def test_by_machine():
    out = metrics.daily_stacked(pl.DataFrame(BASE), by="machine")
    row = out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)
    assert row["laptop"] == 1.0 and row["server"] == 2.0


def test_by_machine_fallback_when_absent():
    df = pl.DataFrame([{k: v for k, v in r.items() if k != "machine"} for r in BASE])
    out = metrics.daily_stacked(df, by="machine")
    assert "local" in out.columns
    assert out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)["local"] == 3.0
