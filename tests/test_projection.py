from datetime import datetime, timedelta, timezone
import polars as pl
import metrics
from tests.test_reported_util import LOG_SCHEMA, _mk  # reuse fixtures


def _rows(utils, base, reset):
    return [
        {"sampled_at": base + timedelta(minutes=10 * i), "util_5h": u, "util_7d": 0.0,
         "resets_5h_iso": reset.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"}
        for i, u in enumerate(utils)
    ]


def test_rising_util_returns_finite_eta_before_reset():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 18, 0, tzinfo=timezone.utc)
    log = _mk(_rows([0.2, 0.4], base, reset))   # +0.2 over 10 min -> 0.02/min
    now = base + timedelta(minutes=10)
    proj = metrics.project_time_to_cap(log, now, "5h")
    # 0.6 util remaining at 0.02/min = 30 min
    assert proj.eta is not None
    assert abs(proj.eta.total_seconds() - 30 * 60) < 90
    assert proj.before_reset is True


def test_flat_util_returns_none():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 18, 0, tzinfo=timezone.utc)
    log = _mk(_rows([0.5, 0.5], base, reset))
    proj = metrics.project_time_to_cap(log, base + timedelta(minutes=10), "5h")
    assert proj.eta is None


def test_eta_past_reset_flags_before_reset_false():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = base + timedelta(minutes=15)  # window resets very soon
    log = _mk(_rows([0.2, 0.25], base, reset))  # slow slope -> 100% long after reset
    proj = metrics.project_time_to_cap(log, base + timedelta(minutes=10), "5h")
    assert proj.before_reset is False
