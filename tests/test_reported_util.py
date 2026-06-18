from datetime import datetime, timedelta, timezone
import polars as pl
import pytest
import metrics

LOG_SCHEMA = {  # subset of calibration_log SCHEMA that these helpers read
    "sampled_at": pl.Datetime("ms", "UTC"),
    "util_5h": pl.Float64, "util_7d": pl.Float64,
    "resets_5h_iso": pl.Utf8, "resets_7d_iso": pl.Utf8,
    "rate_limit_tier": pl.Utf8,
}


def _mk(rows):
    return pl.DataFrame(rows, schema=LOG_SCHEMA)


def test_jitter_does_not_break_but_real_reset_does():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset_a = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)   # window A end
    reset_b = datetime(2026, 5, 23, 17, 0, tzinfo=timezone.utc)   # window B end (~5h later)
    rows = []
    # Window A: three samples, each resets_5h_iso jittered by a few seconds
    for i, jit in enumerate((0, 3, 7)):
        rows.append({
            "sampled_at": base + timedelta(minutes=5 * i),
            "util_5h": 0.1 * (i + 1), "util_7d": 0.2,
            "resets_5h_iso": (reset_a + timedelta(seconds=jit)).isoformat(),
            "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x",
        })
    # Window B: one sample, reset jumps ~5h forward -> real reset
    rows.append({
        "sampled_at": base + timedelta(minutes=20),
        "util_5h": 0.05, "util_7d": 0.2,
        "resets_5h_iso": reset_b.isoformat(),
        "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x",
    })
    series, _ = metrics.reported_util_series(_mk(rows), "5h", gap_break_minutes=60)
    ys = series["util_pct"].to_list()
    # exactly one None (the real reset), none among the jittered window-A samples
    assert ys.count(None) == 1
    assert ys[3] is None                              # the break is the real reset
    assert ys[:3] == pytest.approx([10.0, 20.0, 30.0])  # window A continuous (float-tolerant)


def test_sampling_gap_breaks():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc).isoformat()
    rows = [
        {"sampled_at": base, "util_5h": 0.1, "util_7d": 0.0, "resets_5h_iso": reset,
         "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x"},
        {"sampled_at": base + timedelta(hours=40), "util_5h": 0.2, "util_7d": 0.0,
         "resets_5h_iso": reset, "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},
    ]
    series, _ = metrics.reported_util_series(_mk(rows), "5h", gap_break_minutes=15)
    assert series["util_pct"].to_list().count(None) == 1


def test_pro_rows_dropped_and_cap_hits():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc).isoformat()
    rows = [
        {"sampled_at": base, "util_5h": 0.99, "util_7d": 0.0, "resets_5h_iso": reset,
         "resets_7d_iso": None, "rate_limit_tier": "default_claude_ai"},          # Pro -> dropped
        {"sampled_at": base + timedelta(minutes=5), "util_5h": 1.0, "util_7d": 0.0,
         "resets_5h_iso": reset, "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                             # Max5x, cap hit
    ]
    series, cap_hits = metrics.reported_util_series(_mk(rows), "5h")
    assert series.height == 1                # Pro row dropped
    assert cap_hits.height == 1
    assert cap_hits["util_pct"][0] == 100.0


def test_windows_over_threshold_and_peak():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset_a = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    reset_b = datetime(2026, 5, 23, 17, 0, tzinfo=timezone.utc)
    rows = [
        {"sampled_at": base, "util_5h": 0.30, "util_7d": 0.0,
         "resets_5h_iso": reset_a.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                     # window A peak 0.30 > 0.20
        {"sampled_at": base + timedelta(minutes=20), "util_5h": 0.10, "util_7d": 0.0,
         "resets_5h_iso": reset_b.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                     # window B peak 0.10 < 0.20
    ]
    log = _mk(rows)
    assert metrics.windows_over_threshold(log, "5h", 0.20) == (1, 2)
    assert metrics.peak_reported(log, "5h") == 0.30
