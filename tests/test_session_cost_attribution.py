"""Unit tests for per-session cost attribution and binning."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

import metrics

BASE = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)

# resets_*_iso are jittered ISO timestamps of the reset instant. Two readings of the
# SAME window differ only in sub-second jitter (round to the same minute); a different
# window is hours away. These fixtures mirror that reality.
R5 = "2026-05-18T15:00:00.069366+00:00"        # 5h window A
R5b = "2026-05-18T15:00:00.773137+00:00"       # same window A, different jitter
R5_NEXT = "2026-05-18T20:00:00.123456+00:00"   # 5h window B (a reset later)
R7 = "2026-05-24T07:00:00.111111+00:00"        # weekly window
R7b = "2026-05-24T07:00:00.888888+00:00"       # same weekly window, different jitter


def _cache(rows: list[dict]) -> pl.DataFrame:
    """rows: dicts with ts, session_id, is_subagent, output_tokens,
    prompt_tokens, raw_total_tokens."""
    return pl.DataFrame(
        rows,
        schema={
            "ts": pl.Datetime("ms", "UTC"),
            "session_id": pl.Utf8,
            "is_subagent": pl.Boolean,
            "output_tokens": pl.Int64,
            "prompt_tokens": pl.Int64,
            "raw_total_tokens": pl.Int64,
        },
    )


def _log(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "sampled_at": pl.Datetime("ms", "UTC"),
            "util_5h": pl.Float64,
            "util_7d": pl.Float64,
            "resets_5h_iso": pl.Utf8,
            "resets_7d_iso": pl.Utf8,
        },
    )


def test_single_session_owns_window_gets_full_delta():
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1000, "raw_total_tokens": 1100},
        {"ts": BASE + timedelta(minutes=4), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 2000, "raw_total_tokens": 2100},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.1,
         "resets_5h_iso": R5b, "resets_7d_iso": R7b},
    ])
    sessions, diag = metrics.session_cost_attribution(df, log)
    row = sessions.filter(pl.col("session_id") == "A")
    assert abs(row["attributed_pct_5h"].item() - 0.4) < 1e-9
    assert abs(row["attributed_pct_weekly"].item() - 0.1) < 1e-9
    assert row["prompt_tokens"].item() == 3000
    assert row["n_requests"].item() == 2
    assert diag["unattributed_5h"] == 0.0


def test_overlapping_sessions_split_by_output_share():
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 300, "prompt_tokens": 1000, "raw_total_tokens": 1300},
        {"ts": BASE + timedelta(minutes=3), "session_id": "B", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 500, "raw_total_tokens": 600},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": R5b, "resets_7d_iso": R7b},
    ])
    sessions, _ = metrics.session_cost_attribution(df, log)
    a = sessions.filter(pl.col("session_id") == "A")["attributed_pct_5h"].item()
    b = sessions.filter(pl.col("session_id") == "B")["attributed_pct_5h"].item()
    assert abs(a - 0.3) < 1e-9   # 0.4 * 300/400
    assert abs(b - 0.1) < 1e-9   # 0.4 * 100/400


def test_reset_straddling_pair_is_skipped():
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1000, "raw_total_tokens": 1100},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.9, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.1, "util_7d": 0.0,
         "resets_5h_iso": R5_NEXT, "resets_7d_iso": R7b},  # new 5h window (hours later)
    ])
    sessions, diag = metrics.session_cost_attribution(df, log)
    assert sessions.filter(pl.col("session_id") == "A")["attributed_pct_5h"].item() == 0.0
    assert diag["unattributed_5h"] == 0.0


def test_delta_with_no_turns_is_unattributed():
    df = _cache([
        {"ts": BASE - timedelta(minutes=30), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1000, "raw_total_tokens": 1100},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.2, "util_7d": 0.0,
         "resets_5h_iso": R5b, "resets_7d_iso": R7b},
    ])
    sessions, diag = metrics.session_cost_attribution(df, log)
    assert sessions.filter(pl.col("session_id") == "A")["attributed_pct_5h"].item() == 0.0
    assert abs(diag["unattributed_5h"] - 0.2) < 1e-9


def test_subagent_output_folds_into_parent_session():
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 50, "prompt_tokens": 1000, "raw_total_tokens": 1050},
        {"ts": BASE + timedelta(minutes=3), "session_id": "A", "is_subagent": True,
         "output_tokens": 50, "prompt_tokens": 800, "raw_total_tokens": 850},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": R5b, "resets_7d_iso": R7b},
    ])
    sessions, _ = metrics.session_cost_attribution(df, log)
    row = sessions.filter(pl.col("session_id") == "A")
    assert abs(row["attributed_pct_5h"].item() - 0.4) < 1e-9  # both turns counted
    assert row["n_requests"].item() == 1  # only the main turn


def test_jittered_reset_ids_same_window_attributes():
    """Regression: sub-second jitter in resets_5h_iso must NOT split one window."""
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1000, "raw_total_tokens": 1100},
        {"ts": BASE + timedelta(minutes=7), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1500, "raw_total_tokens": 1600},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": "2026-05-18T15:00:00.069366+00:00", "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.2, "util_7d": 0.0,
         "resets_5h_iso": "2026-05-18T15:00:00.512741+00:00", "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=10), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": "2026-05-18T15:00:00.998003+00:00", "resets_7d_iso": R7},
    ])
    sessions, _ = metrics.session_cost_attribution(df, log)
    # All three samples are the same window → two valid intervals → 0.2 + 0.2 = 0.4.
    assert abs(sessions.filter(pl.col("session_id") == "A")["attributed_pct_5h"].item() - 0.4) < 1e-9


def test_null_reset_id_excluded():
    """A pair where either reset id is null cannot be assigned a window → excluded."""
    df = _cache([
        {"ts": BASE + timedelta(minutes=2), "session_id": "A", "is_subagent": False,
         "output_tokens": 100, "prompt_tokens": 1000, "raw_total_tokens": 1100},
    ])
    log = _log([
        {"sampled_at": BASE, "util_5h": 0.0, "util_7d": 0.0,
         "resets_5h_iso": None, "resets_7d_iso": R7},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": R5, "resets_7d_iso": R7b},
    ])
    sessions, diag = metrics.session_cost_attribution(df, log)
    assert sessions.filter(pl.col("session_id") == "A")["attributed_pct_5h"].item() == 0.0
    assert diag["unattributed_5h"] == 0.0


def test_empty_inputs_return_empty():
    empty = _cache([])
    sessions, diag = metrics.session_cost_attribution(empty, _log([]))
    assert sessions.is_empty()
    assert diag == {"unattributed_5h": 0.0, "unattributed_7d": 0.0}


def test_bin_sessions_quantile_means_and_counts():
    sessions = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [10.0, 20.0, 30.0, 40.0],
    })
    out = metrics.bin_sessions(sessions, "x", "y", n_bins=2)
    assert out.height == 2
    assert out["n"].to_list() == [2, 2]
    assert out["mean_y"].to_list() == [15.0, 35.0]      # (10,20) and (30,40)
    assert out["bin_median_x"].to_list() == [1.5, 3.5]


def test_bin_sessions_single_member_bin_has_null_std():
    sessions = pl.DataFrame({"x": [1.0], "y": [5.0]})
    out = metrics.bin_sessions(sessions, "x", "y", n_bins=4)
    assert out.height == 1
    assert out["n"].item() == 1
    assert out["std_y"].item() is None  # std of one element


def test_bin_sessions_more_bins_than_rows_collapses():
    sessions = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    out = metrics.bin_sessions(sessions, "x", "y", n_bins=10)
    assert out.height == 3  # clamped to row count, one session per bin


def test_bin_sessions_empty_returns_empty():
    empty = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64})
    out = metrics.bin_sessions(empty, "x", "y", n_bins=8)
    assert out.is_empty()
