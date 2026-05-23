"""Unit tests for per-session cost attribution and binning."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

import metrics

BASE = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)


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
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.1,
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
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
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
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
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.1, "util_7d": 0.0,
         "resets_5h_iso": "w2", "resets_7d_iso": "wk1"},  # new 5h window
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
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.2, "util_7d": 0.0,
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
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
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
        {"sampled_at": BASE + timedelta(minutes=5), "util_5h": 0.4, "util_7d": 0.0,
         "resets_5h_iso": "w1", "resets_7d_iso": "wk1"},
    ])
    sessions, _ = metrics.session_cost_attribution(df, log)
    row = sessions.filter(pl.col("session_id") == "A")
    assert abs(row["attributed_pct_5h"].item() - 0.4) < 1e-9  # both turns counted
    assert row["n_requests"].item() == 1  # only the main turn


def test_empty_inputs_return_empty():
    empty = _cache([])
    sessions, diag = metrics.session_cost_attribution(empty, _log([]))
    assert sessions.is_empty()
    assert diag == {"unattributed_5h": 0.0, "unattributed_7d": 0.0}
