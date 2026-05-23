"""Unit tests for metrics module — cumulative anchor logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import metrics


def _build_weekly_df(rows: list[tuple[datetime, float, bool, bool]]) -> pl.DataFrame:
    """rows: list of (ts, value, is_subagent, is_selected)."""
    return pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "value": [r[1] for r in rows],
            "is_subagent": [r[2] for r in rows],
            "is_selected": [r[3] for r in rows],
        },
        schema={
            "ts": pl.Datetime("ms", "UTC"),
            "value": pl.Float64,
            "is_subagent": pl.Boolean,
            "is_selected": pl.Boolean,
        },
    )


def test_weekly_cumulative_total_uses_all_rows_not_just_selected():
    """The selected_mask_col must NOT affect cumulative_total — only cumulative_selected."""
    base = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)  # Mon after Sun reset
    rows = [
        (base + timedelta(hours=0), 10.0, False, False),
        (base + timedelta(hours=1), 20.0, False, True),
        (base + timedelta(hours=2), 30.0, False, True),
    ]
    df = _build_weekly_df(rows)
    out = metrics.weekly_burn_since_reset(
        df, value_col="value", selected_mask_col="is_selected"
    )
    # Polars >= 1.30: cast list to a Series and .implode() so is_in works with us/ms-aligned Datetimes.
    ts_series = pl.Series("ts", [r[0] for r in rows]).cast(pl.Datetime("ms", "UTC")).implode()
    real_rows = out.filter(pl.col("ts").is_in(ts_series)).sort("ts")
    assert real_rows["cumulative_total"].to_list() == [10.0, 30.0, 60.0]
    assert real_rows["cumulative_selected"].to_list() == [0.0, 20.0, 50.0]


def test_weekly_cumulative_anchor_survives_view_cropping():
    """The bug: previously, filtering df before passing in restarted cumulative at 0.
    Fix: compute against full df, crop the result for display — cumulative carries over."""
    base = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)  # Mon after Sun reset
    rows = [
        (base + timedelta(hours=0), 100.0, False, True),
        (base + timedelta(hours=12), 200.0, False, True),
        (base + timedelta(days=2),   300.0, False, True),  # Wed
    ]
    df = _build_weekly_df(rows)
    full = metrics.weekly_burn_since_reset(df, value_col="value", selected_mask_col="is_selected")
    display_start = base + timedelta(days=2)
    cropped = full.filter(pl.col("ts") >= display_start)
    wed_ts = pl.Series("ts", [rows[2][0]]).cast(pl.Datetime("ms", "UTC")).implode()
    wed_row = cropped.filter(pl.col("ts").is_in(wed_ts))
    assert wed_row["cumulative_total"].item() == 600.0  # 100+200+300, not 300


def test_weekly_no_mask_selected_equals_total():
    """When selected_mask_col=None, cumulative_selected must equal cumulative_total."""
    base = datetime(2026, 5, 18, 10, 0, tzinfo=timezone.utc)
    rows = [
        (base, 10.0, False, False),
        (base + timedelta(hours=1), 20.0, True, False),
    ]
    df = _build_weekly_df(rows)
    out = metrics.weekly_burn_since_reset(df, value_col="value", selected_mask_col=None)
    ts_filter = pl.Series("ts", [r[0] for r in rows]).cast(pl.Datetime("ms", "UTC")).implode()
    real = out.filter(pl.col("ts").is_in(ts_filter)).sort("ts")
    assert real["cumulative_selected"].to_list() == real["cumulative_total"].to_list()


def test_five_hour_cumulative_total_uses_all_rows_not_just_selected():
    base = datetime(2026, 5, 22, 9, 0, tzinfo=timezone.utc)
    # All within one 5h window
    rows = [
        (base + timedelta(minutes=0),  10.0, False, False),
        (base + timedelta(minutes=30), 20.0, False, True),
        (base + timedelta(minutes=60), 30.0, False, True),
    ]
    df = _build_weekly_df(rows)  # same schema works
    out = metrics.five_hour_burn_since_reset(
        df, value_col="value", selected_mask_col="is_selected",
    )
    ts_filter = pl.Series("ts", [r[0] for r in rows]).cast(pl.Datetime("ms", "UTC")).implode()
    real_rows = out.filter(pl.col("ts").is_in(ts_filter)).sort("ts")
    assert real_rows["cumulative_total"].to_list() == [10.0, 30.0, 60.0]
    assert real_rows["cumulative_selected"].to_list() == [0.0, 20.0, 50.0]


def test_five_hour_cumulative_anchor_survives_view_cropping():
    base = datetime(2026, 5, 22, 9, 0, tzinfo=timezone.utc)
    rows = [
        (base + timedelta(minutes=0),   100.0, False, True),
        (base + timedelta(minutes=60),  200.0, False, True),
        (base + timedelta(minutes=120), 300.0, False, True),
    ]
    df = _build_weekly_df(rows)
    full = metrics.five_hour_burn_since_reset(df, value_col="value", selected_mask_col="is_selected")
    cropped = full.filter(pl.col("ts") >= base + timedelta(minutes=120))
    last_row = cropped.filter(pl.col("ts") == rows[2][0])
    assert last_row["cumulative_total"].item() == 600.0


def test_five_hour_no_mask_selected_equals_total():
    """When selected_mask_col=None, cumulative_selected must equal cumulative_total."""
    base = datetime(2026, 5, 22, 9, 0, tzinfo=timezone.utc)
    rows = [
        (base, 10.0, False, False),
        (base + timedelta(minutes=30), 20.0, True, False),
    ]
    df = _build_weekly_df(rows)
    out = metrics.five_hour_burn_since_reset(df, value_col="value", selected_mask_col=None)
    ts_filter = pl.Series("ts", [r[0] for r in rows]).cast(pl.Datetime("ms", "UTC")).implode()
    real = out.filter(pl.col("ts").is_in(ts_filter)).sort("ts")
    assert real["cumulative_selected"].to_list() == real["cumulative_total"].to_list()
