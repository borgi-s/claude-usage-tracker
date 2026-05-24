"""Characterization tests — lock current calibration outputs before vectorization refactor.

These tests build small synthetic fixtures, run the CURRENT implementations,
and assert exact golden values derived by executing the code on 2026-05-24.
They must ALL pass against the unmodified source; the refactor must not change
any of these results.

Functions covered:
  caps.global_cap_from_anchors       (kind="5h" and kind="weekly")
  metrics.observed_window_lengths    (two observed windows)
  metrics.effective_window_hours     (n < min_samples branch and n >= min_samples branch)
  metrics.weekly_burn_since_reset    (spanning a weekly reset boundary)
  metrics.five_hour_burn_since_reset (two gap-separated 5h windows)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

import caps
import metrics


# ---------------------------------------------------------------------------
# Shared UTC helper
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Shared fixtures (re-used across multiple tests)
# ---------------------------------------------------------------------------

# ── 5h window fixture ──────────────────────────────────────────────────────
# Two clusters separated by a 6-hour gap (> gap_hours=5.0).
# Cluster A: 2026-05-18 08:00–08:05 UTC, output_tokens = 3_000_000
# Cluster B: 2026-05-18 14:00–14:05 UTC, output_tokens = 2_000_000
_BASE_5H = _dt(2026, 5, 18, 8, 0)

_CACHE_5H = pl.DataFrame(
    {
        "ts": [
            _BASE_5H,
            _BASE_5H + timedelta(minutes=2),
            _BASE_5H + timedelta(minutes=5),
            _BASE_5H + timedelta(hours=6),                    # gap = 6 h > 5 h → new window
            _BASE_5H + timedelta(hours=6, minutes=2),
            _BASE_5H + timedelta(hours=6, minutes=5),
        ],
        "output_tokens": [1_000_000.0, 1_000_000.0, 1_000_000.0,
                          700_000.0,   700_000.0,   600_000.0],
        "is_subagent": [False] * 6,
    },
    schema={
        "ts": pl.Datetime("ms", "UTC"),
        "output_tokens": pl.Float64,
        "is_subagent": pl.Boolean,
    },
)

# One anchor per cluster, both at util_5h = 0.97 / 0.98 (≥ 0.95, ≤ 1.01)
_LOG_5H = pl.DataFrame(
    {
        "sampled_at": [
            _BASE_5H + timedelta(minutes=5),           # end of cluster A
            _BASE_5H + timedelta(hours=6, minutes=5),  # end of cluster B
        ],
        "util_5h": [0.97, 0.98],
        "util_7d":  [0.3,  0.3],
        "resets_5h_iso": [None, None],
    },
    schema={
        "sampled_at": pl.Datetime("ms", "UTC"),
        "util_5h": pl.Float64,
        "util_7d": pl.Float64,
        "resets_5h_iso": pl.Utf8,
    },
)


# ── Weekly window fixture ──────────────────────────────────────────────────
# Europe/Copenhagen is UTC+2 in summer (CEST).
# Sunday 2026-05-17 07:00 Copenhagen = 2026-05-17 05:00 UTC → weekly reset.
# Week A: Fri 2026-05-15 05:00 UTC and Sat 2026-05-16 05:00 UTC (2.5 M each)
# Week B: Mon 2026-05-18 05:00 UTC and Tue 2026-05-19 05:00 UTC (1.5 M each)
# Anchors: one per week, util_7d = 0.30 (≥ min_util=0.10)

_WEEK_RESET_UTC = _dt(2026, 5, 17, 5, 0)  # Sun 07:00 Copenhagen

_CACHE_WEEKLY = pl.DataFrame(
    {
        "ts": [
            _WEEK_RESET_UTC - timedelta(days=2),   # Fri — week A
            _WEEK_RESET_UTC - timedelta(days=1),   # Sat — week A
            _WEEK_RESET_UTC + timedelta(days=1),   # Mon — week B
            _WEEK_RESET_UTC + timedelta(days=2),   # Tue — week B
        ],
        "output_tokens": [2_500_000.0, 2_500_000.0, 1_500_000.0, 1_500_000.0],
        "is_subagent": [False] * 4,
    },
    schema={
        "ts": pl.Datetime("ms", "UTC"),
        "output_tokens": pl.Float64,
        "is_subagent": pl.Boolean,
    },
)

_LOG_WEEKLY = pl.DataFrame(
    {
        "sampled_at": [
            _WEEK_RESET_UTC - timedelta(days=1),   # Sat anchor (week A)
            _WEEK_RESET_UTC + timedelta(days=2),   # Tue anchor (week B)
        ],
        "util_5h": [0.3, 0.3],
        "util_7d":  [0.30, 0.30],
        "resets_5h_iso": [None, None],
    },
    schema={
        "sampled_at": pl.Datetime("ms", "UTC"),
        "util_5h": pl.Float64,
        "util_7d": pl.Float64,
        "resets_5h_iso": pl.Utf8,
    },
)


# ── Window-length fixtures ─────────────────────────────────────────────────
# Two windows, each 4 h 55 min long (4.9167 h), separated by a 7-hour gap.
# An activity at 03:00 UTC (5 h before window A) satisfies the gap check.

_BASE_OBS = _dt(2026, 5, 18, 8, 0)
_RESET_A = _BASE_OBS + timedelta(hours=4, minutes=55)   # 12:55 UTC
_RESET_B = _BASE_OBS + timedelta(hours=11, minutes=55)  # 19:55 UTC  (7 h later)

_CACHE_OBS = pl.DataFrame(
    {
        "ts": [
            _BASE_OBS - timedelta(hours=5),               # 03:00 — before window A
            _BASE_OBS,                                    # 08:00 — window A start
            _BASE_OBS + timedelta(minutes=2),
            _BASE_OBS + timedelta(hours=7),               # 15:00 — window B start (gap=7h>3.6h)
            _BASE_OBS + timedelta(hours=7, minutes=2),
        ],
        "output_tokens": [100.0] * 5,
        "is_subagent": [False] * 5,
    },
    schema={
        "ts": pl.Datetime("ms", "UTC"),
        "output_tokens": pl.Float64,
        "is_subagent": pl.Boolean,
    },
)

_LOG_OBS = pl.DataFrame(
    {
        "sampled_at": [_RESET_A, _RESET_B],
        "util_5h": [0.97, 0.97],
        "util_7d":  [0.3,  0.3],
        "resets_5h_iso": [_RESET_A.isoformat(), _RESET_B.isoformat()],
    },
    schema={
        "sampled_at": pl.Datetime("ms", "UTC"),
        "util_5h": pl.Float64,
        "util_7d": pl.Float64,
        "resets_5h_iso": pl.Utf8,
    },
)


# ── Five-window fixture for effective_window_hours n >= min_samples ────────
# 5 windows each 4 h 55 min, separated by 7-hour gaps, plus one far-back row.

_BASE_EFF = _dt(2026, 5, 17, 8, 0)

_EFF_WINDOW_TS: list[datetime] = []
_EFF_RESET_ISOS: list[str] = []
for _i in range(5):
    _ws = _BASE_EFF + timedelta(hours=_i * 7)
    _wr = _ws + timedelta(hours=4, minutes=55)
    _EFF_WINDOW_TS.extend([_ws, _ws + timedelta(minutes=2)])
    _EFF_RESET_ISOS.append(_wr.isoformat())

_CACHE_EFF = pl.DataFrame(
    {
        "ts": [_BASE_EFF - timedelta(hours=6)] + _EFF_WINDOW_TS,
        "output_tokens": [100.0] * (1 + len(_EFF_WINDOW_TS)),
        "is_subagent": [False] * (1 + len(_EFF_WINDOW_TS)),
    },
    schema={
        "ts": pl.Datetime("ms", "UTC"),
        "output_tokens": pl.Float64,
        "is_subagent": pl.Boolean,
    },
)

_LOG_EFF = pl.DataFrame(
    {
        "sampled_at": [_BASE_EFF + timedelta(hours=_i * 7 + 4, minutes=55) for _i in range(5)],
        "util_5h": [0.97] * 5,
        "util_7d":  [0.3] * 5,
        "resets_5h_iso": _EFF_RESET_ISOS,
    },
    schema={
        "sampled_at": pl.Datetime("ms", "UTC"),
        "util_5h": pl.Float64,
        "util_7d": pl.Float64,
        "resets_5h_iso": pl.Utf8,
    },
)


# ===========================================================================
# Tests — caps.global_cap_from_anchors
# ===========================================================================

class TestGlobalCapFromAnchors:

    def test_5h_returns_median_cap_and_anchor_count(self):
        """Two 5h windows with anchors at 0.97 and 0.98 → median of two implied caps."""
        median_cap, n_anchors = caps.global_cap_from_anchors(
            _LOG_5H, _CACHE_5H, kind="5h", gap_hours=5.0, min_util=0.95
        )
        assert n_anchors == 2
        # Cluster A: burn=3_000_000, util=0.97 → implied=3_092_783.505...
        # Cluster B: burn=2_000_000, util=0.98 → implied=2_040_816.326...
        # median of two = (3_092_783.505 + 2_040_816.326) / 2 = 2_566_799.916
        assert median_cap == pytest.approx(2566799.915842626, rel=1e-9)

    def test_5h_returns_none_when_no_qualifying_anchors(self):
        """Anchors below min_util threshold → (None, 0)."""
        log_low = _LOG_5H.with_columns(pl.lit(0.50).alias("util_5h"))
        result = caps.global_cap_from_anchors(
            log_low, _CACHE_5H, kind="5h", gap_hours=5.0, min_util=0.95
        )
        assert result == (None, 0)

    def test_weekly_returns_median_cap_and_anchor_count(self):
        """Two weekly windows with anchors at util_7d=0.30 (min_util=0.10) → median cap."""
        median_cap, n_anchors = caps.global_cap_from_anchors(
            _LOG_WEEKLY, _CACHE_WEEKLY, kind="weekly", gap_hours=5.0, min_util=0.10
        )
        assert n_anchors == 2
        # Week A anchor: burn=5_000_000 (2.5M+2.5M), util=0.30 → implied=16_666_666.67
        # Week B anchor: burn=3_000_000 (1.5M+1.5M), util=0.30 → implied=10_000_000.00
        # median of two = (16_666_666.67 + 10_000_000.00) / 2 = 13_333_333.33
        assert median_cap == pytest.approx(13333333.333333334, rel=1e-9)

    def test_weekly_respects_min_util_threshold(self):
        """Anchors below weekly min_util → (None, 0)."""
        log_below = _LOG_WEEKLY.with_columns(pl.lit(0.05).alias("util_7d"))
        result = caps.global_cap_from_anchors(
            log_below, _CACHE_WEEKLY, kind="weekly", gap_hours=5.0, min_util=0.10
        )
        assert result == (None, 0)

    def test_empty_log_returns_none(self):
        empty_log = pl.DataFrame(
            schema={
                "sampled_at": pl.Datetime("ms", "UTC"),
                "util_5h": pl.Float64,
                "util_7d": pl.Float64,
                "resets_5h_iso": pl.Utf8,
            }
        )
        assert caps.global_cap_from_anchors(
            empty_log, _CACHE_5H, kind="5h", gap_hours=5.0
        ) == (None, 0)

    def test_missing_value_col_returns_none(self):
        cache_no_col = _CACHE_5H.drop("output_tokens")
        assert caps.global_cap_from_anchors(
            _LOG_5H, cache_no_col, kind="5h", gap_hours=5.0, value_col="output_tokens"
        ) == (None, 0)


# ===========================================================================
# Tests — metrics.observed_window_lengths
# ===========================================================================

class TestObservedWindowLengths:

    def test_returns_two_lengths_from_two_resets(self):
        """Two distinct resets, each with a valid preceding gap, → two lengths."""
        lengths = metrics.observed_window_lengths(_LOG_OBS, _CACHE_OBS, default_hours=4.5)
        assert len(lengths) == 2
        # Both windows are 4 h 55 min = 4.9166... h
        assert lengths[0] == pytest.approx(4.916666666666667, rel=1e-9)
        assert lengths[1] == pytest.approx(4.916666666666667, rel=1e-9)

    def test_empty_log_returns_empty_list(self):
        empty_log = pl.DataFrame(
            schema={
                "sampled_at": pl.Datetime("ms", "UTC"),
                "util_5h": pl.Float64,
                "util_7d": pl.Float64,
                "resets_5h_iso": pl.Utf8,
            }
        )
        assert metrics.observed_window_lengths(empty_log, _CACHE_OBS) == []

    def test_sanity_filter_drops_too_short_lengths(self):
        """A window shorter than sanity_min=0.5h is discarded.

        Reset at base+10min, single cache row at base+5min → length=5min=0.083h < 0.5h.
        No prior activity so the gap check is skipped; the sanity_min filter fires.
        """
        short_base = _dt(2026, 5, 18, 8, 0)
        reset_ts = short_base + timedelta(minutes=10)
        cache_short = pl.DataFrame(
            {
                "ts": [short_base + timedelta(minutes=5)],
                "output_tokens": [100.0],
                "is_subagent": [False],
            },
            schema={
                "ts": pl.Datetime("ms", "UTC"),
                "output_tokens": pl.Float64,
                "is_subagent": pl.Boolean,
            },
        )
        log_short = pl.DataFrame(
            {
                "sampled_at": [reset_ts],
                "util_5h": [0.97],
                "util_7d":  [0.3],
                "resets_5h_iso": [reset_ts.isoformat()],
            },
            schema={
                "sampled_at": pl.Datetime("ms", "UTC"),
                "util_5h": pl.Float64,
                "util_7d": pl.Float64,
                "resets_5h_iso": pl.Utf8,
            },
        )
        result = metrics.observed_window_lengths(log_short, cache_short, default_hours=4.5)
        assert result == []

    def test_no_cache_activity_in_lookback_returns_empty(self):
        """A reset so far in the future that no cache rows fall in its lookback window."""
        # Reset 100h from base; lookback = (reset-5h, reset] contains no cache rows.
        very_far_reset = _BASE_OBS + timedelta(hours=100)
        log_far = pl.DataFrame(
            {
                "sampled_at": [very_far_reset],
                "util_5h": [0.97],
                "util_7d":  [0.3],
                "resets_5h_iso": [very_far_reset.isoformat()],
            },
            schema={
                "sampled_at": pl.Datetime("ms", "UTC"),
                "util_5h": pl.Float64,
                "util_7d": pl.Float64,
                "resets_5h_iso": pl.Utf8,
            },
        )
        result = metrics.observed_window_lengths(log_far, _CACHE_OBS, default_hours=4.5)
        assert result == []


# ===========================================================================
# Tests — metrics.effective_window_hours
# ===========================================================================

class TestEffectiveWindowHours:

    def test_returns_default_when_fewer_than_min_samples(self):
        """Only 2 observed windows with min_samples=5 → returns (default, 2)."""
        hours, n = metrics.effective_window_hours(
            _LOG_OBS, _CACHE_OBS, default=4.5, min_samples=5
        )
        assert n == 2
        assert hours == pytest.approx(4.5, rel=1e-9)

    def test_returns_median_when_at_least_min_samples(self):
        """5 observed windows each 4.9167 h with min_samples=5 → returns (median, 5)."""
        hours, n = metrics.effective_window_hours(
            _LOG_EFF, _CACHE_EFF, default=4.5, min_samples=5
        )
        assert n == 5
        # All 5 lengths are 4.9167 h → median = 4.9167 h
        assert hours == pytest.approx(4.916666666666667, rel=1e-9)

    def test_returns_median_with_lower_min_samples_threshold(self):
        """2 observed windows satisfy min_samples=2 → returns median, not default."""
        hours, n = metrics.effective_window_hours(
            _LOG_OBS, _CACHE_OBS, default=4.5, min_samples=2
        )
        assert n == 2
        assert hours == pytest.approx(4.916666666666667, rel=1e-9)


# ===========================================================================
# Tests — metrics.weekly_burn_since_reset
# ===========================================================================

class TestWeeklyBurnSinceReset:
    """Fixture: 4 data rows across a weekly reset boundary + 2 injected reset rows."""

    # Week A rows: Fri 2026-05-15, Sat 2026-05-16 (week_start = 2026-05-10 05:00 UTC)
    # Week B rows: Mon 2026-05-18, Tue 2026-05-19 (week_start = 2026-05-17 05:00 UTC)
    # values: 1000, 2000, 3000, 4000

    _ROWS = [
        _dt(2026, 5, 15, 10),
        _dt(2026, 5, 16, 10),
        _dt(2026, 5, 18, 10),
        _dt(2026, 5, 19, 10),
    ]
    _VALUES = [1000.0, 2000.0, 3000.0, 4000.0]

    @classmethod
    def _make_df(cls):
        return pl.DataFrame(
            {
                "ts": cls._ROWS,
                "value": cls._VALUES,
                "is_subagent": [False] * 4,
            },
            schema={
                "ts": pl.Datetime("ms", "UTC"),
                "value": pl.Float64,
                "is_subagent": pl.Boolean,
            },
        )

    def test_cumulative_total_resets_at_week_boundary(self):
        """cumulative_total must restart from 0 after the Sunday 07:00 reset."""
        out = metrics.weekly_burn_since_reset(self._make_df(), value_col="value")
        ts_filter = (
            pl.Series("ts", self._ROWS)
            .cast(pl.Datetime("ms", "UTC"))
            .implode()
        )
        real = out.filter(
            pl.col("ts").is_in(ts_filter) & (pl.col("cumulative_total") != 0)
        ).sort("ts")
        assert real["cumulative_total"].to_list() == [1000.0, 3000.0, 3000.0, 7000.0]

    def test_reset_rows_inserted_at_week_boundaries(self):
        """A zero-valued row must be present at each of the two week-start timestamps."""
        out = metrics.weekly_burn_since_reset(self._make_df(), value_col="value")
        # week_start for Fri/Sat = 2026-05-10 05:00 UTC
        # week_start for Mon/Tue = 2026-05-17 05:00 UTC
        reset_a = _dt(2026, 5, 10, 5)
        reset_b = _dt(2026, 5, 17, 5)
        for reset_ts in [reset_a, reset_b]:
            ts_s = pl.Series("ts", [reset_ts]).cast(pl.Datetime("ms", "UTC")).implode()
            row = out.filter(
                pl.col("ts").is_in(ts_s) & (pl.col("cumulative_total") == 0.0)
            )
            assert row.height >= 1, f"Expected zero-reset row at {reset_ts}"

    def test_total_rows_in_output(self):
        """4 data rows + 2 reset rows = 6 rows total."""
        out = metrics.weekly_burn_since_reset(self._make_df(), value_col="value")
        assert out.height == 6

    def test_no_mask_cumulative_selected_equals_total(self):
        """With no selected_mask_col, cumulative_selected must equal cumulative_total."""
        out = metrics.weekly_burn_since_reset(self._make_df(), value_col="value")
        ts_filter = (
            pl.Series("ts", self._ROWS)
            .cast(pl.Datetime("ms", "UTC"))
            .implode()
        )
        real = out.filter(
            pl.col("ts").is_in(ts_filter) & (pl.col("cumulative_total") != 0)
        ).sort("ts")
        assert real["cumulative_selected"].to_list() == real["cumulative_total"].to_list()

    def test_week_b_cumulative_values(self):
        """Week B rows must show cumulative_total of 3000 and 7000 (not 4000 and 8000)."""
        out = metrics.weekly_burn_since_reset(self._make_df(), value_col="value")
        mon = _dt(2026, 5, 18, 10)
        tue = _dt(2026, 5, 19, 10)
        ts_filter = (
            pl.Series("ts", [mon, tue])
            .cast(pl.Datetime("ms", "UTC"))
            .implode()
        )
        real = out.filter(
            pl.col("ts").is_in(ts_filter) & (pl.col("cumulative_total") != 0)
        ).sort("ts")
        assert real["cumulative_total"].to_list() == [3000.0, 7000.0]


# ===========================================================================
# Tests — metrics.five_hour_burn_since_reset
# ===========================================================================

class TestFiveHourBurnSinceReset:
    """Fixture: 4 rows in two 4.5-hour windows separated by a 5 h 1 min gap."""

    # Window A: 08:00 (1000) and 08:30 (500)  → cumulative 1000, 1500
    # Window B: 13:01 (2000) and 13:31 (800)  → cumulative 2000, 2800
    # gap = 5h 1min > 4.5h → new window

    _BASE = _dt(2026, 5, 18, 8, 0)
    _ROWS = [
        _BASE,
        _BASE + timedelta(minutes=30),
        _BASE + timedelta(hours=5, minutes=1),
        _BASE + timedelta(hours=5, minutes=31),
    ]
    _VALUES = [1000.0, 500.0, 2000.0, 800.0]

    @classmethod
    def _make_df(cls):
        return pl.DataFrame(
            {
                "ts": cls._ROWS,
                "value": cls._VALUES,
                "is_subagent": [False] * 4,
            },
            schema={
                "ts": pl.Datetime("ms", "UTC"),
                "value": pl.Float64,
                "is_subagent": pl.Boolean,
            },
        )

    def test_cumulative_total_in_each_window(self):
        """Data rows must show cumulative totals 1000, 1500 (win A) and 2000, 2800 (win B)."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        ts_filter = (
            pl.Series("ts", self._ROWS)
            .cast(pl.Datetime("ms", "UTC"))
            .implode()
        )
        real = out.filter(
            pl.col("ts").is_in(ts_filter) & (pl.col("cumulative_total") != 0)
        ).sort("ts")
        assert real["cumulative_total"].to_list() == [1000.0, 1500.0, 2000.0, 2800.0]

    def test_reset_rows_inserted_at_window_starts(self):
        """Zero-valued reset rows must appear at both window-start timestamps."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        for ws in [self._BASE, self._BASE + timedelta(hours=5, minutes=1)]:
            ts_s = pl.Series("ts", [ws]).cast(pl.Datetime("ms", "UTC")).implode()
            row = out.filter(
                pl.col("ts").is_in(ts_s) & (pl.col("cumulative_total") == 0.0)
            )
            assert row.height >= 1, f"Expected zero-reset row at {ws}"

    def test_end_drop_rows_inserted_after_each_window(self):
        """Zero-valued end-drop rows must appear at window_start + gap_hours."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        # win A end = 08:00 + 4.5h = 12:30
        # win B end = 13:01 + 4.5h = 17:31
        for end_ts in [
            self._BASE + timedelta(hours=4, minutes=30),
            self._BASE + timedelta(hours=5, minutes=1) + timedelta(hours=4, minutes=30),
        ]:
            ts_s = pl.Series("ts", [end_ts]).cast(pl.Datetime("ms", "UTC")).implode()
            row = out.filter(
                pl.col("ts").is_in(ts_s) & (pl.col("cumulative_total") == 0.0)
            )
            assert row.height >= 1, f"Expected zero end-drop row at {end_ts}"

    def test_total_output_row_count(self):
        """4 data rows + 2 reset rows + 2 end-drop rows = 8 rows total."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        assert out.height == 8

    def test_no_mask_cumulative_selected_equals_total(self):
        """With no selected_mask_col, cumulative_selected must equal cumulative_total."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        ts_filter = (
            pl.Series("ts", self._ROWS)
            .cast(pl.Datetime("ms", "UTC"))
            .implode()
        )
        real = out.filter(
            pl.col("ts").is_in(ts_filter) & (pl.col("cumulative_total") != 0)
        ).sort("ts")
        assert real["cumulative_selected"].to_list() == real["cumulative_total"].to_list()

    def test_window_b_cumulative_values_not_continuation_of_a(self):
        """Window B must restart at 0, not continue from the 1500 total of window A."""
        out = metrics.five_hour_burn_since_reset(
            self._make_df(), gap_hours=4.5, value_col="value"
        )
        win_b_ts = self._ROWS[2]
        ts_s = pl.Series("ts", [win_b_ts]).cast(pl.Datetime("ms", "UTC")).implode()
        row = out.filter(
            pl.col("ts").is_in(ts_s) & (pl.col("cumulative_total") != 0)
        )
        assert row["cumulative_total"].item() == 2000.0  # NOT 3500 (1500 + 2000)
