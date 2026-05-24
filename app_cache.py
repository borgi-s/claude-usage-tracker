"""Cached, filter-independent calibration compute shared by both apps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl
import streamlit as st

import caps as caps_mod
import config
import metrics


def _sig(df: pl.DataFrame, log: pl.DataFrame) -> tuple:
    """Cheap fingerprint: changes whenever the underlying data changes."""
    def part(d: pl.DataFrame, tcol: str) -> tuple:
        if d.is_empty() or tcol not in d.columns:
            return (0, None)
        return (d.height, str(d[tcol].max()))
    return part(df, "ts") + part(log, "sampled_at")


def _sig1(df: pl.DataFrame) -> tuple:
    """Cheap fingerprint for a single DataFrame keyed on its ts column."""
    if df.is_empty() or "ts" not in df.columns:
        return (0, None)
    return (df.height, str(df["ts"].max()))


@dataclass
class Calibration:
    eff_hours: float
    n_observed: int
    cap_5h: float | None
    n_anchor_5h: int
    cap_weekly: float | None
    n_anchor_weekly: int
    sessions: pl.DataFrame
    diag: dict


@st.cache_data(show_spinner=False)
def _calibrate(_sig_key: tuple, _df: pl.DataFrame, _log: pl.DataFrame) -> Calibration:
    eff_hours, n_obs = metrics.effective_window_hours(
        _log, _df, default=config.FIVE_HOUR_WINDOW_HOURS, min_samples=5,
    )
    cap5, n5 = caps_mod.global_cap_from_anchors(_log, _df, "5h", gap_hours=eff_hours)
    capw, nw = caps_mod.global_cap_from_anchors(
        _log, _df, "weekly", gap_hours=24 * 7, min_util=0.10,
    )
    sessions, diag = metrics.session_cost_attribution(_df, _log)
    return Calibration(eff_hours, n_obs, cap5, n5, capw, nw, sessions, diag)


def calibrate(df: pl.DataFrame, log: pl.DataFrame) -> Calibration:
    """Filter-independent calibration, cached on a cheap data signature."""
    return _calibrate(_sig(df, log), df, log)


# ---------------------------------------------------------------------------
# Filter-dependent compute
# ---------------------------------------------------------------------------

@dataclass
class FilteredCompute:
    sessions: pl.DataFrame          # metrics.session_summaries(fdf)
    five_h: pl.DataFrame
    weekly: pl.DataFrame
    five_h_window_shares: list      # list[float]
    per_week_shares: pl.DataFrame   # columns week_start, week_share
    daily: pl.DataFrame


@st.cache_data(show_spinner=False)
def _filtered(
    _key: tuple,
    _df_with_caps: pl.DataFrame,
    _fdf: pl.DataFrame,
    eff_hours: float,
) -> FilteredCompute:
    sessions = metrics.session_summaries(_fdf)
    five_h = metrics.five_hour_burn_since_reset(
        _df_with_caps, gap_hours=eff_hours,
        value_col="share_5h", selected_mask_col="is_selected",
    )
    weekly = metrics.weekly_burn_since_reset(
        _df_with_caps, value_col="share_week", selected_mask_col="is_selected",
    )
    window_shares = metrics.five_hour_window_totals(
        _df_with_caps, gap_hours=eff_hours, value_col="share_5h",
    )
    per_week = (
        _df_with_caps.with_columns(
            pl.col("ts")
            .map_elements(metrics.week_start_for, return_dtype=pl.Datetime("us", "UTC"))
            .cast(pl.Datetime("ms", "UTC"))
            .alias("week_start")
        )
        .group_by("week_start")
        .agg(pl.col("share_week").sum().alias("week_share"))
    )
    daily = metrics.daily_stacked(_fdf)
    return FilteredCompute(sessions, five_h, weekly, window_shares, per_week, daily)


def filtered_compute(
    df_with_caps: pl.DataFrame,
    fdf: pl.DataFrame,
    selected_projects: Sequence[str],
    selected_models: Sequence[str],
    eff_hours: float,
    cap_5h: float | None,
    cap_weekly: float | None,
) -> FilteredCompute:
    """Filter-dependent chart computations, cached on all real dependencies."""
    key = (
        _sig1(df_with_caps),
        tuple(sorted(selected_projects)),
        tuple(sorted(selected_models)),
        round(float(eff_hours), 6),
        None if cap_5h is None else round(float(cap_5h), 3),
        None if cap_weekly is None else round(float(cap_weekly), 3),
    )
    return _filtered(key, df_with_caps, fdf, eff_hours)
