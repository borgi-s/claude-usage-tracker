"""Cached, filter-independent calibration compute shared by both apps."""
from __future__ import annotations

from dataclasses import dataclass

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
