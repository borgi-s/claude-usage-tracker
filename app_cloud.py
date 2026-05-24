"""Read-only cloud viewer for the Claude usage tracker.

Deployed on Streamlit Community Cloud. Downloads data files from Supabase
Storage every 5 minutes and renders the same charts as the local app, minus
calibration and refresh controls.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import streamlit as st

import app_cache
import cache
import calibration_log
import caps as caps_mod
import config
import metrics
import render
import supabase_sync


st.set_page_config(page_title="Claude usage tracker — cloud", layout="wide")

DATA_DIR = Path(os.environ.get("CLOUD_DATA_DIR", "/tmp/usage-tracker"))
FILE_NAMES = ["cache.parquet", "caps.json", "calibration_log.parquet"]
# Per-user folder in the bucket (the Rust agent's SUPABASE_USER_PREFIX). Empty =
# read from bucket root, matching the legacy Python agent's upload location.
USER_PREFIX = os.environ.get("CLOUD_USER_PREFIX", "").strip("/")


@st.cache_resource
def _client():
    try:
        return supabase_sync.from_streamlit_secrets()
    except Exception as e:
        st.error(f"Cloud Supabase client not configured: {e}")
        st.stop()


def _redirect_paths_to_data_dir():
    """Point config-module paths at DATA_DIR so cache.load_cache() finds downloads."""
    config.CACHE_PATH = DATA_DIR / "cache.parquet"
    config.MANIFEST_PATH = DATA_DIR / "cache_manifest.json"
    caps_mod.CAPS_PATH = DATA_DIR / "caps.json"
    calibration_log.LOG_PATH = DATA_DIR / "calibration_log.parquet"


_redirect_paths_to_data_dir()


@st.fragment(run_every=300)
def refresh_data_panel():
    client, bucket = _client()
    try:
        mtime = supabase_sync.last_modified_at(client, bucket, "cache.parquet", prefix=USER_PREFIX)
        mtime_key = mtime.isoformat() if mtime else None
        # Only re-download (and invalidate the chart cache) when the agent has
        # actually written new data. Skipping the download on no-op polls keeps
        # the fragment's stale-fade brief.
        if mtime_key != st.session_state.get("last_cache_mtime"):
            supabase_sync.download_files(client, bucket, FILE_NAMES, target_dir=DATA_DIR, prefix=USER_PREFIX)
            st.session_state["last_cache_mtime"] = mtime_key
            load_data.clear()
        seconds_old = None
        if mtime is not None:
            seconds_old = (datetime.now(tz=timezone.utc) - mtime).total_seconds()
        render.render_live_panel_from_cache(agent_seconds_old=seconds_old)
    except Exception as e:
        st.error(f"Could not fetch latest from Supabase: {e}")


@st.cache_resource(show_spinner="Loading data...")
def load_data():
    df = cache.load_cache()
    return metrics.add_derived(df)


st.title("Claude usage tracker")
st.caption("Read-only cloud view · refreshes every 5 min · data flows from your Windows agent → Supabase → here.")

refresh_data_panel()

df = load_data()
if df.is_empty():
    st.warning("No data yet. Make sure the Windows agent is running and has uploaded at least once.")
    st.stop()


with st.sidebar:
    st.header("Filters")

    cwd_to_short = {c: render.short_project(c) for c in df["project_cwd"].unique().to_list() if c}
    short_to_cwd: dict[str, list[str]] = {}
    for c, s in cwd_to_short.items():
        short_to_cwd.setdefault(s, []).append(c)
    project_options = sorted(short_to_cwd.keys())
    selected_projects = st.multiselect("Project", project_options, default=project_options)

    model_options = sorted([m for m in df["model"].unique().to_list() if m])
    default_models = [m for m in model_options if m != "<synthetic>"]
    selected_models = st.multiselect("Model", model_options, default=default_models)

    sort_choice = st.selectbox("Sort sessions by",
                               ["peak context %", "total cost-weighted", "chronological"],
                               index=2)

    st.divider()
    st.subheader("Plan caps (read-only)")
    st.caption("Calibrated against output tokens — Anthropic's actual rate-limit signal.")
    show_max5x = st.checkbox("Show Max 5x line too", value=True)

    st.divider()
    st.subheader("Session table filter")
    min_turns = st.number_input("Min main turns", min_value=1, max_value=100, value=5, step=1)
    min_duration_s = st.number_input("Min duration (seconds)", min_value=0, max_value=3600, value=60, step=10)


allowed_cwds = []
for s in selected_projects:
    allowed_cwds.extend(short_to_cwd.get(s, []))

df_with_mask = df.with_columns(
    (pl.col("project_cwd").is_in(allowed_cwds) & pl.col("model").is_in(selected_models))
    .alias("is_selected")
)
fdf = df_with_mask.filter(pl.col("is_selected"))

if fdf.is_empty():
    st.warning("No data matches the current filters.")
    st.stop()

data_start_ts = df["ts"].min()
data_end_ts = df["ts"].max()


calib_log_global = calibration_log.load_log()
calib = app_cache.calibrate(df, calib_log_global)
effective_5h_hours = calib.eff_hours
n_observed_resets = calib.n_observed

OUTPUT_CAP_5H_FALLBACK = 2_100_000.0
OUTPUT_CAP_WEEKLY_FALLBACK = 100_000_000.0
global_cap_5h = calib.cap_5h
n_anchor_5h = calib.n_anchor_5h
global_cap_week = calib.cap_weekly
n_anchor_week = calib.n_anchor_weekly
effective_cap_5h = global_cap_5h if global_cap_5h else OUTPUT_CAP_5H_FALLBACK
effective_cap_week = global_cap_week if global_cap_week else OUTPUT_CAP_WEEKLY_FALLBACK

df_with_caps = df_with_mask.with_columns(
    (pl.col("output_tokens") / effective_cap_5h).alias("share_5h"),
    (pl.col("output_tokens") / effective_cap_week).alias("share_week"),
)

total_cw = float(df["cost_weighted_tokens"].sum())
sessions = metrics.session_summaries(fdf)
five_h = metrics.five_hour_burn_since_reset(
    df_with_caps, gap_hours=effective_5h_hours,
    value_col="share_5h", selected_mask_col="is_selected",
)
weekly = metrics.weekly_burn_since_reset(
    df_with_caps, value_col="share_week", selected_mask_col="is_selected",
)
peak_5h_share = float(five_h["cumulative_total"].max() or 0)
peak_weekly_share = float(weekly["cumulative_total"].max() or 0)

span_days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0, 1.0)
daily_avg = total_cw / span_days

five_h_window_shares = metrics.five_hour_window_totals(
    df_with_caps, gap_hours=effective_5h_hours, value_col="share_5h",
)
windows_over_pro_5h = sum(1 for s in five_h_window_shares if s > 0.20)
windows_total_5h = len(five_h_window_shares)

per_week_shares = (
    df_with_caps.with_columns(
        pl.col("ts").map_elements(metrics.week_start_for, return_dtype=pl.Datetime("us", "UTC"))
          .cast(pl.Datetime("ms", "UTC")).alias("week_start")
    )
    .group_by("week_start")
    .agg(pl.col("share_week").sum().alias("week_share"))
)
weeks_over_pro = per_week_shares.filter(pl.col("week_share") > 0.20).height
weeks_total = per_week_shares.height


render.render_kpis(total_cw, daily_avg, peak_5h_share, peak_weekly_share,
                   windows_over_pro_5h, windows_total_5h, weeks_over_pro, weeks_total)
render.render_5h_chart(
    five_h, effective_5h_hours, n_observed_resets, n_anchor_5h,
    effective_cap_5h, show_max5x, data_start_ts, data_end_ts,
    decomposition_key="cloud",
)
render.render_weekly_chart(
    weekly, n_anchor_week, effective_cap_week,
    show_max5x, data_start_ts, data_end_ts,
    decomposition_key="cloud",
)
daily = metrics.daily_stacked(fdf)
render.render_daily_bar(daily)
render.render_cost_vs_session_length(df, calib_log_global)

sessions_with_dur = sessions.with_columns(
    ((pl.col("end") - pl.col("start")).dt.total_milliseconds() / 1000.0).alias("duration_s")
)
total_before_filter = sessions_with_dur.height
sessions_filtered = sessions_with_dur.filter(
    (pl.col("main_turns") >= min_turns) & (pl.col("duration_s") >= min_duration_s)
)
hidden = total_before_filter - sessions_filtered.height
if sort_choice == "peak context %":
    sessions_sorted = sessions_filtered.sort("peak_context_pct", descending=True)
elif sort_choice == "total cost-weighted":
    sessions_sorted = sessions_filtered.sort("total_cost_weighted", descending=True)
else:
    sessions_sorted = sessions_filtered.sort("start")
render.render_sessions_table(sessions_sorted, hidden, min_turns, min_duration_s)

render.render_calibration_history(df)
