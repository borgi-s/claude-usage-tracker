"""Read-only cloud viewer for the Claude usage tracker.

Deployed on Streamlit Community Cloud. Downloads data files from Supabase
Storage every 60 seconds and renders the same charts as the local app, minus
calibration and refresh controls.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import streamlit as st

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


@st.fragment(run_every=60)
def refresh_data_panel():
    client, bucket = _client()
    try:
        supabase_sync.download_files(client, bucket, FILE_NAMES, target_dir=DATA_DIR)
        mtime = supabase_sync.last_modified_at(client, bucket, "cache.parquet")
        seconds_old = None
        if mtime is not None:
            seconds_old = (datetime.now(tz=timezone.utc) - mtime).total_seconds()
        render.render_live_panel_from_cache(agent_seconds_old=seconds_old)
        load_data.clear()
    except Exception as e:
        st.error(f"Could not fetch latest from Supabase: {e}")


@st.cache_resource(show_spinner="Loading data...")
def load_data():
    df = cache.load_cache()
    return metrics.add_derived(df)


st.title("Claude usage tracker")
st.caption("Read-only cloud view · refreshes every 60s · data flows from your Windows agent → Supabase → here.")

refresh_data_panel()

df = load_data()
if df.is_empty():
    st.warning("No data yet. Make sure the Windows agent is running and has uploaded at least once.")
    st.stop()


with st.sidebar:
    st.header("Filters")
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    default_start = max(min_ts.date(), (max_ts - timedelta(days=14)).date())
    date_range = st.date_input(
        "Date range",
        value=(default_start, max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

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
    eff_pro_5h, eff_pro_week, caps_source = caps_mod.effective_caps()
    st.caption(f"Source: {caps_source}")
    st.text(f"Pro 5h cap: {eff_pro_5h/1e6:.1f}M")
    st.text(f"Pro weekly cap: {eff_pro_week/1e6:.1f}M")
    show_max5x = st.checkbox("Show Max 5x line too", value=True)
    pro_5h_raw = eff_pro_5h
    pro_week_raw = eff_pro_week

    st.divider()
    st.subheader("Session table filter")
    min_turns = st.number_input("Min main turns", min_value=1, max_value=100, value=5, step=1)
    min_duration_s = st.number_input("Min duration (seconds)", min_value=0, max_value=3600, value=60, step=10)


allowed_cwds = []
for s in selected_projects:
    allowed_cwds.extend(short_to_cwd.get(s, []))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d = end_d = date_range  # type: ignore[assignment]

start_ts = datetime.combine(start_d, datetime.min.time(), tzinfo=timezone.utc)
end_ts = datetime.combine(end_d, datetime.max.time(), tzinfo=timezone.utc)

fdf = df.filter(
    pl.col("ts").is_between(start_ts, end_ts)
    & pl.col("project_cwd").is_in(allowed_cwds)
    & pl.col("model").is_in(selected_models)
)

if fdf.is_empty():
    st.warning("No data matches the current filters.")
    st.stop()


calib_log_global = calibration_log.load_log()
effective_5h_hours, n_observed_resets = metrics.effective_window_hours(
    calib_log_global, df, default=config.FIVE_HOUR_WINDOW_HOURS, min_samples=5,
)

max5x_5h_fallback = float(pro_5h_raw) * 5
max5x_week_fallback = float(pro_week_raw) * 5
global_cap_5h, n_anchor_5h = caps_mod.global_cap_from_anchors(
    calib_log_global, df, "5h", gap_hours=effective_5h_hours,
)
global_cap_week, n_anchor_week = caps_mod.global_cap_from_anchors(
    calib_log_global, df, "weekly", gap_hours=24 * 7,
)
effective_cap_5h = global_cap_5h if global_cap_5h else max5x_5h_fallback
effective_cap_week = global_cap_week if global_cap_week else max5x_week_fallback

fdf_with_caps = fdf.with_columns(
    (pl.col("cost_weighted_tokens") / effective_cap_5h).alias("share_5h"),
    (pl.col("cost_weighted_tokens") / effective_cap_week).alias("share_week"),
)

total_cw = float(fdf["cost_weighted_tokens"].sum())
sessions = metrics.session_summaries(fdf)
five_h = metrics.five_hour_burn_since_reset(
    fdf_with_caps, gap_hours=effective_5h_hours, value_col="share_5h",
)
weekly = metrics.weekly_burn_since_reset(fdf_with_caps, value_col="share_week")
peak_5h_share = float(five_h["cumulative_total"].max() or 0)
peak_weekly_share = float(weekly["cumulative_total"].max() or 0)
span_days = max((fdf["ts"].max() - fdf["ts"].min()).total_seconds() / 86400.0, 1.0)
daily_avg = total_cw / span_days

five_h_window_shares = metrics.five_hour_window_totals(
    fdf_with_caps, gap_hours=effective_5h_hours, value_col="share_5h",
)
windows_over_pro_5h = sum(1 for s in five_h_window_shares if s > 0.20)
windows_total_5h = len(five_h_window_shares)

per_week_shares = (
    fdf_with_caps.with_columns(
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
render.render_5h_chart(five_h, effective_5h_hours, n_observed_resets, n_anchor_5h,
                       effective_cap_5h, show_max5x, start_ts, end_ts)
render.render_weekly_chart(weekly, n_anchor_week, effective_cap_week,
                           show_max5x, start_ts, end_ts)
daily = metrics.daily_stacked(fdf)
render.render_daily_bar(daily)

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
