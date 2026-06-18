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
# Account-wide files (caps + calibration) come from ONE canonical prefix — the
# always-on poller. The merged cache.parquet is built from all machine prefixes.
CAPS_FILE_NAMES = ["caps.json", "calibration_log.parquet"]


def _prefixes() -> list[str]:
    """Machine prefixes to merge. CLOUD_USER_PREFIXES (comma-separated) wins;
    falls back to the legacy single CLOUD_USER_PREFIX ('' = bucket root)."""
    multi = os.environ.get("CLOUD_USER_PREFIXES", "").strip()
    if multi:
        return [p.strip().strip("/") for p in multi.split(",") if p.strip()]
    return [os.environ.get("CLOUD_USER_PREFIX", "").strip("/")]


PREFIXES = _prefixes()
# Caps/calibration are account-wide (same on every machine), so read them from one
# prefix — the always-on poller. Defaults to the first prefix if unset.
CAPS_PREFIX = os.environ.get("CLOUD_CAPS_PREFIX", PREFIXES[0]).strip("/")


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
        # Freshness across ALL machines: re-download only when any prefix's
        # cache.parquet changed. Compute each prefix's mtime once; reuse for both
        # the change-key and the "newest activity" age shown in the live panel.
        mtimes = {
            p: supabase_sync.last_modified_at(client, bucket, "cache.parquet", prefix=p)
            for p in PREFIXES
        }
        mtime_key = "|".join(
            f"{p}:{mt.isoformat() if mt else 'none'}" for p, mt in mtimes.items()
        )
        if mtime_key != st.session_state.get("last_cache_mtime"):
            # 1. Download each machine's cache.parquet to a distinct local file and
            #    merge into config.CACHE_PATH (with a `machine` column per row).
            prefix_paths = supabase_sync.download_cache_per_prefix(
                client, bucket, PREFIXES, target_dir=DATA_DIR
            )
            cache.merge_cache_parquets(prefix_paths, config.CACHE_PATH)
            # 2. caps.json + calibration_log are account-wide — take them from the
            #    canonical (poller) prefix only. Best-effort: a missing caps file
            #    must not block the merged cache view (the viewer has fallback caps).
            try:
                supabase_sync.download_files(
                    client, bucket, CAPS_FILE_NAMES, target_dir=DATA_DIR, prefix=CAPS_PREFIX
                )
            except Exception as caps_err:
                st.warning(f"Caps/calibration unavailable from '{CAPS_PREFIX}': {caps_err}")
            st.session_state["last_cache_mtime"] = mtime_key
            load_data.clear()
        # Live-panel age = newest activity across all machines.
        newest = max((mt for mt in mtimes.values() if mt is not None), default=None)
        seconds_old = (
            (datetime.now(tz=timezone.utc) - newest).total_seconds() if newest else None
        )
        render.render_live_panel_from_cache(
            agent_seconds_old=seconds_old, log=calibration_log.load_log())
    except Exception as e:
        st.error(f"Could not fetch latest from Supabase: {e}")


@st.cache_resource(show_spinner="Loading data...")
def load_data():
    df = cache.load_cache()
    return metrics.add_derived(df)


st.title("Claude usage tracker")
st.caption("Read-only cloud view · refreshes every 5 min · data flows from your machines → Supabase → here.")

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


log = calibration_log.load_log()

total_usd = float(df["dollar_cost"].sum())
span_days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0, 1.0)
daily_avg_usd = total_usd / span_days
peak_5h = metrics.peak_reported(log, "5h")
peak_weekly = metrics.peak_reported(log, "weekly")
w5_over, w5_total = metrics.windows_over_threshold(log, "5h", 0.20)
wk_over, wk_total = metrics.windows_over_threshold(log, "weekly", 0.20)

render.render_kpis(total_usd, daily_avg_usd, peak_5h, peak_weekly,
                   w5_over, w5_total, wk_over, wk_total)
render.render_reported_usage_chart(log, "5h", data_start_ts, data_end_ts)
render.render_reported_usage_chart(log, "weekly", data_start_ts, data_end_ts)
render.render_daily_bar(fdf, decomposition_key="cloud")

fc = app_cache.filtered_compute(fdf)
sessions = fc.sessions

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
