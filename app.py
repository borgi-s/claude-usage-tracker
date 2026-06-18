"""Streamlit dashboard for Claude Code token usage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import streamlit as st

import anthropic_client
import app_cache
import cache
import calibration_log
import caps as caps_mod
import config
import metrics
import render
import supabase_sync


st.set_page_config(page_title="Claude usage tracker", layout="wide")


@st.cache_resource(show_spinner="Loading cache...")
def load_data():
    df = cache.load_cache()
    return metrics.add_derived(df)


@st.cache_resource
def _supabase_client():
    """Build the Windows-agent Supabase client once. Returns (client, bucket) or None."""
    try:
        return supabase_sync.from_env()
    except Exception:
        return None


def refresh_and_reload():
    cache.refresh_cache()
    load_data.clear()


st.title("Claude usage tracker")
st.caption(config.CAP_DISCLAIMER)


def _render_usage_view(
    *,
    util_5h: float | None,
    resets_5h: datetime | None,
    util_7d: float | None,
    resets_7d: datetime | None,
    log,
    sampled_at: datetime,
    sub_type: str,
    rate_limit_tier: str,
    stale: bool,
):
    now = datetime.now(tz=timezone.utc)
    cols = st.columns(4)
    if util_5h is not None:
        cols[0].metric("5h utilization", f"{util_5h*100:.0f}%")
        cols[0].progress(min(1.0, util_5h))
        if resets_5h:
            mins = max(0, (resets_5h - now).total_seconds() / 60)
            cols[1].metric("5h resets in", f"{int(mins//60)}h {int(mins%60)}m")
    if util_7d is not None:
        cols[2].metric("7d utilization", f"{util_7d*100:.0f}%")
        cols[2].progress(min(1.0, util_7d))
        if resets_7d:
            hours = max(0, (resets_7d - now).total_seconds() / 3600)
            cols[3].metric("7d resets in", f"{int(hours//24)}d {int(hours%24)}h")

    if util_5h is not None:
        proj = metrics.project_time_to_cap(log, now, "5h")
        cols[1].caption("5h → 100%: " + render.format_projection(proj))
    age_s = (now - sampled_at).total_seconds()
    age_str = (f"{int(age_s)}s ago" if age_s < 90
               else f"{int(age_s/60)}m ago" if age_s < 5400 else f"{age_s/3600:.1f}h ago")
    prefix = f"Updated {age_str}" + (" · stale (endpoint rate-limited)" if stale else "")
    st.caption(prefix + f" · sub `{sub_type}` tier `{rate_limit_tier}`")


@st.fragment(run_every=300)
def live_usage_panel():
    st.subheader("Live plan usage")
    # Keep the chart data fresh: incremental disk reparse + force full app
    # rerun if anything new landed on disk.
    _, refresh_stats = cache.refresh_cache()
    if refresh_stats["new_or_changed"] > 0:
        load_data.clear()
        st.rerun(scope="app")
    try:
        snap = anthropic_client.fetch_usage()
    except anthropic_client.RateLimited:
        prev = caps_mod.load_caps()
        if not prev.sampled_at:
            st.warning("Usage endpoint rate-limited. No cached calibration yet — will retry next tick.")
            return
        sampled_at = datetime.fromisoformat(prev.sampled_at)
        if sampled_at.tzinfo is None:
            sampled_at = sampled_at.replace(tzinfo=timezone.utc)
        resets_5h = datetime.fromisoformat(prev.resets_5h_iso) if prev.resets_5h_iso else None
        resets_7d = datetime.fromisoformat(prev.resets_7d_iso) if prev.resets_7d_iso else None
        _render_usage_view(
            util_5h=prev.sample_util_5h,
            resets_5h=resets_5h,
            util_7d=prev.sample_util_7d,
            resets_7d=resets_7d,
            log=calibration_log.load_log(),
            sampled_at=sampled_at,
            sub_type=prev.subscription_type or "unknown",
            rate_limit_tier=prev.rate_limit_tier or "unknown",
            stale=True,
        )
        return
    except Exception as e:
        st.warning(f"Could not reach usage endpoint: {e}")
        return

    # Window-aligned burns: use the actual Anthropic window the utilization refers to.
    df_all = load_data()
    window_start_5h = (snap.five_hour.resets_at - timedelta(hours=config.FIVE_HOUR_WINDOW_HOURS)
                       if snap.five_hour and snap.five_hour.resets_at else None)
    window_start_weekly = (snap.seven_day.resets_at - timedelta(days=7)
                           if snap.seven_day and snap.seven_day.resets_at else None)
    burn_5h = calibration_log.cost_weighted_sum_in_window(df_all, window_start_5h, snap.sampled_at)
    burn_weekly = calibration_log.cost_weighted_sum_in_window(df_all, window_start_weekly, snap.sampled_at)

    # Append fresh sample first, then aggregate
    agg_5h = calibration_log.window_aggregates(df_all, window_start_5h, snap.sampled_at)
    agg_weekly = calibration_log.window_aggregates(df_all, window_start_weekly, snap.sampled_at)
    calibration_log.append_sample({
        "sampled_at": snap.sampled_at,
        "util_5h": snap.five_hour.utilization if snap.five_hour else None,
        "util_7d": snap.seven_day.utilization if snap.seven_day else None,
        "burn_5h_cost_weighted": burn_5h,
        "burn_7d_cost_weighted": burn_weekly,
        "input_5h": agg_5h["input"],
        "cache_creation_5h": agg_5h["cache_creation"],
        "cache_read_5h": agg_5h["cache_read"],
        "output_5h": agg_5h["output"],
        "input_7d": agg_weekly["input"],
        "cache_creation_7d": agg_weekly["cache_creation"],
        "cache_read_7d": agg_weekly["cache_read"],
        "output_7d": agg_weekly["output"],
        "subscription_type": snap.subscription_type,
        "rate_limit_tier": snap.rate_limit_tier,
        "resets_5h_iso": snap.five_hour.resets_at.isoformat() if snap.five_hour and snap.five_hour.resets_at else None,
        "resets_7d_iso": snap.seven_day.resets_at.isoformat() if snap.seven_day and snap.seven_day.resets_at else None,
    })

    snapshot = caps_mod.DerivedCaps(
        sampled_at=snap.sampled_at.isoformat(),
        sample_util_5h=snap.five_hour.utilization if snap.five_hour else None,
        sample_util_7d=snap.seven_day.utilization if snap.seven_day else None,
        subscription_type=snap.subscription_type,
        resets_5h_iso=snap.five_hour.resets_at.isoformat() if snap.five_hour and snap.five_hour.resets_at else None,
        resets_7d_iso=snap.seven_day.resets_at.isoformat() if snap.seven_day and snap.seven_day.resets_at else None,
        rate_limit_tier=snap.rate_limit_tier,
    )
    caps_mod.save_caps(snapshot)

    sb = _supabase_client()
    if sb is not None:
        client, bucket = sb
        try:
            supabase_sync.upload_files(client, bucket, [
                config.CACHE_PATH,
                caps_mod.CAPS_PATH,
                calibration_log.LOG_PATH,
            ])
        except Exception as e:
            st.caption(f":warning: Supabase sync failed: {e}")

    _render_usage_view(
        util_5h=snap.five_hour.utilization if snap.five_hour else None,
        resets_5h=snap.five_hour.resets_at if snap.five_hour else None,
        util_7d=snap.seven_day.utilization if snap.seven_day else None,
        resets_7d=snap.seven_day.resets_at if snap.seven_day else None,
        log=calibration_log.load_log(),
        sampled_at=snap.sampled_at,
        sub_type=snap.subscription_type,
        rate_limit_tier=snap.rate_limit_tier,
        stale=False,
    )


live_usage_panel()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    if st.button("Refresh from disk", width="stretch"):
        refresh_and_reload()
        st.rerun()

    df = load_data()
    if df.is_empty():
        st.info("Cache empty. Click Refresh.")
        st.stop()

    # cwd → short label for the picker
    cwd_to_short = {c: render.short_project(c) for c in df["project_cwd"].unique().to_list() if c}
    short_to_cwd: dict[str, list[str]] = {}
    for c, s in cwd_to_short.items():
        short_to_cwd.setdefault(s, []).append(c)
    project_options = sorted(short_to_cwd.keys())
    selected_projects = st.multiselect("Project", project_options, default=project_options)

    model_options = sorted([m for m in df["model"].unique().to_list() if m])
    default_models = [m for m in model_options if m != "<synthetic>"]
    selected_models = st.multiselect("Model", model_options, default=default_models)

    sort_choice = st.selectbox(
        "Sort sessions by",
        ["peak context %", "total cost-weighted", "chronological"],
        index=2,
    )

    st.divider()
    st.subheader("Session table filter")
    min_turns = st.number_input("Min main turns", min_value=1, max_value=100, value=5, step=1)
    min_duration_s = st.number_input("Min duration (seconds)", min_value=0, max_value=3600, value=60, step=10)


# ---------- Filtering ----------
allowed_cwds = []
for s in selected_projects:
    allowed_cwds.extend(short_to_cwd.get(s, []))

# Per-row selected mask (project + model). Full df flows through cumulative
# computation; mask controls what's emphasized in the "project share" view.
df_with_mask = df.with_columns(
    (pl.col("project_cwd").is_in(allowed_cwds) & pl.col("model").is_in(selected_models))
    .alias("is_selected")
)

# fdf is the date-unfiltered, project+model-filtered subset for daily bar + session table.
fdf = df_with_mask.filter(pl.col("is_selected"))

if fdf.is_empty():
    st.warning("No data matches the current filters.")
    st.stop()

# Bounds for calendar-band shading — span the full df, not the filtered subset.
data_start_ts = df["ts"].min()
data_end_ts = df["ts"].max()


log = calibration_log.load_log()

# ---- KPIs ----
total_usd = float(df["dollar_cost"].sum())
span_days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0, 1.0)
daily_avg_usd = total_usd / span_days
peak_5h = metrics.peak_reported(log, "5h")
peak_weekly = metrics.peak_reported(log, "weekly")
w5_over, w5_total = metrics.windows_over_threshold(log, "5h", 0.20)
wk_over, wk_total = metrics.windows_over_threshold(log, "weekly", 0.20)
render.render_kpis(total_usd, daily_avg_usd, peak_5h, peak_weekly,
                   w5_over, w5_total, wk_over, wk_total)

# ---- Charts ----
render.render_reported_usage_chart(log, "5h", data_start_ts, data_end_ts)
render.render_reported_usage_chart(log, "weekly", data_start_ts, data_end_ts)
render.render_daily_bar(fdf, decomposition_key="app")

# ---- Session table ----
fc = app_cache.filtered_compute(fdf)
sessions = fc.sessions


# ---------- Session aggregation for summary table ----------
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


# ---------- Summary table ----------
render.render_sessions_table(sessions_sorted, hidden, min_turns, min_duration_s)
