"""Streamlit dashboard for Claude Code token usage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import streamlit as st

import anthropic_client
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
    derived: caps_mod.DerivedCaps,
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

    bits = []
    if derived.max5x_5h:
        bits.append(f"Max 5x 5h cap ≈ {derived.max5x_5h/1e6:.0f}M (Pro {derived.pro_5h/1e6:.0f}M)")
    if derived.max5x_weekly:
        bits.append(f"Max 5x weekly cap ≈ {derived.max5x_weekly/1e6:.0f}M (Pro {derived.pro_weekly/1e6:.0f}M)")

    age_s = (now - sampled_at).total_seconds()
    if age_s < 90:
        age_str = f"{int(age_s)}s ago"
    elif age_s < 5400:
        age_str = f"{int(age_s/60)}m ago"
    else:
        age_str = f"{age_s/3600:.1f}h ago"
    prefix = f"Updated {age_str}" + (" · stale (endpoint rate-limited)" if stale else "")

    if bits:
        st.caption(prefix + " · " + " · ".join(bits) + f" · sub `{sub_type}` tier `{rate_limit_tier}`")
    else:
        st.caption(prefix + f" · sub `{sub_type}` tier `{rate_limit_tier}`  ·  utilization too low to derive caps yet.")


@st.fragment(run_every=60)
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
            derived=prev,
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
    # Use effective window length if we have enough observed resets; else config default.
    _eff_5h, _ = metrics.effective_window_hours(
        calibration_log.load_log(), df_all,
        default=config.FIVE_HOUR_WINDOW_HOURS, min_samples=5,
    )
    window_start_5h = (
        snap.five_hour.resets_at - timedelta(hours=_eff_5h)
        if snap.five_hour and snap.five_hour.resets_at else None
    )
    window_start_weekly = (
        snap.seven_day.resets_at - timedelta(days=7)
        if snap.seven_day and snap.seven_day.resets_at else None
    )
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

    # Continuous calibration: median of recent valid samples
    snap_meta = {
        "sampled_at": snap.sampled_at.isoformat(),
        "sample_burn_5h": burn_5h,
        "sample_burn_7d": burn_weekly,
        "sample_util_5h": snap.five_hour.utilization if snap.five_hour else None,
        "sample_util_7d": snap.seven_day.utilization if snap.seven_day else None,
        "subscription_type": snap.subscription_type,
        "rate_limit_tier": snap.rate_limit_tier,
        "resets_5h_iso": snap.five_hour.resets_at.isoformat() if snap.five_hour and snap.five_hour.resets_at else None,
        "resets_7d_iso": snap.seven_day.resets_at.isoformat() if snap.seven_day and snap.seven_day.resets_at else None,
    }
    log_now = calibration_log.load_log()
    derived = caps_mod.derive_continuous_caps(log_now, snap_metadata=snap_meta)
    # If continuous didn't yield caps yet (few valid samples), fall back to single-sample
    if derived.max5x_5h is None or derived.max5x_weekly is None:
        single = caps_mod.derive_from_reading(
            burn_5h=burn_5h, util_5h=snap_meta["sample_util_5h"],
            burn_7d=burn_weekly, util_7d=snap_meta["sample_util_7d"],
            subscription_type=snap.subscription_type,
            resets_5h_iso=snap_meta["resets_5h_iso"],
            resets_7d_iso=snap_meta["resets_7d_iso"],
            rate_limit_tier=snap.rate_limit_tier,
        )
        # Prefer single-sample value where continuous was None
        if derived.max5x_5h is None and single.max5x_5h:
            derived.max5x_5h = single.max5x_5h
            derived.pro_5h = single.pro_5h
        if derived.max5x_weekly is None and single.max5x_weekly:
            derived.max5x_weekly = single.max5x_weekly
            derived.pro_weekly = single.pro_weekly
    caps_mod.save_caps(derived)

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
        derived=derived,
        sampled_at=snap.sampled_at,
        sub_type=snap.subscription_type,
        rate_limit_tier=snap.rate_limit_tier,
        stale=False,
    )


live_usage_panel()


render.render_calibration_history(load_data())


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

    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    default_start = max(min_ts.date(), (max_ts - timedelta(days=14)).date())
    date_range = st.date_input(
        "Date range",
        value=(default_start, max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

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
    st.subheader("Plan caps (cost-weighted tokens)")
    eff_pro_5h, eff_pro_week, caps_source = caps_mod.effective_caps()
    st.caption(f"Source: {caps_source}")
    pro_5h_raw = st.number_input("Pro 5h cap", value=int(eff_pro_5h), step=1_000_000)
    pro_week_raw = st.number_input("Pro weekly cap", value=int(eff_pro_week), step=10_000_000)
    show_max5x = st.checkbox("Show Max 5x line too", value=True)
    pro_5h = pro_5h_raw
    pro_week = pro_week_raw

    st.divider()
    st.subheader("Session table filter")
    min_turns = st.number_input("Min main turns", min_value=1, max_value=100, value=5, step=1)
    min_duration_s = st.number_input("Min duration (seconds)", min_value=0, max_value=3600, value=60, step=10)


# ---------- Filtering ----------
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


# ---------- Single global cap derived from 100% anchors ----------
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

# Share columns: each row's contribution to the cumulative % of cap
fdf_with_caps = fdf.with_columns(
    (pl.col("cost_weighted_tokens") / effective_cap_5h).alias("share_5h"),
    (pl.col("cost_weighted_tokens") / effective_cap_week).alias("share_week"),
)


# ---------- KPIs ----------
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

# Per-window totals in SHARES, compared against Pro = 0.20 of Max5x
five_h_window_shares = metrics.five_hour_window_totals(
    fdf_with_caps, gap_hours=effective_5h_hours, value_col="share_5h",
)
windows_over_pro_5h = sum(1 for s in five_h_window_shares if s > 0.20)
windows_total_5h = len(five_h_window_shares)

per_week_shares = (
    fdf_with_caps.with_columns(
        pl.col("ts")
        .map_elements(metrics.week_start_for, return_dtype=pl.Datetime("us", "UTC"))
        .cast(pl.Datetime("ms", "UTC"))
        .alias("week_start")
    )
    .group_by("week_start")
    .agg(pl.col("share_week").sum().alias("week_share"))
)
weeks_over_pro = per_week_shares.filter(pl.col("week_share") > 0.20).height
weeks_total = per_week_shares.height

render.render_kpis(total_cw, daily_avg, peak_5h_share, peak_weekly_share,
                   windows_over_pro_5h, windows_total_5h, weeks_over_pro, weeks_total)


# ---------- Chart 1: 5h fixed window ----------
render.render_5h_chart(five_h, effective_5h_hours, n_observed_resets, n_anchor_5h,
                       effective_cap_5h, show_max5x, start_ts, end_ts)


# ---------- Chart 2: Weekly cumulative (fixed reset) ----------
render.render_weekly_chart(weekly, n_anchor_week, effective_cap_week,
                           show_max5x, start_ts, end_ts)


# ---------- Chart 3: Daily stacked ----------
daily = metrics.daily_stacked(fdf)
render.render_daily_bar(daily)


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
