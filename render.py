"""Shared rendering helpers used by both app.py and app_cloud.py."""
from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import app_cache
import caps as caps_mod
import calibration_log
import config
import metrics


def _rangeselector_xaxis() -> dict:
    """Plotly xaxis config: Yahoo-style range buttons + no rangeslider."""
    return dict(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1D",  step="day",   stepmode="backward"),
                dict(count=5,  label="5D",  step="day",   stepmode="backward"),
                dict(count=14, label="14D", step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            x=0, y=1.15,
        ),
        rangeslider=dict(visible=False),
        type="date",
    )


def build_reported_figure(series: pl.DataFrame, cap_hits: pl.DataFrame, kind: str,
                          data_start_ts, data_end_ts) -> go.Figure:
    label = "5h" if kind == "5h" else "weekly"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["ts"].to_list(), y=series["util_pct"].to_list(),
        mode="lines", name=f"reported {label} util",
        line=dict(width=1.6, color="#4f8cff"), connectgaps=False,
    ))
    if not cap_hits.is_empty():
        fig.add_trace(go.Scatter(
            x=cap_hits["ts"].to_list(), y=cap_hits["util_pct"].to_list(),
            mode="markers", name="cap hit (≥99%)",
            marker=dict(size=8, color="red", symbol="circle"),
            hovertemplate="%{x}<br>%{y:.0f}%<extra>cap hit</extra>",
        ))
    fig.add_hline(y=20.0, line_dash="dash", line_color="red",
                  annotation_text="Pro cap (est. 20%)", annotation_position="top left")
    fig.add_hline(y=100.0, line_dash="dot", line_color="orange",
                  annotation_text="Max 5x cap (100%)", annotation_position="top left")
    if data_start_ts is not None and data_end_ts is not None:
        add_calendar_bands(fig, data_start_ts, data_end_ts)
    fig.update_layout(
        height=350, margin=dict(t=60, b=20, l=10, r=10),
        yaxis_title="% of Max 5x cap", yaxis_ticksuffix="%",
        yaxis=dict(autorange=True),
        xaxis=_rangeselector_xaxis(),
        legend=dict(orientation="h"),
    )
    return fig


def render_reported_usage_chart(log: pl.DataFrame, kind: str,
                                data_start_ts, data_end_ts) -> None:
    title = ("5h usage limit (reported)" if kind == "5h"
             else "Weekly usage limit (reported)")
    st.subheader(title)
    st.caption("Anthropic's reported utilization (Max 5x plan only). Account-wide — the "
               "project/model filter does not affect this chart. Line breaks at each window "
               "reset and across sampling gaps; red dots mark where you hit the cap.")
    series, cap_hits = metrics.reported_util_series(log, kind)
    if series.is_empty():
        st.info("No Max-5x reported-usage samples yet.")
        return
    fig = build_reported_figure(series, cap_hits, kind, data_start_ts, data_end_ts)
    st.plotly_chart(fig, width="stretch")


def short_project(cwd: str) -> str:
    if not cwd:
        return "(unknown)"
    return os.path.basename(cwd.rstrip("/\\")) or cwd


def add_calendar_bands(fig, start_utc: datetime, end_utc: datetime) -> None:
    """Shade local-night intervals and weekends with distinct colors."""
    tz = ZoneInfo(config.LOCAL_TZ)
    night_start_h, night_end_h = config.NIGHT_HOURS
    start_local_date = start_utc.astimezone(tz).date() - timedelta(days=1)
    end_local_date = end_utc.astimezone(tz).date() + timedelta(days=1)

    cur = start_local_date
    while cur <= end_local_date:
        if cur.weekday() == 5:
            ws_local = datetime.combine(cur, time(0, 0), tzinfo=tz)
            we_local = datetime.combine(cur + timedelta(days=2), time(0, 0), tzinfo=tz)
            ws = max(ws_local.astimezone(timezone.utc), start_utc)
            we = min(we_local.astimezone(timezone.utc), end_utc)
            if ws < we:
                fig.add_vrect(x0=ws, x1=we, fillcolor="rgba(220,140,60,0.18)",
                              line_width=0, layer="below")
        cur += timedelta(days=1)
    cur = start_local_date
    while cur <= end_local_date:
        ns_local = datetime.combine(cur, time(night_start_h, 0), tzinfo=tz)
        ne_local = datetime.combine(cur + timedelta(days=1), time(night_end_h, 0), tzinfo=tz)
        ns = max(ns_local.astimezone(timezone.utc), start_utc)
        ne = min(ne_local.astimezone(timezone.utc), end_utc)
        if ns < ne:
            fig.add_vrect(x0=ns, x1=ne, fillcolor="rgba(70,90,180,0.32)",
                          line_width=0, layer="below")
        cur += timedelta(days=1)


def render_kpis(
    total_cw: float, daily_avg: float,
    peak_5h_share: float, peak_weekly_share: float,
    windows_over_pro_5h: int, windows_total_5h: int,
    weeks_over_pro: int, weeks_total: int,
):
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Cost-weighted", f"{total_cw/1e6:.1f}M")
    k2.metric("Daily avg", f"{daily_avg/1e6:.1f}M/d")
    k3.metric("Peak 5h", f"{peak_5h_share*100:.0f}%",
              help="% of Max 5x 5h cap (peak across all windows)")
    k4.metric("5h-windows over Pro", f"{windows_over_pro_5h} / {windows_total_5h}")
    k5.metric("Peak weekly", f"{peak_weekly_share*100:.0f}%",
              help="% of Max 5x weekly cap (peak across all weeks)")
    k6.metric("Weeks over Pro", f"{weeks_over_pro} / {weeks_total}")


def render_daily_bar(daily: pl.DataFrame):
    st.subheader("Daily burn — main vs subagents")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"].to_list(), y=daily["main"].to_list(),
                         name="main thread", marker_color="#4f8cff"))
    fig.add_trace(go.Bar(x=daily["date"].to_list(), y=daily["subagent"].to_list(),
                         name="subagents", marker_color="#ff8a4f"))
    fig.update_layout(
        barmode="stack", height=300, margin=dict(t=60, b=20, l=10, r=10),
        yaxis_title="cost-weighted tokens",
        yaxis=dict(autorange=True),
        xaxis=_rangeselector_xaxis(),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")


def render_sessions_table(sessions_sorted: pl.DataFrame, hidden: int,
                          min_turns: int, min_duration_s: int):
    st.subheader("All sessions in current filter")
    if hidden:
        st.caption(f"Hiding {hidden} degenerate session(s) (< {min_turns} turns or < {min_duration_s}s). "
                   "Adjust in the sidebar.")
    table = sessions_sorted.with_columns(
        pl.col("project_cwd").map_elements(short_project, return_dtype=pl.Utf8).alias("project"),
        (pl.col("peak_context_pct") * 100).round(1).alias("peak_ctx_%"),
        (pl.col("main_cost_weighted") / 1e6).round(2).alias("main_M"),
        (pl.col("subagent_cost_weighted") / 1e6).round(2).alias("sub_M"),
        (pl.col("total_cost_weighted") / 1e6).round(2).alias("total_M"),
    ).select([
        "start", "project", "model", "main_turns", "subagent_count",
        "peak_ctx_%", "peak_prompt_tokens", "main_M", "sub_M", "total_M",
    ])
    st.dataframe(table, width="stretch", height=400)


def render_calibration_history(df_cache: pl.DataFrame):
    log = calibration_log.load_log()
    if log.is_empty():
        return
    implied = caps_mod.implied_cap_series(log)
    tz = ZoneInfo(config.LOCAL_TZ)
    local_hours = [t.astimezone(tz).hour + t.astimezone(tz).minute / 60.0
                   for t in implied["sampled_at"].to_list()]
    implied = implied.with_columns(pl.Series("hour_local", local_hours, dtype=pl.Float64))

    with st.expander("Calibration history", expanded=False):
        col1, col2 = st.columns(2)
        for col, key, label in [
            (col1, "implied_5h", "Max 5x 5h cap (M)"),
            (col2, "implied_weekly", "Max 5x weekly cap (M)"),
        ]:
            valid = implied.drop_nulls(key)
            if valid.is_empty():
                col.caption(f"No valid samples yet for {label}")
                continue
            median = float(valid[key].median())
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid["sampled_at"].to_list(),
                y=(valid[key] / 1e6).to_list(),
                mode="markers",
                marker=dict(
                    size=6,
                    color=valid["hour_local"].to_list(),
                    colorscale="HSV", cmin=0, cmax=24,
                    showscale=True,
                    colorbar=dict(title="local hr", thickness=12),
                ),
                customdata=valid["hour_local"].to_list(),
                hovertemplate="%{x}<br>cap %{y:.1f}M<br>hour %{customdata:.1f}<extra></extra>",
                name="implied cap",
            ))
            fig.add_hline(
                y=median / 1e6, line_dash="dash", line_color="orange",
                annotation_text=f"median {median/1e6:.1f}M",
                annotation_position="top left",
            )
            fig.update_layout(
                title=label, height=240,
                margin=dict(t=40, b=20, l=10, r=10),
                yaxis_title="M cost-weighted tokens",
                showlegend=False,
            )
            col.plotly_chart(fig, width="stretch")

        st.markdown("**Implied cap by hour-of-day** (local time, median ± IQR)")
        col3, col4 = st.columns(2)
        for col, key, label in [
            (col3, "implied_5h", "Max 5x 5h cap by hour"),
            (col4, "implied_weekly", "Max 5x weekly cap by hour"),
        ]:
            valid = implied.drop_nulls(key)
            if valid.height < 6:
                col.caption(f"Need ≥6 valid samples for {label} (have {valid.height})")
                continue
            buckets = (
                valid.with_columns(pl.col("hour_local").floor().cast(pl.Int8).alias("hr"))
                .group_by("hr")
                .agg(
                    pl.col(key).median().alias("med"),
                    pl.col(key).quantile(0.25).alias("p25"),
                    pl.col(key).quantile(0.75).alias("p75"),
                    pl.col(key).count().alias("n"),
                )
                .sort("hr")
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=buckets["hr"].to_list(),
                y=(buckets["p75"] / 1e6).to_list(),
                mode="lines", line=dict(width=0, color="rgba(79,140,255,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=buckets["hr"].to_list(),
                y=(buckets["p25"] / 1e6).to_list(),
                mode="lines", line=dict(width=0, color="rgba(79,140,255,0)"),
                fill="tonexty", fillcolor="rgba(79,140,255,0.20)",
                name="IQR (p25–p75)", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=buckets["hr"].to_list(),
                y=(buckets["med"] / 1e6).to_list(),
                mode="lines+markers", line=dict(color="#4f8cff", width=2),
                marker=dict(
                    size=[max(6.0, 4 + n * 1.5) for n in buckets["n"].to_list()],
                    sizemode="diameter",
                ),
                customdata=buckets["n"].to_list(),
                hovertemplate="hour %{x}<br>median %{y:.1f}M<br>%{customdata} samples<extra></extra>",
                name="raw bin median (size = samples)",
            ))
            kind = "5h" if key == "implied_5h" else "weekly"
            min_burn = 1_000_000 if kind == "5h" else 10_000_000
            fitted = caps_mod.hour_of_day_cap_series(log, kind, min_burn=min_burn)
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=[c / 1e6 for c in fitted],
                mode="lines", line=dict(color="#ffa54f", width=1.5, dash="dot"),
                name="fitted (smoothed + interp)",
                hovertemplate="hour %{x}<br>fitted %{y:.1f}M<extra></extra>",
            ))
            ns, ne = config.NIGHT_HOURS
            if ns > ne:
                fig.add_vrect(x0=ns, x1=24, fillcolor="rgba(70,90,180,0.18)", line_width=0, layer="below")
                fig.add_vrect(x0=0, x1=ne, fillcolor="rgba(70,90,180,0.18)", line_width=0, layer="below")
            fig.update_layout(
                title=label, height=240,
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(title="hour of day (local)", range=[0, 24], dtick=3),
                yaxis_title="M cost-weighted tokens",
                showlegend=True,
                legend=dict(orientation="h", y=-0.25),
            )
            col.plotly_chart(fig, width="stretch")

        eff_h, n_obs = metrics.effective_window_hours(
            log, df_cache, default=config.FIVE_HOUR_WINDOW_HOURS, min_samples=5,
        )
        st.caption(
            f"Samples used per series: {int(caps_mod.CONTINUOUS_MIN_UTIL*100)}% ≤ util ≤ "
            f"{int(caps_mod.CONTINUOUS_MAX_UTIL*100)}%. "
            f"Total samples in log: {log.height}. "
            f"Effective 5h window length: {eff_h:.2f}h "
            f"({'calibrated from ' + str(n_obs) + ' observed reset(s)' if n_obs >= 5 else f'default — need 5+ resets, have {n_obs}'})."
        )


def render_live_panel_from_cache(*, agent_seconds_old: float | None):
    """Cloud-only live panel: builds from caps.json instead of an API call."""
    prev = caps_mod.load_caps()
    st.subheader("Live plan usage")

    if agent_seconds_old is None:
        st.error("Agent has never reported — Supabase bucket is empty.")
        return
    if agent_seconds_old < 120:
        st.success(f"Agent live · reported {int(agent_seconds_old)}s ago")
    elif agent_seconds_old < 600:
        st.warning(f"Agent may be stale · last reported {int(agent_seconds_old/60)}m ago")
    else:
        st.error(f"Agent appears offline · last reported {int(agent_seconds_old/60)}m ago")

    if not prev.sampled_at:
        st.caption("No cached calibration available yet.")
        return

    sampled_at = datetime.fromisoformat(prev.sampled_at)
    if sampled_at.tzinfo is None:
        sampled_at = sampled_at.replace(tzinfo=timezone.utc)
    resets_5h = datetime.fromisoformat(prev.resets_5h_iso) if prev.resets_5h_iso else None
    resets_7d = datetime.fromisoformat(prev.resets_7d_iso) if prev.resets_7d_iso else None
    now = datetime.now(tz=timezone.utc)
    cols = st.columns(4)
    if prev.sample_util_5h is not None:
        cols[0].metric("5h utilization", f"{prev.sample_util_5h*100:.0f}%")
        cols[0].progress(min(1.0, prev.sample_util_5h))
        if resets_5h:
            mins = max(0, (resets_5h - now).total_seconds() / 60)
            cols[1].metric("5h resets in", f"{int(mins//60)}h {int(mins%60)}m")
    if prev.sample_util_7d is not None:
        cols[2].metric("7d utilization", f"{prev.sample_util_7d*100:.0f}%")
        cols[2].progress(min(1.0, prev.sample_util_7d))
        if resets_7d:
            hours = max(0, (resets_7d - now).total_seconds() / 3600)
            cols[3].metric("7d resets in", f"{int(hours//24)}d {int(hours%24)}h")
    if prev.max5x_5h or prev.max5x_weekly:
        bits = []
        if prev.max5x_5h:
            bits.append(f"Max 5x 5h cap ≈ {prev.max5x_5h/1e6:.0f}M (Pro {prev.pro_5h/1e6:.0f}M)")
        if prev.max5x_weekly:
            bits.append(f"Max 5x weekly cap ≈ {prev.max5x_weekly/1e6:.0f}M (Pro {prev.pro_weekly/1e6:.0f}M)")
        st.caption("Calibrated · " + " · ".join(bits))


@st.fragment
def _cost_vs_session_length_interactive(sessions: pl.DataFrame, diag: dict) -> None:
    """X-axis/bin controls + the two charts, isolated in a fragment.

    Widget changes rerun only this block (cheap `bin_sessions` + redraw) instead of
    the whole app — and never recompute the expensive, widget-independent attribution
    upstream. `sessions`/`diag` come from the last full run via the captured args.
    """
    x_label_to_col = {
        "Prompt tokens": "prompt_tokens",
        "Requests (turns)": "n_requests",
        "Raw total tokens": "raw_total_tokens",
    }
    ctrl_x, ctrl_bins = st.columns([2, 1])
    x_label = ctrl_x.selectbox("X-axis", list(x_label_to_col.keys()), key="cvsl_x")
    n_bins = ctrl_bins.slider("Bins", min_value=4, max_value=20, value=8, key="cvsl_bins")
    x_col = x_label_to_col[x_label]

    left, right = st.columns(2)
    for col, y_col, title in (
        (left, "attributed_pct_5h", "5h window"),
        (right, "attributed_pct_weekly", "Weekly window"),
    ):
        binned = metrics.bin_sessions(sessions, x_col, y_col, n_bins)
        fig = go.Figure()
        if not binned.is_empty():
            std_err = [(v * 100 if v is not None else 0.0) for v in binned["std_y"].to_list()]
            fig.add_trace(go.Scatter(
                x=binned["bin_median_x"].to_list(),
                y=(binned["mean_y"] * 100).to_list(),
                error_y=dict(type="data", array=std_err, visible=True),
                mode="lines+markers",
                customdata=binned["n"].to_list(),
                hovertemplate="x=%{x:.0f}<br>mean=%{y:.2f}%<br>n=%{customdata}<extra></extra>",
                line=dict(color="#4f8cff"),
            ))
        fig.update_layout(
            title=title, height=350, margin=dict(t=40, b=20, l=10, r=10),
            xaxis_title=x_label, yaxis_title="% of cap consumed (attributed)",
        )
        col.plotly_chart(fig, width="stretch", key=f"cvsl_{y_col}")

    st.caption(
        f"Unattributed burn (API % with no matching logged turn): "
        f"5h {diag['unattributed_5h'] * 100:.1f}%, "
        f"weekly {diag['unattributed_7d'] * 100:.1f}% (cumulative across all windows)."
    )


def render_cost_vs_session_length(df: pl.DataFrame, log: pl.DataFrame) -> None:
    """Binned mean+std of each session's attributed cap-% vs session size.

    df must be the derived cache (with `ts`); see the render-helper data-prep
    convention. log is calibration_log.load_log().

    Attribution (an expensive full-cache scan) is computed here, once per app run;
    the interactive controls/charts live in a fragment so widget changes don't
    trigger a full rerun or recompute attribution.
    """
    st.subheader("Cost vs session length")
    st.caption(
        "Each session's measured share of the cap — attributed from real API % "
        "deltas, split by output-token share — vs how big the session was. "
        "An upward bend means longer sessions burn disproportionately more."
    )
    if df.is_empty():
        st.info("No session data yet.")
        return
    calib = app_cache.calibrate(df, log)
    sessions, diag = calib.sessions, calib.diag
    if sessions.is_empty():
        st.info("Not enough calibration data to attribute cost yet.")
        return
    _cost_vs_session_length_interactive(sessions, diag)
