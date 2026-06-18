"""Shared rendering helpers used by both app.py and app_cloud.py."""
from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import caps as caps_mod
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


def render_kpis(total_usd: float, daily_avg_usd: float,
                peak_5h: float | None, peak_weekly: float | None,
                windows_over_pro_5h: int, windows_total_5h: int,
                weeks_over_pro: int, weeks_total: int):
    def pct(v):
        return f"{v*100:.0f}%" if v is not None else "—"
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total $", f"${total_usd:,.0f}")
    k2.metric("Daily avg", f"${daily_avg_usd:,.1f}/d")
    k3.metric("Peak 5h", pct(peak_5h), help="Max reported 5h utilization (Max 5x)")
    k4.metric("5h-windows over Pro", f"{windows_over_pro_5h} / {windows_total_5h}")
    k5.metric("Peak weekly", pct(peak_weekly), help="Max reported weekly utilization (Max 5x)")
    k6.metric("Weeks over Pro", f"{weeks_over_pro} / {weeks_total}")


_DAILY_PALETTE = ["#4f8cff", "#ff8a4f", "#46b46e", "#b46edc", "#dcb446", "#46b4b4"]


def build_daily_figure(daily: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    series_cols = [c for c in daily.columns if c != "date"]
    x = daily["date"].to_list()
    for i, col in enumerate(series_cols):
        fig.add_trace(go.Bar(x=x, y=daily[col].to_list(), name=col,
                             marker_color=_DAILY_PALETTE[i % len(_DAILY_PALETTE)]))
    fig.update_layout(
        barmode="stack", height=300, margin=dict(t=60, b=20, l=10, r=10),
        yaxis_title="USD", yaxis=dict(autorange=True),
        xaxis=_rangeselector_xaxis(), legend=dict(orientation="h"),
    )
    return fig


def render_daily_bar(fdf: pl.DataFrame, decomposition_key: str) -> None:
    st.subheader("Daily burn (USD)")
    st.caption("Estimated API-equivalent cost (what this usage would cost at pay-as-you-go "
               "API rates — comparable to ccusage). Cache writes priced at 1.25× input (5m).")
    mode = st.radio("Decomposition", ["by machine", "main vs sub"],
                    index=0, horizontal=True, key=f"daily_decomp_{decomposition_key}")
    by = "machine" if mode == "by machine" else "is_subagent"
    daily = metrics.daily_stacked(fdf, by=by)
    if daily.is_empty():
        st.info("No data for the daily chart.")
        return
    st.plotly_chart(build_daily_figure(daily), width="stretch")


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


def format_projection(proj) -> str:
    if proj is None or proj.eta is None:
        return "—"
    if not proj.before_reset:
        return "won't hit 100% before reset"
    if proj.eta.total_seconds() <= 0:
        return "at/over the cap now"
    mins = int(proj.eta.total_seconds() // 60)
    return f"~{mins // 60}h {mins % 60}m"


def render_live_panel_from_cache(*, agent_seconds_old: float | None, log: pl.DataFrame):
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
    if prev.sample_util_5h is not None:
        proj = metrics.project_time_to_cap(log, now, "5h")
        cols[1].caption("5h → 100%: " + format_projection(proj))


