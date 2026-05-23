"""Pure computation: cost weighting, rolling windows, per-session context curves."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from zoneinfo import ZoneInfo

import polars as pl

import config


def _parse_window(window: str) -> timedelta:
    if window.endswith("h"):
        return timedelta(hours=int(window[:-1]))
    if window.endswith("d"):
        return timedelta(days=int(window[:-1]))
    raise ValueError(f"Unsupported window string: {window}")


def _inject_gap_probes(df: pl.DataFrame, window: str) -> pl.DataFrame:
    """Insert zero-valued probe rows so plotly interpolation across long gaps
    correctly drops to 0 once the rolling window has fully decayed.

    Expects columns ts (Datetime) and cost_weighted_tokens (Float64). Probes are
    inserted at (prev_ts + window) and (next_ts - 1ms) for any gap > window.
    """
    if df.height < 2:
        return df
    df = df.sort("ts")
    window_td = _parse_window(window)
    ts_list = df["ts"].to_list()
    probes: list = []
    for i in range(len(ts_list) - 1):
        prev_ts, next_ts = ts_list[i], ts_list[i + 1]
        if (next_ts - prev_ts) > window_td:
            decay_ts = prev_ts + window_td
            if decay_ts < next_ts:
                probes.append(decay_ts)
                probes.append(next_ts - timedelta(milliseconds=1))
    if not probes:
        return df
    probe_df = pl.DataFrame(
        {"ts": probes, "cost_weighted_tokens": [0.0] * len(probes)},
        schema={"ts": pl.Datetime("ms", "UTC"), "cost_weighted_tokens": pl.Float64},
    )
    return pl.concat([df, probe_df], how="diagonal").sort("ts")


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Parse timestamps and add cost-weighted, raw-input, context-prompt columns."""
    if df.is_empty():
        return df
    w = config.COST_WEIGHTS
    return (
        df.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime("ms", "UTC"), format="%Y-%m-%dT%H:%M:%S%.fZ", strict=False)
            .alias("ts"),
        )
        .with_columns(
            (
                pl.col("input_tokens") * w["input"]
                + pl.col("cache_creation_input_tokens") * w["cache_creation"]
                + pl.col("cache_read_input_tokens") * w["cache_read"]
                + pl.col("output_tokens") * w["output"]
            ).alias("cost_weighted_tokens"),
            (
                pl.col("input_tokens")
                + pl.col("cache_creation_input_tokens")
                + pl.col("cache_read_input_tokens")
            ).alias("prompt_tokens"),
            (
                pl.col("input_tokens")
                + pl.col("cache_creation_input_tokens")
                + pl.col("cache_read_input_tokens")
                + pl.col("output_tokens")
            ).alias("raw_total_tokens"),
        )
        .filter(pl.col("ts").is_not_null())
        .sort("ts")
    )


def rolling_burn(df: pl.DataFrame, window: str, by_subagent: bool = True) -> pl.DataFrame:
    """Rolling sum of cost_weighted_tokens over `window` (e.g., '5h', '7d').

    Returns one row per assistant turn with the rolling-sum value at that timestamp.
    If by_subagent, splits into main vs subagent series.
    """
    if df.is_empty():
        return pl.DataFrame()
    work = df.select(["ts", "cost_weighted_tokens", "is_subagent"]).sort("ts")
    if by_subagent:
        combined = _inject_gap_probes(
            work.select(["ts", "cost_weighted_tokens", "is_subagent"]),
            window,
        )
        # All three rolling series computed on the same combined timeline so
        # gap probes lock all three to 0 at the same timestamps.
        return combined.with_columns(
            pl.col("cost_weighted_tokens")
              .rolling_sum_by("ts", window_size=window).alias("rolling_total"),
            pl.when(pl.col("is_subagent").fill_null(True))
              .then(0.0).otherwise(pl.col("cost_weighted_tokens"))
              .rolling_sum_by("ts", window_size=window).alias("rolling_main"),
            pl.when(pl.col("is_subagent").fill_null(False))
              .then(pl.col("cost_weighted_tokens")).otherwise(0.0)
              .rolling_sum_by("ts", window_size=window).alias("rolling_sub"),
        ).select(["ts", "rolling_total", "rolling_main", "rolling_sub"])
    total_src = _inject_gap_probes(work.select(["ts", "cost_weighted_tokens"]), window)
    return total_src.with_columns(
        pl.col("cost_weighted_tokens").rolling_sum_by("ts", window_size=window).alias("rolling_total")
    ).select(["ts", "rolling_total"])


def daily_stacked(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "main": pl.Float64, "subagent": pl.Float64})
    pivoted = (
        df.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by(["date", "is_subagent"])
        .agg(pl.col("cost_weighted_tokens").sum().alias("total"))
        .pivot(values="total", index="date", on="is_subagent")
        .sort("date")
    )
    rename_map = {}
    for c in pivoted.columns:
        if c in ("true", "True"):
            rename_map[c] = "subagent"
        elif c in ("false", "False"):
            rename_map[c] = "main"
    pivoted = pivoted.rename(rename_map)
    for needed in ("main", "subagent"):
        if needed not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0.0).alias(needed))
    return pivoted.with_columns(
        pl.col("main").fill_null(0.0),
        pl.col("subagent").fill_null(0.0),
    )


def fraction_time_over_cap(rolling_df: pl.DataFrame, cap: float, col: str = "rolling_total") -> float:
    """Fraction of total elapsed time where the rolling sum was above cap.

    Treats the rolling value as constant between consecutive turn timestamps
    (left-edge step function).
    """
    if rolling_df.is_empty() or rolling_df.height < 2:
        return 0.0
    df = rolling_df.sort("ts").with_columns(
        pl.col("ts").shift(-1).alias("next_ts"),
    ).with_columns(
        ((pl.col("next_ts") - pl.col("ts")).dt.total_milliseconds() / 1000.0).alias("gap_s"),
    ).drop_nulls("gap_s")
    total = float(df["gap_s"].sum())
    if total <= 0:
        return 0.0
    over = float(df.filter(pl.col(col) > cap)["gap_s"].sum())
    return over / total


def cap_crossings(rolling_df: pl.DataFrame, cap: float, col: str = "rolling_total") -> pl.DataFrame:
    """Timestamps where rolling sum first crosses above cap (and re-crossings after dipping)."""
    if rolling_df.is_empty():
        return pl.DataFrame(schema={"ts": pl.Datetime, col: pl.Float64})
    df = rolling_df.with_columns(
        (pl.col(col) > cap).alias("over"),
    )
    df = df.with_columns(
        pl.col("over").shift(1).fill_null(False).alias("over_prev")
    )
    crossings = df.filter(pl.col("over") & ~pl.col("over_prev")).select(["ts", col])
    return crossings


def observed_window_lengths(
    log: pl.DataFrame,
    cache_df: pl.DataFrame,
    default_hours: float = 4.5,
    sanity_min: float = 0.5,
    sanity_max: float = 6.0,
) -> list[float]:
    """Return observed 5h window lengths in hours, one per unique reset boundary.

    For each unique resets_5h_iso in the calibration log:
      window_end   = resets_5h_iso (Anthropic-reported)
      window_start = first activity in cache_df that follows a >= default_hours
                     gap and lies within (resets - default_hours - 0.5h, resets)
      length       = window_end - window_start
    Lengths outside [sanity_min, sanity_max] hours are discarded.
    """
    if log.is_empty() or "resets_5h_iso" not in log.columns or cache_df.is_empty():
        return []
    unique_resets = (
        log.filter(pl.col("resets_5h_iso").is_not_null())
        .select("resets_5h_iso")
        .unique()
    )
    if unique_resets.is_empty():
        return []

    cache_sorted = cache_df.sort("ts")
    lengths: list[float] = []
    for reset_iso in unique_resets["resets_5h_iso"].to_list():
        try:
            reset_dt = datetime.fromisoformat(reset_iso)
        except (ValueError, TypeError):
            continue
        if reset_dt.tzinfo is None:
            reset_dt = reset_dt.replace(tzinfo=timezone.utc)

        window_start_floor = reset_dt - timedelta(hours=default_hours + 0.5)
        in_window = cache_sorted.filter(
            (pl.col("ts") >= window_start_floor) & (pl.col("ts") <= reset_dt)
        )
        if in_window.is_empty():
            continue
        first_activity = in_window["ts"].min()

        # Confirm there's a gap of at least 0.8 * default_hours before first_activity
        before = cache_sorted.filter(pl.col("ts") < first_activity)
        if not before.is_empty():
            prev_activity = before["ts"].max()
            gap = (first_activity - prev_activity).total_seconds() / 3600.0
            if gap < default_hours * 0.8:
                continue

        length = (reset_dt - first_activity).total_seconds() / 3600.0
        if sanity_min <= length <= sanity_max:
            lengths.append(length)
    return lengths


def effective_window_hours(
    log: pl.DataFrame,
    cache_df: pl.DataFrame,
    default: float | None = None,
    min_samples: int = 5,
) -> tuple[float, int]:
    """Return (window_hours, n_observed). Uses median observed length once n >= min_samples,
    otherwise default."""
    if default is None:
        default = config.FIVE_HOUR_WINDOW_HOURS
    observed = observed_window_lengths(log, cache_df, default_hours=default)
    if len(observed) < min_samples:
        return float(default), len(observed)
    return float(median(observed)), len(observed)


def week_start_for(ts_utc: datetime) -> datetime:
    """Most recent week-reset boundary (default: Sunday 07:00 local) ≤ ts_utc."""
    tz = ZoneInfo(config.LOCAL_TZ)
    reset_wd = config.WEEKLY_RESET_WEEKDAY
    reset_h = config.WEEKLY_RESET_HOUR_LOCAL
    ts_local = ts_utc.astimezone(tz)
    days_back = (ts_local.weekday() - reset_wd) % 7
    candidate = (ts_local - timedelta(days=days_back)).replace(
        hour=reset_h, minute=0, second=0, microsecond=0
    )
    if candidate > ts_local:
        candidate -= timedelta(days=7)
    return candidate.astimezone(timezone.utc)


def five_hour_burn_since_reset(
    df: pl.DataFrame,
    gap_hours: float | None = None,
    value_col: str = "cost_weighted_tokens",
    selected_mask_col: str | None = None,
) -> pl.DataFrame:
    """Cumulative burn within each fixed 5h window.

    A new window opens at the first activity satisfying either:
      - elapsed >= gap_hours since the previous activity, OR
      - elapsed >= gap_hours since the current window's start (window expired)

    Returns ts, cumulative_total, cumulative_selected, cumulative_main, cumulative_sub with:
      - reset marker at each window_start (cum=0)
      - end-of-window drop at window_start + gap_hours when there is a gap
        before the next window opens (cum=0)
    """
    schema = {
        "ts": pl.Datetime("ms", "UTC"),
        "cumulative_total": pl.Float64,
        "cumulative_selected": pl.Float64,
        "cumulative_main": pl.Float64,
        "cumulative_sub": pl.Float64,
    }
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    if gap_hours is None:
        gap_hours = config.FIVE_HOUR_WINDOW_HOURS

    cols = ["ts", value_col, "is_subagent"]
    if selected_mask_col:
        cols.append(selected_mask_col)
    sorted_df = df.sort("ts").select(cols)

    ts_list = sorted_df["ts"].to_list()
    gap = timedelta(hours=gap_hours)
    window_starts: list = []
    current_start = None
    last_ts = None
    for ts in ts_list:
        if current_start is None:
            current_start = ts
        elif (ts - last_ts) >= gap or (ts - current_start) >= gap:
            current_start = ts
        window_starts.append(current_start)
        last_ts = ts

    selected = pl.col(selected_mask_col) if selected_mask_col else pl.lit(True)

    work = sorted_df.with_columns(
        pl.Series("window_start", window_starts, dtype=pl.Datetime("ms", "UTC"))
    ).with_columns(
        pl.col(value_col).cum_sum().over("window_start").alias("cumulative_total"),
        pl.when(selected).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("window_start").alias("cumulative_selected"),
        pl.when(selected & ~pl.col("is_subagent")).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("window_start").alias("cumulative_main"),
        pl.when(selected & pl.col("is_subagent")).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("window_start").alias("cumulative_sub"),
    )

    unique_ws = sorted(set(window_starts))
    reset_rows = [
        {
            "ts": ws,
            "cumulative_total": 0.0, "cumulative_selected": 0.0,
            "cumulative_main": 0.0, "cumulative_sub": 0.0,
        }
        for ws in unique_ws
    ]
    end_drop_rows = []
    for i, ws in enumerate(unique_ws):
        window_end = ws + gap
        next_ws = unique_ws[i + 1] if i + 1 < len(unique_ws) else None
        if next_ws is None or next_ws > window_end:
            end_drop_rows.append({
                "ts": window_end,
                "cumulative_total": 0.0, "cumulative_selected": 0.0,
                "cumulative_main": 0.0, "cumulative_sub": 0.0,
            })

    extras = pl.DataFrame(reset_rows + end_drop_rows, schema=schema)
    return pl.concat(
        [
            work.select([
                "ts", "cumulative_total", "cumulative_selected",
                "cumulative_main", "cumulative_sub",
            ]),
            extras,
        ],
        how="diagonal",
    ).sort("ts")


def five_hour_window_totals(
    df: pl.DataFrame,
    gap_hours: float | None = None,
    value_col: str = "cost_weighted_tokens",
) -> list[float]:
    """Per-window totals of value_col, using the same window logic as
    five_hour_burn_since_reset."""
    if df.is_empty():
        return []
    if gap_hours is None:
        gap_hours = config.FIVE_HOUR_WINDOW_HOURS
    sorted_df = df.sort("ts").select(["ts", value_col])
    ts_list = sorted_df["ts"].to_list()
    cw_list = sorted_df[value_col].to_list()
    gap = timedelta(hours=gap_hours)
    totals: list[float] = []
    current_start = None
    current_total = 0.0
    last_ts = None
    for ts, cw in zip(ts_list, cw_list):
        if current_start is None:
            current_start = ts
            current_total = 0.0
        elif (ts - last_ts) >= gap or (ts - current_start) >= gap:
            totals.append(current_total)
            current_start = ts
            current_total = 0.0
        current_total += cw
        last_ts = ts
    if current_start is not None:
        totals.append(current_total)
    return totals


def weekly_burn_since_reset(
    df: pl.DataFrame,
    value_col: str = "cost_weighted_tokens",
    selected_mask_col: str | None = None,
) -> pl.DataFrame:
    """Cumulative burn within each fixed weekly window.

    Returns ts, cumulative_total, cumulative_selected, cumulative_main, cumulative_sub.
    - cumulative_total: sum over ALL rows
    - cumulative_selected: sum over rows where selected_mask_col is True (or all rows if None)
    - cumulative_main: sum over selected rows that are NOT subagents
    - cumulative_sub:  sum over selected rows that ARE subagents

    Inserts a zero-valued reset row at each week boundary for clean sawtooth rendering.
    """
    schema = {
        "ts": pl.Datetime("ms", "UTC"),
        "cumulative_total": pl.Float64,
        "cumulative_selected": pl.Float64,
        "cumulative_main": pl.Float64,
        "cumulative_sub": pl.Float64,
    }
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    selected = pl.col(selected_mask_col) if selected_mask_col else pl.lit(True)

    work = (
        df.select(["ts", value_col, "is_subagent"] + ([selected_mask_col] if selected_mask_col else []))
        .sort("ts")
        .with_columns(
            pl.col("ts")
            .map_elements(week_start_for, return_dtype=pl.Datetime("us", "UTC"))
            .cast(pl.Datetime("ms", "UTC"))
            .alias("week_start"),
        )
    )
    work = work.with_columns(
        pl.col(value_col).cum_sum().over("week_start").alias("cumulative_total"),
        pl.when(selected).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("week_start").alias("cumulative_selected"),
        pl.when(selected & ~pl.col("is_subagent")).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("week_start").alias("cumulative_main"),
        pl.when(selected & pl.col("is_subagent")).then(pl.col(value_col)).otherwise(0.0)
          .cum_sum().over("week_start").alias("cumulative_sub"),
    )

    week_starts = work["week_start"].unique().sort().to_list()
    if week_starts:
        reset_df = pl.DataFrame(
            {
                "ts": week_starts,
                "cumulative_total": [0.0] * len(week_starts),
                "cumulative_selected": [0.0] * len(week_starts),
                "cumulative_main": [0.0] * len(week_starts),
                "cumulative_sub": [0.0] * len(week_starts),
            },
            schema=schema,
        )
        out = pl.concat(
            [
                work.select([
                    "ts", "cumulative_total", "cumulative_selected",
                    "cumulative_main", "cumulative_sub",
                ]),
                reset_df,
            ],
            how="diagonal",
        ).sort("ts")
    else:
        out = work.select([
            "ts", "cumulative_total", "cumulative_selected",
            "cumulative_main", "cumulative_sub",
        ])

    return out


def downsample_cumulative(df: pl.DataFrame, max_points: int = 500) -> pl.DataFrame:
    """Reduce a cumulative-series df to ~max_points rows via time-binning.

    Cumulative values monotonically climb within a window, so taking the LAST value
    per bin preserves the climb shape. Bins are aligned to the first row's
    timestamp (t0-relative, not Unix-epoch-floored). The exact first and last
    rows are always preserved in the output regardless of binning.

    Pass-through cases (input returned unchanged):
    - df.height <= max_points
    - span (last ts - first ts) <= 0

    Input df must have columns: ts (Datetime), cumulative_total,
    cumulative_selected, cumulative_main, cumulative_sub.
    """
    if df.height <= max_points:
        return df
    span = (df["ts"].max() - df["ts"].min()).total_seconds()
    if span <= 0:
        return df
    bin_seconds = max(60.0, span / max_points)
    bin_ms = int(bin_seconds * 1000)
    t0_ms_int = int(df["ts"].min().timestamp() * 1000)  # type: ignore[union-attr]
    cols = ["ts", "cumulative_total", "cumulative_selected", "cumulative_main", "cumulative_sub"]
    binned = (
        df.sort("ts")
        .with_columns(
            (((pl.col("ts").dt.timestamp("ms") - t0_ms_int) // bin_ms) * bin_ms + t0_ms_int)
            .cast(pl.Datetime("ms", "UTC")).alias("bin"),
        )
        .group_by("bin")
        .agg(
            pl.col("cumulative_total").sort_by("ts").last(),
            pl.col("cumulative_selected").sort_by("ts").last(),
            pl.col("cumulative_main").sort_by("ts").last(),
            pl.col("cumulative_sub").sort_by("ts").last(),
        )
        .sort("bin")
        .rename({"bin": "ts"})
    )
    # Always preserve first and last rows so boundary values are never lost.
    # Boundary rows go first so unique("ts", keep="first") retains them over bin aggregates.
    boundary = pl.concat([df.select(cols).head(1), df.select(cols).tail(1)])
    return pl.concat([boundary, binned]).unique("ts", keep="first").sort("ts")


def session_summaries(df: pl.DataFrame) -> pl.DataFrame:
    """Per-session: start, end, peak_context_pct, total_cost_weighted, subagent_count, model."""
    if df.is_empty():
        return pl.DataFrame()

    def window_for(model: str) -> int:
        return config.context_window_for(model)

    main = df.filter(~pl.col("is_subagent"))
    if main.is_empty():
        return pl.DataFrame()

    main = main.with_columns(
        pl.col("model").map_elements(window_for, return_dtype=pl.Int64).alias("context_window"),
    ).with_columns(
        (pl.col("prompt_tokens") / pl.col("context_window")).alias("context_pct"),
    )

    per_session_main = main.group_by("session_id").agg(
        pl.col("ts").min().alias("start"),
        pl.col("ts").max().alias("end"),
        pl.col("project_cwd").last().alias("project_cwd"),
        pl.col("model").last().alias("model"),
        pl.col("context_pct").max().alias("peak_context_pct"),
        pl.col("prompt_tokens").max().alias("peak_prompt_tokens"),
        pl.col("cost_weighted_tokens").sum().alias("main_cost_weighted"),
        pl.col("ts").count().alias("main_turns"),
    )

    sub_agg = (
        df.filter(pl.col("is_subagent"))
        .group_by("session_id")
        .agg(
            pl.col("cost_weighted_tokens").sum().alias("subagent_cost_weighted"),
            pl.col("subagent_id").n_unique().alias("subagent_count"),
            pl.col("ts").count().alias("subagent_turns"),
        )
    )

    out = per_session_main.join(sub_agg, on="session_id", how="left").with_columns(
        pl.col("subagent_cost_weighted").fill_null(0.0),
        pl.col("subagent_count").fill_null(0),
        pl.col("subagent_turns").fill_null(0),
    ).with_columns(
        (pl.col("main_cost_weighted") + pl.col("subagent_cost_weighted")).alias("total_cost_weighted"),
    ).sort("start")

    return out


def session_context_curve(df: pl.DataFrame, session_id: str) -> dict:
    """For one session, return main-thread + per-subagent context-utilization curves.

    Each curve has columns ts, prompt_tokens, pct (= prompt_tokens / context_window for the model).
    """
    sdf = df.filter(pl.col("session_id") == session_id).sort("ts")
    if sdf.is_empty():
        return {"main": pl.DataFrame(), "subagents": {}, "main_window": config.DEFAULT_CONTEXT_WINDOW}

    def pct_for(df_in: pl.DataFrame) -> pl.DataFrame:
        return df_in.with_columns(
            pl.col("model")
            .map_elements(config.context_window_for, return_dtype=pl.Int64)
            .alias("context_window"),
        ).with_columns(
            (pl.col("prompt_tokens") / pl.col("context_window") * 100).alias("pct"),
        )

    main = sdf.filter(~pl.col("is_subagent")).select(["ts", "prompt_tokens", "model"])
    main = pct_for(main).select(["ts", "prompt_tokens", "pct", "context_window"])

    subs: dict[str, pl.DataFrame] = {}
    sub_df = sdf.filter(pl.col("is_subagent"))
    if not sub_df.is_empty():
        sub_df = pct_for(sub_df.select(["ts", "prompt_tokens", "model", "subagent_id"]))
        for sid in sub_df["subagent_id"].drop_nulls().unique().to_list():
            subs[sid] = sub_df.filter(pl.col("subagent_id") == sid).select(["ts", "prompt_tokens", "pct"]).sort("ts")

    main_window = int(main["context_window"].mode().first()) if not main.is_empty() else config.DEFAULT_CONTEXT_WINDOW
    return {"main": main, "subagents": subs, "main_window": main_window}


def detect_compactions(main_curve: pl.DataFrame, drop_ratio: float = 0.4, min_drop_abs: int = 20_000) -> pl.DataFrame:
    """Detect timestamps where main-thread prompt_tokens drops sharply between consecutive turns.

    A compaction event = next turn's prompt_tokens is < (1 - drop_ratio) * current AND
    the absolute drop is at least min_drop_abs. Returns rows {ts, before, after}.
    """
    if main_curve.is_empty() or main_curve.height < 2:
        return pl.DataFrame(schema={"ts": pl.Datetime("ms", "UTC"), "before": pl.Int64, "after": pl.Int64})
    df = main_curve.sort("ts").with_columns(
        pl.col("prompt_tokens").shift(-1).alias("next_pt"),
        pl.col("ts").shift(-1).alias("next_ts"),
    ).drop_nulls(["next_pt", "next_ts"])
    drops = df.filter(
        (pl.col("next_pt") < pl.col("prompt_tokens") * (1 - drop_ratio))
        & ((pl.col("prompt_tokens") - pl.col("next_pt")) >= min_drop_abs)
    )
    return drops.select(
        pl.col("next_ts").alias("ts"),
        pl.col("prompt_tokens").alias("before"),
        pl.col("next_pt").alias("after"),
    )


def session_cost_attribution(
    df: pl.DataFrame, log: pl.DataFrame
) -> tuple[pl.DataFrame, dict]:
    """Attribute each window's API utilization delta to sessions by output-token share.

    Returns (sessions, diagnostics):
      - sessions: one row per session_id with attributed_pct_5h, attributed_pct_weekly
        (0-1 fractions), prompt_tokens, n_requests, raw_total_tokens.
      - diagnostics: {"unattributed_5h": float, "unattributed_7d": float} — summed
        delta (fraction) within valid same-window intervals that matched no logged turn.
        Cross-reset and null-reset intervals are excluded entirely (not counted here).
    """
    out_schema = {
        "session_id": pl.Utf8,
        "attributed_pct_5h": pl.Float64,
        "attributed_pct_weekly": pl.Float64,
        "prompt_tokens": pl.Int64,
        "n_requests": pl.Int64,
        "raw_total_tokens": pl.Int64,
    }
    diagnostics = {"unattributed_5h": 0.0, "unattributed_7d": 0.0}
    if df.is_empty():
        return pl.DataFrame(schema=out_schema), diagnostics

    per_session = df.group_by("session_id").agg(
        pl.col("prompt_tokens").sum().cast(pl.Int64).alias("prompt_tokens"),
        pl.col("raw_total_tokens").sum().cast(pl.Int64).alias("raw_total_tokens"),
        (~pl.col("is_subagent")).sum().cast(pl.Int64).alias("n_requests"),
    )

    def attribute(util_col: str, reset_col: str) -> tuple[pl.DataFrame, float]:
        per_sess_schema = {"session_id": pl.Utf8, "attributed": pl.Float64}
        if log.is_empty() or util_col not in log.columns:
            return pl.DataFrame(schema=per_sess_schema), 0.0
        lg = (
            log.select(
                pl.col("sampled_at"),
                pl.col(util_col).alias("util"),
                pl.col(reset_col).alias("reset_id"),
            )
            .drop_nulls(["sampled_at", "util"])
            .sort("sampled_at")
        )
        if lg.height < 2:
            return pl.DataFrame(schema=per_sess_schema), 0.0
        # resets_*_iso are ISO timestamps of the window's reset instant carrying ~1s
        # of jitter (they are NOT stable labels). Two consecutive samples are the same
        # window iff both reset instants parse and are within an hour of each other;
        # genuinely distinct windows are ≥5h (5h) / 7d (weekly) apart.
        lg = lg.with_columns(
            pl.col("reset_id")
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f%z", strict=False)
            .alias("win_dt"),
        ).with_columns(
            pl.col("sampled_at").shift(1).alias("start_ts"),
            pl.col("util").shift(1).alias("prev_util"),
            pl.col("win_dt").shift(1).alias("prev_win_dt"),
        ).with_columns((pl.col("util") - pl.col("prev_util")).alias("delta"))
        intervals = (
            lg.filter(
                pl.col("start_ts").is_not_null()
                & pl.col("win_dt").is_not_null()
                & pl.col("prev_win_dt").is_not_null()
                & ((pl.col("win_dt") - pl.col("prev_win_dt")).dt.total_seconds().abs() <= 3600)
                & (pl.col("delta") > 0)
            )
            .select(
                pl.col("start_ts"),
                pl.col("sampled_at").alias("end_ts"),
                pl.col("delta"),
            )
            .sort("end_ts")
            .with_row_index("interval_id")
        )
        if intervals.is_empty():
            return pl.DataFrame(schema=per_sess_schema), 0.0

        tokens = df.select(["ts", "session_id", "output_tokens"]).sort("ts")
        # Half-open (start_ts, end_ts]: forward asof picks the smallest end_ts >= ts,
        # then the ts > start_ts filter drops rows that precede the matched interval.
        matched = tokens.join_asof(
            intervals, left_on="ts", right_on="end_ts", strategy="forward"
        ).filter(
            pl.col("start_ts").is_not_null() & (pl.col("ts") > pl.col("start_ts"))
        )
        totals = matched.group_by("interval_id").agg(
            pl.col("output_tokens").sum().alias("interval_output")
        )
        matched = matched.join(totals, on="interval_id", how="left").filter(
            pl.col("interval_output") > 0
        )
        matched = matched.with_columns(
            (pl.col("delta") * pl.col("output_tokens") / pl.col("interval_output")).alias(
                "attributed"
            )
        )
        per_sess = matched.group_by("session_id").agg(
            pl.col("attributed").sum().alias("attributed")
        )
        # unattributed = within-window positive delta that matched no logged turn.
        # (Cross-reset / null-reset intervals are excluded entirely, not counted here.)
        attributed_total = float(matched["attributed"].sum() or 0.0)
        total_delta = float(intervals["delta"].sum())
        unattributed = max(0.0, total_delta - attributed_total)
        return per_sess, unattributed

    a5, u5 = attribute("util_5h", "resets_5h_iso")
    aw, uw = attribute("util_7d", "resets_7d_iso")
    diagnostics["unattributed_5h"] = u5
    diagnostics["unattributed_7d"] = uw

    out = (
        per_session
        .join(a5.rename({"attributed": "attributed_pct_5h"}), on="session_id", how="left")
        .join(aw.rename({"attributed": "attributed_pct_weekly"}), on="session_id", how="left")
        .with_columns(
            pl.col("attributed_pct_5h").fill_null(0.0),
            pl.col("attributed_pct_weekly").fill_null(0.0),
        )
        .select(list(out_schema.keys()))
    )
    return out, diagnostics


def bin_sessions(
    sessions: pl.DataFrame, x_col: str, y_col: str, n_bins: int
) -> pl.DataFrame:
    """Quantile-bin sessions on x_col (equal count per bin) and aggregate y_col.

    Returns columns bin_median_x, mean_y, std_y, n, sorted by bin_median_x.
    std_y is null for single-member bins.
    """
    schema = {
        "bin_median_x": pl.Float64,
        "mean_y": pl.Float64,
        "std_y": pl.Float64,
        "n": pl.Int64,
    }
    if sessions.is_empty() or x_col not in sessions.columns or y_col not in sessions.columns:
        return pl.DataFrame(schema=schema)
    s = sessions.select(
        pl.col(x_col).cast(pl.Float64).alias("x"),
        pl.col(y_col).cast(pl.Float64).alias("y"),
    ).drop_nulls()
    if s.is_empty():
        return pl.DataFrame(schema=schema)
    n_bins = max(1, min(n_bins, s.height))
    s = s.with_columns(
        (((pl.col("x").rank(method="ordinal") - 1).cast(pl.Int64) * n_bins) // pl.len())
        .alias("bin")
    )
    return (
        s.group_by("bin")
        .agg(
            pl.col("x").median().alias("bin_median_x"),
            pl.col("y").mean().alias("mean_y"),
            pl.col("y").std().alias("std_y"),
            pl.len().cast(pl.Int64).alias("n"),
        )
        .sort("bin_median_x")
        .select(list(schema.keys()))
    )
