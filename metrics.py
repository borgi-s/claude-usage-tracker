"""Pure computation: cost weighting, rolling windows, per-session context curves."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl

import config


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Parse timestamps and add cost-weighted, raw-input, context-prompt columns."""
    if df.is_empty():
        return df
    w = config.COST_WEIGHTS
    priced = (
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
    # Per-model USD: build a small (model -> prices) frame and join (avoids a
    # row-wise python call). price_for handles unknown models + warns.
    models = [m for m in priced["model"].unique().to_list()]
    price_rows = []
    for m in models:
        p = config.price_for(m)
        price_rows.append({"model": m, "_p_in": p["input"], "_p_out": p["output"],
                           "_p_cw": p["cache_write"], "_p_cr": p["cache_read"]})
    price_df = pl.DataFrame(price_rows) if price_rows else pl.DataFrame(
        schema={"model": pl.Utf8, "_p_in": pl.Float64, "_p_out": pl.Float64,
                "_p_cw": pl.Float64, "_p_cr": pl.Float64})
    return (
        priced.join(price_df, on="model", how="left")
        .with_columns(
            (
                (
                    pl.col("input_tokens") * pl.col("_p_in")
                    + pl.col("output_tokens") * pl.col("_p_out")
                    + pl.col("cache_creation_input_tokens") * pl.col("_p_cw")
                    + pl.col("cache_read_input_tokens") * pl.col("_p_cr")
                ) / 1_000_000.0
            ).alias("dollar_cost")
        )
        .drop(["_p_in", "_p_out", "_p_cw", "_p_cr"])
    )


MAX5X_TIER = "default_claude_max_5x"
_RESET_JUMP = {"5h": timedelta(minutes=30), "weekly": timedelta(days=1)}
_DEFAULT_GAP_MIN = {"5h": 15, "weekly": 60}


def _parse_reset(s):
    """Parse an ISO reset string to tz-aware UTC datetime."""
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _window_ids(reset_list, kind: str) -> list[int]:
    """Jitter-tolerant integer window id per sample. New id when the parsed reset
    time jumps forward by more than the kind's tolerance (resets jitter by seconds;
    a real reset jumps ~5h / ~7d)."""
    jump = _RESET_JUMP[kind]
    ids: list[int] = []
    wid = -1
    prev_reset = None
    for r in reset_list:
        rdt = _parse_reset(r)
        if wid < 0:
            wid = 0
        elif rdt is not None and prev_reset is not None and (rdt - prev_reset) > jump:
            wid += 1
        ids.append(wid)
        if rdt is not None:
            prev_reset = rdt
    return ids


def _max5x(log: pl.DataFrame) -> pl.DataFrame:
    """Filter log to Max-5x tier rows only."""
    if log.is_empty() or "rate_limit_tier" not in log.columns:
        return log.head(0)
    return log.filter(pl.col("rate_limit_tier") == MAX5X_TIER)


@dataclass
class CapProjection:
    eta: timedelta | None       # time from `now` until util reaches 1.0
    before_reset: bool          # whether 100% arrives before the window resets


def project_time_to_cap(log: pl.DataFrame, now: datetime, kind: str = "5h") -> CapProjection:
    """Project time until utilization reaches 100% based on burn-rate slope.

    Computes the slope of utilization change over the current (jitter-tolerant) window,
    extrapolates to util=1.0, and returns the ETA. If the projected 100% time is past
    the parsed window reset, before_reset is False.

    Returns CapProjection(eta=None, before_reset=True) when:
    - <2 in-window samples, OR
    - flat or declining utilization (slope <= 0)
    """
    util_col = "util_5h" if kind == "5h" else "util_7d"
    reset_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    mx = _max5x(log).drop_nulls(["sampled_at", util_col]).sort("sampled_at")
    if mx.height < 2:
        return CapProjection(None, True)
    reset_list = mx[reset_col].to_list() if reset_col in mx.columns else [None] * mx.height
    wids = _window_ids(reset_list, kind)
    last_wid = wids[-1]
    cur = [(t, u, r) for t, u, r, w in
           zip(mx["sampled_at"].to_list(), mx[util_col].to_list(), reset_list, wids)
           if w == last_wid]
    if len(cur) < 2:
        return CapProjection(None, True)
    t0, u0, _ = cur[0]
    t1, u1, r1 = cur[-1]
    dt_s = (t1 - t0).total_seconds()
    if dt_s <= 0 or (u1 - u0) <= 0:
        return CapProjection(None, True)
    slope = (u1 - u0) / dt_s              # util per second, > 0
    secs_to_full = (1.0 - u1) / slope
    full_ts = t1 + timedelta(seconds=secs_to_full)
    eta = full_ts - now
    if eta.total_seconds() < 0:
        eta = timedelta(0)
    reset_dt = _parse_reset(r1)
    before_reset = reset_dt is None or full_ts <= reset_dt
    return CapProjection(eta, before_reset)


def reported_util_series(log: pl.DataFrame, kind: str, gap_break_minutes: int | None = None):
    """Return (series, cap_hits) from calibration log.

    series: DataFrame with ts (Datetime) + util_pct (Float64, None at break boundaries)
    cap_hits: DataFrame with ts + util_pct for samples with util >= 0.99

    Max-5x only; breaks at real resets (parsed-time jump) and sampling gaps.
    """
    util_col = "util_5h" if kind == "5h" else "util_7d"
    reset_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    out_schema = {"ts": pl.Datetime("ms", "UTC"), "util_pct": pl.Float64}
    empty = pl.DataFrame(schema=out_schema)
    if log.is_empty() or util_col not in log.columns:
        return empty, empty
    mx = _max5x(log).drop_nulls(["sampled_at", util_col]).sort("sampled_at")
    if mx.is_empty():
        return empty, empty
    gap = timedelta(minutes=gap_break_minutes if gap_break_minutes is not None
                    else _DEFAULT_GAP_MIN[kind])
    jump = _RESET_JUMP[kind]
    ts_list = mx["sampled_at"].to_list()
    util_list = mx[util_col].to_list()
    reset_list = mx[reset_col].to_list() if reset_col in mx.columns else [None] * len(ts_list)

    xs: list = []
    ys: list = []
    prev_ts = None
    prev_reset = None
    for ts, u, r in zip(ts_list, util_list, reset_list):
        rdt = _parse_reset(r)
        if prev_ts is not None:
            gap_hit = (ts - prev_ts) > gap
            reset_hit = (rdt is not None and prev_reset is not None and (rdt - prev_reset) > jump)
            if gap_hit or reset_hit:
                xs.append(ts)
                ys.append(None)
        xs.append(ts)
        ys.append(float(u) * 100.0)
        prev_ts = ts
        if rdt is not None:
            prev_reset = rdt

    series = pl.DataFrame({"ts": xs, "util_pct": ys}, schema=out_schema)
    cap_hits = mx.filter(pl.col(util_col) >= 0.99).select(
        pl.col("sampled_at").alias("ts"),
        (pl.col(util_col) * 100.0).alias("util_pct"),
    )
    return series, cap_hits


def peak_reported(log: pl.DataFrame, kind: str):
    """Return max util over Max-5x samples."""
    util_col = "util_5h" if kind == "5h" else "util_7d"
    mx = _max5x(log)
    if mx.is_empty() or util_col not in mx.columns:
        return None
    vals = mx.drop_nulls(util_col)
    return float(vals[util_col].max()) if not vals.is_empty() else None


def windows_over_threshold(log: pl.DataFrame, kind: str, threshold: float) -> tuple[int, int]:
    """Return (n_windows_over, n_windows_total) by jitter-tolerant window id."""
    util_col = "util_5h" if kind == "5h" else "util_7d"
    reset_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    mx = _max5x(log).drop_nulls(["sampled_at", util_col]).sort("sampled_at")
    if mx.is_empty():
        return 0, 0
    reset_list = mx[reset_col].to_list() if reset_col in mx.columns else [None] * mx.height
    wids = _window_ids(reset_list, kind)
    grouped = (
        mx.with_columns(pl.Series("_wid", wids))
        .group_by("_wid")
        .agg(pl.col(util_col).max().alias("peak"))
    )
    n_total = grouped.height
    n_over = grouped.filter(pl.col("peak") > threshold).height
    return n_over, n_total


def daily_stacked(df: pl.DataFrame, by: str = "is_subagent",
                  value_col: str = "dollar_cost") -> pl.DataFrame:
    """Sum value_col per (UTC) day, pivoted by `by`.

    by="is_subagent" -> columns date, main, subagent.
    by="machine"     -> columns date, <machine values...>; a single 'local' column
                        when df has no 'machine' column (local single-machine cache).
    """
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})
    work = df.with_columns(pl.col("ts").dt.date().alias("date"))
    dim = by
    if by == "machine" and "machine" not in work.columns:
        work = work.with_columns(pl.lit("local").alias("machine"))
    pivoted = (
        work.group_by(["date", dim])
        .agg(pl.col(value_col).sum().alias("v"))
        .pivot(values="v", index="date", on=dim)
        .sort("date")
        .fill_null(0.0)
    )
    if by == "is_subagent":
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
    return pivoted


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


