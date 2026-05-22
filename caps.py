"""Derive plan caps from a live utilization reading and persist them to caps.json."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Optional
from zoneinfo import ZoneInfo

import polars as pl

import config


CAPS_PATH = Path(__file__).parent / "caps.json"
MIN_UTILIZATION_FOR_CALIBRATION = 0.02  # below this, derived cap is too noisy


@dataclass
class DerivedCaps:
    max5x_5h: Optional[float]
    max5x_weekly: Optional[float]
    pro_5h: Optional[float]
    pro_weekly: Optional[float]
    sampled_at: Optional[str]
    sample_burn_5h: Optional[float]
    sample_burn_7d: Optional[float]
    sample_util_5h: Optional[float]
    sample_util_7d: Optional[float]
    subscription_type: Optional[str]
    resets_5h_iso: Optional[str] = None
    resets_7d_iso: Optional[str] = None
    rate_limit_tier: Optional[str] = None


def _empty() -> DerivedCaps:
    return DerivedCaps(None, None, None, None, None, None, None, None, None, None, None, None, None)


def load_caps() -> DerivedCaps:
    if not CAPS_PATH.exists():
        return _empty()
    try:
        d = json.loads(CAPS_PATH.read_text(encoding="utf-8"))
        known = {f.name for f in DerivedCaps.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        return DerivedCaps(**d)
    except (json.JSONDecodeError, TypeError):
        return _empty()


def save_caps(caps: DerivedCaps) -> None:
    CAPS_PATH.write_text(json.dumps(asdict(caps), indent=2), encoding="utf-8")


def derive_from_reading(
    burn_5h: float,
    util_5h: Optional[float],
    burn_7d: float,
    util_7d: Optional[float],
    subscription_type: str,
    resets_5h_iso: Optional[str] = None,
    resets_7d_iso: Optional[str] = None,
    rate_limit_tier: Optional[str] = None,
) -> DerivedCaps:
    """Implied cap = burn / utilization. Skip derivation if utilization is too small."""
    sampled_at = datetime.now(tz=timezone.utc).isoformat()

    max5x_5h: Optional[float] = None
    if util_5h is not None and util_5h >= MIN_UTILIZATION_FOR_CALIBRATION and burn_5h > 0:
        max5x_5h = burn_5h / util_5h
    max5x_weekly: Optional[float] = None
    if util_7d is not None and util_7d >= MIN_UTILIZATION_FOR_CALIBRATION and burn_7d > 0:
        max5x_weekly = burn_7d / util_7d

    return DerivedCaps(
        max5x_5h=max5x_5h,
        max5x_weekly=max5x_weekly,
        pro_5h=(max5x_5h / 5) if max5x_5h else None,
        pro_weekly=(max5x_weekly / 5) if max5x_weekly else None,
        sampled_at=sampled_at,
        sample_burn_5h=burn_5h,
        sample_burn_7d=burn_7d,
        sample_util_5h=util_5h,
        sample_util_7d=util_7d,
        subscription_type=subscription_type,
        resets_5h_iso=resets_5h_iso,
        resets_7d_iso=resets_7d_iso,
        rate_limit_tier=rate_limit_tier,
    )


CONTINUOUS_MIN_UTIL = 0.10
CONTINUOUS_MAX_UTIL = 0.95
CONTINUOUS_LOOKBACK_HOURS = 72


def implied_cap_series(log: pl.DataFrame, min_util: float = CONTINUOUS_MIN_UTIL,
                       max_util: float = CONTINUOUS_MAX_UTIL) -> pl.DataFrame:
    """Return per-sample implied caps with valid utilization filter applied.

    Columns: sampled_at, implied_5h, implied_weekly, util_5h, util_7d.
    Rows where either utilization is out of range are dropped per series via nulls.
    """
    if log.is_empty():
        return pl.DataFrame(schema={
            "sampled_at": pl.Datetime("ms", "UTC"),
            "implied_5h": pl.Float64,
            "implied_weekly": pl.Float64,
            "util_5h": pl.Float64,
            "util_7d": pl.Float64,
        })
    return log.select(
        pl.col("sampled_at"),
        pl.when((pl.col("util_5h") >= min_util) & (pl.col("util_5h") <= max_util))
          .then(pl.col("burn_5h_cost_weighted") / pl.col("util_5h"))
          .otherwise(None).alias("implied_5h"),
        pl.when((pl.col("util_7d") >= min_util) & (pl.col("util_7d") <= max_util))
          .then(pl.col("burn_7d_cost_weighted") / pl.col("util_7d"))
          .otherwise(None).alias("implied_weekly"),
        pl.col("util_5h"),
        pl.col("util_7d"),
    )


def derive_continuous_caps(log: pl.DataFrame, snap_metadata: Optional[dict] = None,
                           lookback_hours: float = CONTINUOUS_LOOKBACK_HOURS) -> DerivedCaps:
    """Aggregate implied caps from recent samples in the calibration log.

    snap_metadata may contain keys: subscription_type, rate_limit_tier,
    resets_5h_iso, resets_7d_iso, sample_util_5h, sample_util_7d,
    sample_burn_5h, sample_burn_7d, sampled_at — used to overlay the most
    recent metadata onto the aggregated caps.
    """
    if log.is_empty():
        return _empty()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)
    recent = log.filter(pl.col("sampled_at") >= cutoff)
    if recent.is_empty():
        recent = log

    implied = implied_cap_series(recent).drop_nulls("implied_5h")
    implied_w = implied_cap_series(recent).drop_nulls("implied_weekly")

    max5x_5h = float(implied["implied_5h"].median()) if not implied.is_empty() else None
    max5x_weekly = float(implied_w["implied_weekly"].median()) if not implied_w.is_empty() else None

    meta = snap_metadata or {}
    return DerivedCaps(
        max5x_5h=max5x_5h,
        max5x_weekly=max5x_weekly,
        pro_5h=(max5x_5h / 5) if max5x_5h else None,
        pro_weekly=(max5x_weekly / 5) if max5x_weekly else None,
        sampled_at=meta.get("sampled_at") or datetime.now(tz=timezone.utc).isoformat(),
        sample_burn_5h=meta.get("sample_burn_5h"),
        sample_burn_7d=meta.get("sample_burn_7d"),
        sample_util_5h=meta.get("sample_util_5h"),
        sample_util_7d=meta.get("sample_util_7d"),
        subscription_type=meta.get("subscription_type"),
        resets_5h_iso=meta.get("resets_5h_iso"),
        resets_7d_iso=meta.get("resets_7d_iso"),
        rate_limit_tier=meta.get("rate_limit_tier"),
    )


def cap_series(log: pl.DataFrame, kind: str, min_burn: float = 1_000_000.0) -> pl.DataFrame:
    """Return sorted (sampled_at, cap) for the given kind ('5h' or 'weekly').

    Filters out samples with implausibly low burn (≤ min_burn cost-weighted tokens)
    so noisy near-zero-burn samples don't contaminate the time-varying calibration.
    """
    schema = {"sampled_at": pl.Datetime("ms", "UTC"), "cap": pl.Float64}
    if log.is_empty():
        return pl.DataFrame(schema=schema)
    implied = implied_cap_series(log)
    col = "implied_5h" if kind == "5h" else "implied_weekly"
    burn_col = "burn_5h_cost_weighted" if kind == "5h" else "burn_7d_cost_weighted"
    return (
        implied.drop_nulls(col)
        .join(log.select(["sampled_at", burn_col]), on="sampled_at", how="left")
        .filter(pl.col(burn_col) >= min_burn)
        .select(pl.col("sampled_at"), pl.col(col).alias("cap"))
        .sort("sampled_at")
    )


def attach_time_varying_cap(
    df: pl.DataFrame,
    ts_col: str,
    caps_df: pl.DataFrame,
    fallback_cap: float,
) -> pl.DataFrame:
    """Asof-join nearest-in-time cap from caps_df onto df. Falls back to fallback_cap
    when caps_df is empty or for rows outside its coverage."""
    if caps_df.is_empty():
        return df.with_columns(pl.lit(fallback_cap).cast(pl.Float64).alias("cap"))
    return (
        df.sort(ts_col)
        .join_asof(
            caps_df.sort("sampled_at"),
            left_on=ts_col, right_on="sampled_at",
            strategy="nearest",
        )
        .with_columns(pl.col("cap").fill_null(fallback_cap))
    )


def effective_caps() -> tuple[float, float, str]:
    """Return (pro_5h, pro_weekly, source_label).

    Source preference: derived caps if present and non-null, else config defaults.
    """
    d = load_caps()
    if d.pro_5h and d.pro_weekly:
        return d.pro_5h, d.pro_weekly, f"calibrated {d.sampled_at[:10]}"
    return (
        float(config.PRO_CAP_5H_COST_WEIGHTED),
        float(config.PRO_CAP_WEEKLY_COST_WEIGHTED),
        "config.py default",
    )


# ---------------------------------------------------------------------------
# Hour-of-day cap model
# ---------------------------------------------------------------------------

def _per_hour_medians(
    log: pl.DataFrame, kind: str, min_burn: float, tz: ZoneInfo,
) -> list[Optional[float]]:
    """Raw median implied cap per local-hour bin (24 entries, None where empty)."""
    implied = implied_cap_series(log)
    col = "implied_5h" if kind == "5h" else "implied_weekly"
    burn_col = "burn_5h_cost_weighted" if kind == "5h" else "burn_7d_cost_weighted"

    valid = (
        implied.drop_nulls(col)
        .join(log.select(["sampled_at", burn_col]), on="sampled_at", how="left")
        .filter(pl.col(burn_col) >= min_burn)
    )
    if valid.is_empty():
        return [None] * 24

    samples_by_hour: dict[int, list[float]] = {h: [] for h in range(24)}
    for ts, cap in zip(valid["sampled_at"].to_list(), valid[col].to_list()):
        h = ts.astimezone(tz).hour
        samples_by_hour[h].append(float(cap))

    raw: list[Optional[float]] = [None] * 24
    for h, samples in samples_by_hour.items():
        if samples:
            raw[h] = float(median(samples))
    return raw


def _smooth_rolling_circular(raw: list[Optional[float]], window: int = 3) -> list[Optional[float]]:
    """3-bin (default) circular rolling median; only uses non-null neighbors."""
    n = len(raw)
    half = window // 2
    out: list[Optional[float]] = [None] * n
    for i in range(n):
        neighbors: list[float] = []
        for offset in range(-half, half + 1):
            j = (i + offset) % n
            if raw[j] is not None:
                neighbors.append(raw[j])  # type: ignore[arg-type]
        if neighbors:
            out[i] = float(median(neighbors))
    return out


def _interpolate_empty_circular(smoothed: list[Optional[float]]) -> list[float]:
    """Linear interpolation across empty bins, circular wrap."""
    n = len(smoothed)
    filled = [i for i, v in enumerate(smoothed) if v is not None]
    if not filled:
        return [0.0] * n  # all empty
    out: list[float] = []
    for h in range(n):
        if smoothed[h] is not None:
            out.append(float(smoothed[h]))  # type: ignore[arg-type]
            continue
        # Search forward and backward for nearest filled
        prev_h = prev_dist = None
        next_h = next_dist = None
        for offset in range(1, n + 1):
            ph = (h - offset) % n
            if smoothed[ph] is not None:
                prev_h, prev_dist = ph, offset
                break
        for offset in range(1, n + 1):
            nh = (h + offset) % n
            if smoothed[nh] is not None:
                next_h, next_dist = nh, offset
                break
        if prev_h is not None and next_h is not None:
            total = prev_dist + next_dist
            out.append(
                smoothed[prev_h] * (next_dist / total)  # type: ignore[operator]
                + smoothed[next_h] * (prev_dist / total)
            )
        elif prev_h is not None:
            out.append(float(smoothed[prev_h]))  # type: ignore[arg-type]
        elif next_h is not None:
            out.append(float(smoothed[next_h]))  # type: ignore[arg-type]
        else:
            out.append(0.0)
    return out


def hour_of_day_cap_series(
    log: pl.DataFrame,
    kind: str = "5h",
    min_burn: float = 1_000_000.0,
    tz_name: Optional[str] = None,
) -> list[float]:
    """Build a 24-element hour-of-day cap function for the given kind ('5h' or 'weekly').

    - Bin samples by local hour-of-day, take median per bin.
    - Smooth with 3-hour rolling median (circular).
    - Linearly interpolate across empty bins (circular).
    Returns a list of 24 caps. If log is empty, returns 24 zeros.
    """
    if log.is_empty():
        return [0.0] * 24
    tz = ZoneInfo(tz_name or config.LOCAL_TZ)
    raw = _per_hour_medians(log, kind, min_burn, tz)
    smoothed = _smooth_rolling_circular(raw, window=3)
    return _interpolate_empty_circular(smoothed)


def hour_of_day_sample_counts(
    log: pl.DataFrame,
    kind: str = "5h",
    min_burn: float = 1_000_000.0,
    tz_name: Optional[str] = None,
) -> list[int]:
    """Return number of valid samples in each of the 24 hour-of-day bins."""
    if log.is_empty():
        return [0] * 24
    tz = ZoneInfo(tz_name or config.LOCAL_TZ)
    implied = implied_cap_series(log)
    col = "implied_5h" if kind == "5h" else "implied_weekly"
    burn_col = "burn_5h_cost_weighted" if kind == "5h" else "burn_7d_cost_weighted"
    valid = (
        implied.drop_nulls(col)
        .join(log.select(["sampled_at", burn_col]), on="sampled_at", how="left")
        .filter(pl.col(burn_col) >= min_burn)
    )
    counts = [0] * 24
    for ts in valid["sampled_at"].to_list():
        h = ts.astimezone(tz).hour
        counts[h] += 1
    return counts


def global_cap_from_anchors(
    log: pl.DataFrame,
    cache_df: pl.DataFrame,
    kind: str,
    gap_hours: float,
    min_util: float = 0.95,
    value_col: str = "output_tokens",
) -> tuple[Optional[float], int]:
    """Median cap such that the chart's window-cumulative at 100% anchor moments = 100%.

    For each ≥min_util anchor:
      - find the chart-detected window containing the anchor's timestamp
        (gap-based, same logic as five_hour_burn_since_reset)
      - compute total burn in that window up to (and including) the anchor moment,
        summing the column ``value_col``
      - implied_cap = burn_at_anchor / util_at_anchor
    Return (median_cap, n_anchors).

    Default ``value_col="output_tokens"`` because Anthropic's 5h cap is metered on
    output (verified empirically: two 100% anchors at the same output volume but
    very different cost-weighted volumes due to model mix). Pass a different
    column if you want to calibrate against a different aggregate.
    """
    if log.is_empty() or cache_df.is_empty():
        return None, 0
    if value_col not in cache_df.columns:
        return None, 0
    util_col = "util_5h" if kind == "5h" else "util_7d"
    if util_col not in log.columns:
        return None, 0
    anchors = log.filter(
        (pl.col(util_col) >= min_util) & (pl.col(util_col) <= 1.01)
    )
    if anchors.is_empty():
        return None, 0

    cache_sorted = cache_df.sort("ts")
    ts_list = cache_sorted["ts"].to_list()
    val_list = cache_sorted[value_col].to_list()
    gap = timedelta(hours=gap_hours) if kind == "5h" else timedelta(days=7)

    implied: list[float] = []
    for row in anchors.iter_rows(named=True):
        anchor_ts = row["sampled_at"]
        if anchor_ts is None:
            continue
        if anchor_ts.tzinfo is None:
            anchor_ts = anchor_ts.replace(tzinfo=timezone.utc)
        util_anth = row[util_col]

        current_start = None
        last_ts = None
        burn_in_window = 0.0
        for ts, v in zip(ts_list, val_list):
            if ts > anchor_ts:
                break
            if current_start is None:
                current_start = ts
            elif (ts - last_ts) >= gap or (ts - current_start) >= gap:
                current_start = ts
                burn_in_window = 0.0
            burn_in_window += float(v)
            last_ts = ts

        if burn_in_window > 0 and util_anth > 0:
            implied.append(burn_in_window / util_anth)

    if not implied:
        return None, 0
    return float(median(implied)), len(implied)


def calibrate_hourly_to_log(
    hourly_cap: list[float],
    log: pl.DataFrame,
    cache_df: pl.DataFrame,
    kind: str,
    fallback_cap: float,
    window_hours: float,
    min_util: float = 0.95,
    max_util: float = 1.01,
) -> tuple[list[float], int]:
    """Scale hourly_cap so cumulative shares match measured utilizations at sample times.

    For each near-100% anchor sample with util in [min_util, max_util]:
      predicted_share = sum_over_window(cost_weighted / hourly_cap[hour_of_row])
      ratio = predicted_share / util
    Apply median(ratios) as a global multiplicative scale to hourly_cap.

    By default uses only ≥ 95% anchors (lowest-noise samples — closest to ground truth).
    Falls back to using all samples ≥ 0.50 util if no near-100% anchors exist.

    Returns (scaled_caps, n_samples_used).
    """
    if (
        not hourly_cap or all(c <= 0 for c in hourly_cap)
        or log.is_empty() or cache_df.is_empty()
    ):
        return hourly_cap, 0

    util_col = "util_5h" if kind == "5h" else "util_7d"
    resets_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    window_delta = timedelta(hours=window_hours) if kind == "5h" else timedelta(days=7)

    if util_col not in log.columns:
        return hourly_cap, 0

    valid = log.filter(
        (pl.col(util_col) >= min_util) & (pl.col(util_col) <= max_util)
    )
    # Fallback: if no near-100% anchors, broaden to >=0.50 util
    if valid.is_empty():
        valid = log.filter(
            (pl.col(util_col) >= 0.50) & (pl.col(util_col) <= max_util)
        )
    if valid.is_empty():
        return hourly_cap, 0

    ratios: list[float] = []
    for row in valid.iter_rows(named=True):
        util_anth = row[util_col]
        sample_ts = row["sampled_at"]
        if sample_ts is None:
            continue
        if sample_ts.tzinfo is None:
            sample_ts = sample_ts.replace(tzinfo=timezone.utc)

        # Determine the window the API util refers to
        resets_iso = row.get(resets_col) if resets_col in log.columns else None
        if resets_iso:
            try:
                window_end = datetime.fromisoformat(resets_iso)
                if window_end.tzinfo is None:
                    window_end = window_end.replace(tzinfo=timezone.utc)
                window_start = window_end - window_delta
            except (ValueError, TypeError):
                window_start = sample_ts - window_delta
        else:
            window_start = sample_ts - window_delta

        in_window = cache_df.filter(
            (pl.col("ts") >= window_start) & (pl.col("ts") <= sample_ts)
        )
        if in_window.is_empty():
            continue

        with_caps = attach_hour_of_day_cap(in_window, "ts", hourly_cap, fallback_cap)
        predicted_share = float(
            (with_caps["cost_weighted_tokens"] / with_caps["cap"]).sum()
        )
        if predicted_share > 0:
            ratios.append(predicted_share / util_anth)

    if not ratios:
        return hourly_cap, 0

    scale = float(median(ratios))
    return [c * scale for c in hourly_cap], len(ratios)


def attach_hour_of_day_cap(
    df: pl.DataFrame,
    ts_col: str,
    hourly_cap: list[float],
    fallback_cap: float,
    tz_name: Optional[str] = None,
) -> pl.DataFrame:
    """Add a 'cap' column to df by looking up hourly_cap[local_hour(ts)]."""
    if not hourly_cap or all(c == 0 for c in hourly_cap):
        return df.with_columns(pl.lit(fallback_cap).cast(pl.Float64).alias("cap"))
    tz = ZoneInfo(tz_name or config.LOCAL_TZ)
    caps_for_rows: list[float] = []
    for ts in df[ts_col].to_list():
        h = ts.astimezone(tz).hour
        c = hourly_cap[h]
        caps_for_rows.append(c if c > 0 else fallback_cap)
    return df.with_columns(pl.Series("cap", caps_for_rows, dtype=pl.Float64))
