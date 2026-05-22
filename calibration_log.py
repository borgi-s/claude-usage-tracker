"""Append-only log of API calibration samples. Used later to regress real cost weights."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


LOG_PATH = Path(__file__).parent / "calibration_log.parquet"


SCHEMA = {
    "sampled_at": pl.Datetime("ms", "UTC"),
    "util_5h": pl.Float64,
    "util_7d": pl.Float64,
    "burn_5h_cost_weighted": pl.Float64,
    "burn_7d_cost_weighted": pl.Float64,
    "input_5h": pl.Int64,
    "cache_creation_5h": pl.Int64,
    "cache_read_5h": pl.Int64,
    "output_5h": pl.Int64,
    "input_7d": pl.Int64,
    "cache_creation_7d": pl.Int64,
    "cache_read_7d": pl.Int64,
    "output_7d": pl.Int64,
    "subscription_type": pl.Utf8,
    "rate_limit_tier": pl.Utf8,
    "resets_5h_iso": pl.Utf8,
    "resets_7d_iso": pl.Utf8,
}


def append_sample(row: dict) -> None:
    # Fill missing keys from schema with nulls so each row matches SCHEMA shape
    for key in SCHEMA:
        row.setdefault(key, None)
    df = pl.DataFrame([row], schema=SCHEMA)
    if LOG_PATH.exists():
        existing = pl.read_parquet(LOG_PATH)
        # diagonal concat handles older logs missing the new columns
        combined = pl.concat([existing, df], how="diagonal")
    else:
        combined = df
    combined.write_parquet(LOG_PATH)


def load_log() -> pl.DataFrame:
    if not LOG_PATH.exists():
        return pl.DataFrame(schema=SCHEMA)
    return pl.read_parquet(LOG_PATH)


def window_aggregates(df: pl.DataFrame, start_ts: datetime, end_ts: datetime) -> dict:
    """Sum raw token columns for rows in [start_ts, end_ts]."""
    if df.is_empty() or start_ts is None or end_ts is None:
        return {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}
    w = df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))
    if w.is_empty():
        return {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}
    return {
        "input": int(w["input_tokens"].sum()),
        "cache_creation": int(w["cache_creation_input_tokens"].sum()),
        "cache_read": int(w["cache_read_input_tokens"].sum()),
        "output": int(w["output_tokens"].sum()),
    }


def cost_weighted_sum_in_window(df: pl.DataFrame, start_ts: datetime, end_ts: datetime) -> float:
    if df.is_empty() or start_ts is None or end_ts is None:
        return 0.0
    w = df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts))
    if w.is_empty():
        return 0.0
    return float(w["cost_weighted_tokens"].sum())
