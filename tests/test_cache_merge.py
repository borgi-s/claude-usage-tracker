"""Unit tests for cache.merge_cache_parquets — multi-machine concat + machine stamp."""
from __future__ import annotations

from pathlib import Path

import polars as pl

import cache


def _write_cache(path: Path, session_ids: list[str]) -> None:
    """Write a minimal cache.parquet with the real ROW_SCHEMA columns."""
    n = len(session_ids)
    df = pl.DataFrame(
        {
            "timestamp": [f"2026-05-30T10:0{i}:00.000Z" for i in range(n)],
            "session_id": session_ids,
            "subagent_id": [None] * n,
            "is_subagent": [False] * n,
            "project_cwd": ["/p"] * n,
            "model": ["claude-opus-4-7"] * n,
            "version": ["1.0"] * n,
            "input_tokens": [1] * n,
            "output_tokens": [2] * n,
            "cache_creation_input_tokens": [0] * n,
            "cache_read_input_tokens": [0] * n,
            "source_file": ["a.jsonl"] * n,
            "is_rate_limit_error": [False] * n,
        },
        schema=cache.ROW_SCHEMA,
    )
    df.write_parquet(path)


def test_merge_stamps_machine_and_unions_rows(tmp_path: Path):
    win = tmp_path / "cache__borgi.parquet"
    lin = tmp_path / "cache__borgi-linux.parquet"
    _write_cache(win, ["w1", "w2"])
    _write_cache(lin, ["l1"])
    out = tmp_path / "cache.parquet"

    rows = cache.merge_cache_parquets({"borgi": win, "borgi-linux": lin}, out)

    assert rows == 3
    merged = pl.read_parquet(out)
    assert merged.height == 3
    assert "machine" in merged.columns
    assert set(merged["machine"].to_list()) == {"borgi", "borgi-linux"}
    # original columns preserved
    for col in cache.ROW_SCHEMA:
        assert col in merged.columns
    # machine value tracks the source prefix
    win_rows = merged.filter(pl.col("machine") == "borgi")
    assert set(win_rows["session_id"].to_list()) == {"w1", "w2"}


def test_merge_empty_writes_schema_only_cache(tmp_path: Path):
    out = tmp_path / "cache.parquet"
    rows = cache.merge_cache_parquets({}, out)
    assert rows == 0
    merged = pl.read_parquet(out)
    assert merged.height == 0
    assert "machine" in merged.columns
    for col in cache.ROW_SCHEMA:
        assert col in merged.columns
