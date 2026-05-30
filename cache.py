"""Parquet sidecar with mtime-diff incremental parse."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import polars as pl

import config
from parser import TurnRow, iter_rows, walk_jsonl


ROW_SCHEMA = {
    "timestamp": pl.Utf8,
    "session_id": pl.Utf8,
    "subagent_id": pl.Utf8,
    "is_subagent": pl.Boolean,
    "project_cwd": pl.Utf8,
    "model": pl.Utf8,
    "version": pl.Utf8,
    "input_tokens": pl.Int64,
    "output_tokens": pl.Int64,
    "cache_creation_input_tokens": pl.Int64,
    "cache_read_input_tokens": pl.Int64,
    "source_file": pl.Utf8,
    "is_rate_limit_error": pl.Boolean,
}


def _load_manifest() -> dict[str, float]:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict[str, float]) -> None:
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parse_files(files: list[Path]) -> pl.DataFrame:
    rows: list[dict] = []
    for p in files:
        for row in iter_rows(p):
            rows.append(asdict(row))
    if not rows:
        return pl.DataFrame(schema=ROW_SCHEMA)
    df = pl.DataFrame(rows, schema=ROW_SCHEMA)
    return df


def refresh_cache(root: Path | None = None) -> tuple[pl.DataFrame, dict]:
    """Walk all JSONL under root, reparse anything new/modified, append to cache.parquet."""
    root = root or config.CLAUDE_PROJECTS_ROOT
    manifest = _load_manifest()

    all_files = list(walk_jsonl(root))
    current_mtimes = {str(p): p.stat().st_mtime for p in all_files}

    new_or_changed = [
        Path(p) for p, mt in current_mtimes.items()
        if manifest.get(p) != mt
    ]
    deleted = [p for p in manifest if p not in current_mtimes]

    if config.CACHE_PATH.exists():
        existing = pl.read_parquet(config.CACHE_PATH)
    else:
        existing = pl.DataFrame(schema=ROW_SCHEMA)

    if deleted or new_or_changed:
        if not existing.is_empty():
            stale_files = set(str(p) for p in new_or_changed) | set(deleted)
            existing = existing.filter(~pl.col("source_file").is_in(list(stale_files)))

        fresh = _parse_files(new_or_changed)
        combined = pl.concat([existing, fresh], how="vertical") if not fresh.is_empty() else existing
    else:
        combined = existing

    combined.write_parquet(config.CACHE_PATH)
    _save_manifest(current_mtimes)

    stats = {
        "total_files": len(all_files),
        "new_or_changed": len(new_or_changed),
        "deleted": len(deleted),
        "total_rows": combined.height,
    }
    return combined, stats


def load_cache() -> pl.DataFrame:
    if not config.CACHE_PATH.exists():
        return pl.DataFrame(schema=ROW_SCHEMA)
    return pl.read_parquet(config.CACHE_PATH)


def merge_cache_parquets(prefix_paths: dict[str, Path], out_path: Path) -> int:
    """Merge per-machine cache.parquet files into one frame at out_path.

    For each (prefix, path) in prefix_paths, read the parquet and add a `machine`
    column equal to the prefix, then diagonal-concat all of them (diagonal so two
    machines on different agent versions can't break the concat). Writes the merged
    frame to out_path and returns its row count. With no inputs, writes a
    schema-only (empty) cache carrying the `machine` column so load_cache() and the
    downstream pipeline still work.
    """
    frames = []
    for prefix, path in prefix_paths.items():
        df = pl.read_parquet(path).with_columns(pl.lit(prefix).alias("machine"))
        frames.append(df)
    if frames:
        merged = pl.concat(frames, how="diagonal")
    else:
        merged = pl.DataFrame(schema={**ROW_SCHEMA, "machine": pl.Utf8})
    merged.write_parquet(out_path)
    return merged.height


if __name__ == "__main__":
    _, stats = refresh_cache()
    print(stats)
