"""Parse Claude Code JSONL transcripts into a flat row-per-assistant-turn schema."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SUBAGENT_PATH_RE = re.compile(r"[\\/]subagents[\\/]agent-([a-f0-9]+)\.jsonl$", re.IGNORECASE)


@dataclass
class TurnRow:
    timestamp: str
    session_id: str
    subagent_id: str | None
    is_subagent: bool
    project_cwd: str
    model: str
    version: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    source_file: str
    is_rate_limit_error: bool


def _classify_path(path: Path) -> tuple[bool, str | None]:
    m = SUBAGENT_PATH_RE.search(str(path))
    if m:
        return True, m.group(1)
    return False, None


def _extract_usage(obj: dict) -> dict | None:
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage


def _is_rate_limit_error(obj: dict) -> bool:
    if obj.get("type") not in {"api-error", "error"}:
        return False
    err = obj.get("error") or {}
    if isinstance(err, dict):
        etype = (err.get("type") or "").lower()
        if "rate" in etype or "limit" in etype or err.get("status") == 429:
            return True
    msg = obj.get("message") or {}
    if isinstance(msg, dict):
        err2 = msg.get("error") or {}
        if isinstance(err2, dict):
            etype = (err2.get("type") or "").lower()
            if "rate" in etype or "limit" in etype:
                return True
    return False


def iter_rows(path: Path) -> Iterator[TurnRow]:
    is_sub, sub_id = _classify_path(path)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            rate_limited = _is_rate_limit_error(obj)
            usage = _extract_usage(obj)
            if usage is None and not rate_limited:
                continue

            ts = obj.get("timestamp")
            if not ts:
                continue

            session_id = obj.get("sessionId") or ""
            cwd = obj.get("cwd") or ""
            version = obj.get("version") or ""
            model = ""
            if isinstance(obj.get("message"), dict):
                model = obj["message"].get("model") or ""

            yield TurnRow(
                timestamp=ts,
                session_id=session_id,
                subagent_id=sub_id,
                is_subagent=is_sub,
                project_cwd=cwd,
                model=model,
                version=version,
                input_tokens=int((usage or {}).get("input_tokens") or 0),
                output_tokens=int((usage or {}).get("output_tokens") or 0),
                cache_creation_input_tokens=int((usage or {}).get("cache_creation_input_tokens") or 0),
                cache_read_input_tokens=int((usage or {}).get("cache_read_input_tokens") or 0),
                source_file=str(path),
                is_rate_limit_error=rate_limited,
            )


def walk_jsonl(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.jsonl")
