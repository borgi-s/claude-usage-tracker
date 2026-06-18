"""Persist the latest live utilization snapshot to caps.json (read by the cloud live panel)."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


CAPS_PATH = Path(__file__).parent / "caps.json"


@dataclass
class DerivedCaps:
    """Latest live snapshot. (Name kept for import stability; no longer derives caps.)"""
    sampled_at: Optional[str] = None
    sample_util_5h: Optional[float] = None
    sample_util_7d: Optional[float] = None
    subscription_type: Optional[str] = None
    resets_5h_iso: Optional[str] = None
    resets_7d_iso: Optional[str] = None
    rate_limit_tier: Optional[str] = None


def _empty() -> DerivedCaps:
    return DerivedCaps()


def load_caps() -> DerivedCaps:
    if not CAPS_PATH.exists():
        return _empty()
    try:
        d = json.loads(CAPS_PATH.read_text(encoding="utf-8"))
        known = {f for f in DerivedCaps.__dataclass_fields__}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in known}
        return DerivedCaps(**d)
    except (json.JSONDecodeError, TypeError):
        return _empty()


def save_caps(caps: DerivedCaps) -> None:
    CAPS_PATH.write_text(json.dumps(asdict(caps), indent=2), encoding="utf-8")
