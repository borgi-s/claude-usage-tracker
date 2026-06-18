"""Config constants. Edit values here to recalibrate caps and weights."""
from __future__ import annotations

import warnings
from pathlib import Path


CLAUDE_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
CACHE_PATH = Path(__file__).parent / "cache.parquet"
MANIFEST_PATH = Path(__file__).parent / "cache_manifest.json"


COST_WEIGHTS = {
    "input": 1.0,
    "cache_creation": 1.25,
    "cache_read": 0.1,
    "output": 5.0,
}


MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 200_000


CONTEXT_THRESHOLDS = {"green_max": 0.25, "yellow_max": 0.50}
ABSOLUTE_CONTEXT_REFERENCE = 200_000


LOCAL_TZ = "Europe/Copenhagen"
NIGHT_HOURS = (22, 6)  # local-time start, end of night band (wraps over midnight)

WEEKLY_RESET_WEEKDAY = 6  # Mon=0 ... Sun=6
WEEKLY_RESET_HOUR_LOCAL = 7

# Effective 5h window length. Anthropic publishes 5h but user observation
# suggests the cap behaves like it resets ~30 min sooner.
FIVE_HOUR_WINDOW_HOURS = 4.5


def context_window_for(model: str) -> int:
    if not model:
        return DEFAULT_CONTEXT_WINDOW
    for key, val in MODEL_CONTEXT_WINDOWS.items():
        if model.startswith(key):
            return val
    return DEFAULT_CONTEXT_WINDOW


# USD per 1,000,000 tokens. cache_write = 1.25 * input (5-minute ephemeral, Claude
# Code's default); cache_read = 0.1 * input. Source: Anthropic pricing (2026-06-04).
def _tier(inp: float, out: float) -> dict:
    return {"input": inp, "output": out, "cache_write": inp * 1.25, "cache_read": inp * 0.1}


# Order does not matter for correctness (price_for does longest-prefix), but keep
# specific prefixes readable.
MODEL_PRICING: dict[str, dict] = {
    "claude-fable-5": _tier(10.0, 50.0),
    "claude-opus-4-": _tier(5.0, 25.0),
    "claude-sonnet-4-": _tier(3.0, 15.0),
    "claude-haiku-4-": _tier(1.0, 5.0),
    "claude-3-opus": _tier(15.0, 75.0),
    "claude-3-7-sonnet": _tier(3.0, 15.0),
    "claude-3-5-sonnet": _tier(3.0, 15.0),
    "claude-3-5-haiku": _tier(0.80, 4.0),
    "claude-3-haiku": _tier(0.25, 1.25),
}

_PRICING_FALLBACK = _tier(3.0, 15.0)  # Sonnet-tier
_warned_models: set[str] = set()


def price_for(model: str) -> dict:
    """USD-per-MTok prices for a model id, by longest matching prefix.

    Unlike context_window_for (first-match-in-dict-order), this picks the MOST
    SPECIFIC prefix so e.g. 'claude-fable-5' can't be shadowed. Unknown / <synthetic>
    models fall back to Sonnet-tier and warn once.
    """
    if model:
        best = None
        for prefix in MODEL_PRICING:
            if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        if best is not None:
            return MODEL_PRICING[best]
    if model not in _warned_models:
        _warned_models.add(model)
        warnings.warn(f"price_for: unknown model {model!r}; using Sonnet-tier fallback pricing")
    return _PRICING_FALLBACK
