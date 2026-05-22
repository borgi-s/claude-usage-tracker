"""Config constants. Edit values here to recalibrate caps and weights."""
from __future__ import annotations

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


PRO_CAP_5H_COST_WEIGHTED = 30_000_000
PRO_CAP_WEEKLY_COST_WEIGHTED = 140_000_000

MAX5X_CAP_5H_COST_WEIGHTED = PRO_CAP_5H_COST_WEIGHTED * 5
MAX5X_CAP_WEEKLY_COST_WEIGHTED = PRO_CAP_WEEKLY_COST_WEIGHTED * 5


CAP_DISCLAIMER = (
    "Caps are community-derived estimates. Anthropic does not publish exact "
    "token quotas for subscription plans. Edit config.py to recalibrate."
)


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
