"""Read OAuth token from ~/.claude/.credentials.json and call the usage endpoint."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


CREDS_PATH = Path.home() / ".claude" / ".credentials.json"
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
ANTHROPIC_BETA = "oauth-2025-04-20"


class RateLimited(RuntimeError):
    def __init__(self, msg: str, retry_after_s: Optional[float] = None):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        target = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
        return max(0.0, (target - datetime.now(tz=timezone.utc)).total_seconds())
    except ValueError:
        return None


@dataclass
class UsageBucket:
    utilization: float
    resets_at: Optional[datetime]


@dataclass
class UsageSnapshot:
    five_hour: Optional[UsageBucket]
    seven_day: Optional[UsageBucket]
    sampled_at: datetime
    subscription_type: str
    rate_limit_tier: str


def load_oauth_token() -> tuple[str, dict]:
    """Return (access_token, full_oauth_dict). Raises if file missing or expired."""
    if not CREDS_PATH.exists():
        raise FileNotFoundError(f"Credentials file not found: {CREDS_PATH}")
    raw = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    oauth = raw.get("claudeAiOauth") or {}
    token = oauth.get("accessToken")
    if not token:
        raise RuntimeError("No accessToken in credentials.json")
    expires_at = oauth.get("expiresAt")
    if expires_at and expires_at < int(time.time() * 1000):
        raise RuntimeError(
            "OAuth access token is expired. Run any Claude Code command to refresh it."
        )
    return token, oauth


def _parse_resets_at(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000 if value > 1e12 else value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def fetch_usage(timeout_s: float = 10.0) -> UsageSnapshot:
    token, oauth = load_oauth_token()
    req = urlrequest.Request(
        USAGE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-beta": ANTHROPIC_BETA,
            "Accept": "application/json",
            "User-Agent": "claude-usage-tracker/0.1",
        },
        method="GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 429:
            retry_after = e.headers.get("Retry-After") if e.headers else None
            raise RateLimited(f"Usage endpoint is rate-limited. Retry-After: {retry_after or 'unspecified'}",
                              retry_after_s=_parse_retry_after(retry_after))
        raise RuntimeError(f"HTTP {e.code} from usage endpoint: {e.read().decode('utf-8', 'replace')[:300]}")
    except URLError as e:
        raise RuntimeError(f"Network error reaching usage endpoint: {e.reason}")

    data = json.loads(body)

    def bucket(key: str) -> Optional[UsageBucket]:
        b = data.get(key)
        if not isinstance(b, dict):
            return None
        u = b.get("utilization")
        if u is None:
            return None
        # API returns utilization as 0-100 (percentage). Normalize to 0.0-1.0 fraction.
        return UsageBucket(utilization=float(u) / 100.0, resets_at=_parse_resets_at(b.get("resets_at")))

    return UsageSnapshot(
        five_hour=bucket("five_hour"),
        seven_day=bucket("seven_day"),
        sampled_at=datetime.now(tz=timezone.utc),
        subscription_type=str(oauth.get("subscriptionType") or "unknown"),
        rate_limit_tier=str(oauth.get("rateLimitTier") or "unknown"),
    )
