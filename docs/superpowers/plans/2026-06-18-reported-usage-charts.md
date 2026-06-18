# Reported usage charts + USD daily burn — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two reconstructed (cap-calibrated) usage charts with charts that plot Anthropic's *reported* `util_5h`/`util_7d` straight from the log, switch daily burn to real USD stacked by machine, delete the cap-calibration machinery, and add a "time to 100%" live projection.

**Architecture:** The reported charts read `calibration_log.parquet` (account-wide, Max-5x-filtered); daily burn + the session table read the merged `cache.parquet`. A new `MODEL_PRICING` table drives a per-row `dollar_cost` column. The whole anchor/hour-of-day/continuous cap-derivation stack in `caps.py`, the cumulative/share/window-length helpers in `metrics.py`, and `app_cache.calibrate` are deleted. `caps.json` keeps only a live snapshot (util/resets/plan) for the cloud live panel.

**Tech Stack:** Python 3.12, Polars, Plotly, Streamlit, pytest.

**Design spec:** `docs/superpowers/specs/2026-06-18-reported-usage-charts-design.md` (read it — it has the verified data realities and rationale).

## Global Constraints

- **Working interpreter:** the project `.venv` is broken (built from a deleted Anaconda env). Until repaired, run Python/pytest via the standalone interpreter with the venv packages on the path. In this plan, **`PYRUN`** means:
  `PYTHONIOENCODING=utf-8 PYTHONPATH=".venv/Lib/site-packages" "/c/Users/borgi/AppData/Local/Programs/Python/Python312/python.exe"`
  So a test run is `PYRUN -m pytest tests/test_x.py -v`. Task 0 establishes/repairs this; every later `Run:` uses `PYRUN`.
- **Timestamps:** all internal datetimes are UTC; display in `Europe/Copenhagen` via `config.LOCAL_TZ`.
- **Commits:** NO `Co-Authored-By: Claude` and NO "Generated with Claude Code" attribution in any commit message (per `CLAUDE.md`).
- **Reset columns carry jitter** (`resets_5h_iso`/`resets_7d_iso` are ~unique per sample). Never compare them by raw string equality — parse to datetime and compare with a tolerance. Reuse the pattern already in `metrics.session_cost_attribution` (`metrics.py:746-764`).
- **`machine` column exists only on the cloud merged cache** (`cache.merge_cache_parquets`), never on the local `cache.parquet`. Every machine-consuming path must guard `"machine" in df.columns` and fall back to a single `"local"` series.
- **Pricing is API-equivalent cost** (what the usage would cost at pay-as-you-go API rates), not the subscription bill — matches `ccusage`.
- **Cap calibration must not be reintroduced** (see the "do NOT regress" history in `CLAUDE.md`).

---

### Task 0: Establish a working test interpreter

**Files:** none (environment only)

**Interfaces:**
- Produces: a runnable `PYRUN -m pytest` command used by every later task.

- [ ] **Step 1: Confirm the standalone interpreter can import the stack and run pytest**

Run:
```bash
cd "C:/Users/borgi/Documents/claude-usage-tracker" && PYTHONIOENCODING=utf-8 PYTHONPATH=".venv/Lib/site-packages" "/c/Users/borgi/AppData/Local/Programs/Python/Python312/python.exe" -m pytest tests/ -q
```
Expected: pytest collects and runs the existing suite (some tests will be deleted later; right now they should pass or at least collect). If `pytest` is missing from `.venv/Lib/site-packages`, install into the venv site-packages dir:
```bash
PYTHONIOENCODING=utf-8 "/c/Users/borgi/AppData/Local/Programs/Python/Python312/python.exe" -m pip install --target ".venv/Lib/site-packages" pytest
```

- [ ] **Step 2: Record the interpreter in the plan's `PYRUN` and proceed** (no commit — environment only).

---

### Task 1: `MODEL_PRICING` table + `price_for` in config

**Files:**
- Modify: `config.py`
- Test: `tests/test_pricing.py` (create)

**Interfaces:**
- Produces:
  - `config.MODEL_PRICING: dict[str, dict]` — keys are model-id prefixes; each value `{"input": float, "output": float, "cache_write": float, "cache_read": float}` in USD per 1,000,000 tokens.
  - `config.price_for(model: str) -> dict` — explicit **longest-prefix** match; unknown/`<synthetic>` → Sonnet-tier prices, and logs a warning once per unknown model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pricing.py`:
```python
import config


def test_longest_prefix_wins():
    # claude-fable-5 must not be shadowed by a shorter prefix
    assert config.price_for("claude-fable-5")["input"] == 10.0
    assert config.price_for("claude-opus-4-7")["input"] == 5.0
    assert config.price_for("claude-sonnet-4-6")["output"] == 15.0
    assert config.price_for("claude-haiku-4-5")["input"] == 1.0


def test_cache_multipliers():
    p = config.price_for("claude-opus-4-7")
    assert p["cache_write"] == 6.25   # 1.25 * 5
    assert p["cache_read"] == 0.5     # 0.1 * 5


def test_historical_models_present():
    assert config.price_for("claude-3-opus")["input"] == 15.0
    assert config.price_for("claude-3-5-sonnet")["input"] == 3.0


def test_unknown_falls_back_to_sonnet_and_warns(recwarn):
    p = config.price_for("totally-unknown-model")
    assert p["input"] == 3.0  # Sonnet-tier fallback
    assert any("unknown" in str(w.message).lower() or "totally-unknown" in str(w.message)
               for w in recwarn.list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_pricing.py -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'price_for'`.

- [ ] **Step 3: Implement in `config.py`** (add near the other token/weight constants)

```python
import warnings

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_pricing.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_pricing.py
git commit -m "feat(config): add MODEL_PRICING table and price_for (USD per MTok)"
```

---

### Task 2: `dollar_cost` column in `metrics.add_derived`

**Files:**
- Modify: `metrics.py:50-82` (`add_derived`)
- Test: `tests/test_dollar_cost.py` (create)

**Interfaces:**
- Consumes: `config.price_for` (Task 1).
- Produces: `add_derived` output gains a `dollar_cost: Float64` column = per-row USD across the four token types using `price_for(model)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dollar_cost.py`:
```python
import polars as pl
import metrics
from cache import ROW_SCHEMA


def _row(model, inp=0, out=0, cw=0, cr=0):
    return {
        "timestamp": "2026-05-23T10:00:00.000Z", "session_id": "s", "subagent_id": None,
        "is_subagent": False, "project_cwd": "/p", "model": model, "version": "1",
        "input_tokens": inp, "output_tokens": out,
        "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr,
        "source_file": "f", "is_rate_limit_error": False,
    }


def test_dollar_cost_matches_hand_calc():
    df = pl.DataFrame([_row("claude-opus-4-7", inp=1_000_000, out=1_000_000,
                            cw=1_000_000, cr=1_000_000)], schema=ROW_SCHEMA)
    out = metrics.add_derived(df)
    # opus: in 5, out 25, cache_write 6.25, cache_read 0.5  -> 36.75 for 1M each
    assert abs(out["dollar_cost"][0] - 36.75) < 1e-6


def test_unknown_model_uses_sonnet_fallback():
    df = pl.DataFrame([_row("mystery", inp=1_000_000)], schema=ROW_SCHEMA)
    out = metrics.add_derived(df)
    assert abs(out["dollar_cost"][0] - 3.0) < 1e-6  # sonnet input price
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_dollar_cost.py -v`
Expected: FAIL — `dollar_cost` column not found.

- [ ] **Step 3: Implement** — add a `dollar_cost` column inside `add_derived`'s second `.with_columns(...)` block. Because pricing is per-model, compute it with a map over distinct models. Insert after the `raw_total_tokens` alias (still inside the same `.with_columns(...)`):

```python
            # dollar_cost added below via a per-model price join (see note).
```

Then, replace the trailing `.filter(...).sort("ts")` chain so pricing is joined in. Concretely, restructure the return of `add_derived` to:

```python
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
```

Keep the early `if df.is_empty(): return df` guard at the top unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_dollar_cost.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_dollar_cost.py
git commit -m "feat(metrics): add per-row dollar_cost column to add_derived"
```

---

### Task 3: Reported-util series + window helpers (jitter-tolerant)

**Files:**
- Modify: `metrics.py` (add new functions; suggested location: after `add_derived`)
- Test: `tests/test_reported_util.py` (create)

**Interfaces:**
- Produces:
  - `metrics._parse_reset(s: str | None) -> datetime | None` — parse an ISO reset string to tz-aware UTC datetime.
  - `metrics._window_ids(ts_list, reset_list, kind) -> list[int]` — jitter-tolerant integer window id per sample (new id when parsed reset jumps > 30 min for `"5h"`, > 1 day for `"weekly"`).
  - `metrics.reported_util_series(log, kind, gap_break_minutes=None) -> tuple[pl.DataFrame, pl.DataFrame]` — returns `(series, cap_hits)`. `series` has columns `ts` (Datetime) + `util_pct` (Float64, **None at break boundaries**); Max-5x only; breaks at real resets (parsed-time jump) and sampling gaps. `cap_hits` has `ts` + `util_pct` for samples with util ≥ 0.99.
  - `metrics.peak_reported(log, kind) -> float | None` — max util over Max-5x samples.
  - `metrics.windows_over_threshold(log, kind, threshold) -> tuple[int, int]` — `(n_windows_over, n_windows_total)` by jitter-tolerant window id.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reported_util.py`:
```python
from datetime import datetime, timedelta, timezone
import polars as pl
import pytest
import metrics

LOG_SCHEMA = {  # subset of calibration_log SCHEMA that these helpers read
    "sampled_at": pl.Datetime("ms", "UTC"),
    "util_5h": pl.Float64, "util_7d": pl.Float64,
    "resets_5h_iso": pl.Utf8, "resets_7d_iso": pl.Utf8,
    "rate_limit_tier": pl.Utf8,
}


def _mk(rows):
    return pl.DataFrame(rows, schema=LOG_SCHEMA)


def test_jitter_does_not_break_but_real_reset_does():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset_a = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)   # window A end
    reset_b = datetime(2026, 5, 23, 17, 0, tzinfo=timezone.utc)   # window B end (~5h later)
    rows = []
    # Window A: three samples, each resets_5h_iso jittered by a few seconds
    for i, jit in enumerate((0, 3, 7)):
        rows.append({
            "sampled_at": base + timedelta(minutes=5 * i),
            "util_5h": 0.1 * (i + 1), "util_7d": 0.2,
            "resets_5h_iso": (reset_a + timedelta(seconds=jit)).isoformat(),
            "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x",
        })
    # Window B: one sample, reset jumps ~5h forward -> real reset
    rows.append({
        "sampled_at": base + timedelta(minutes=20),
        "util_5h": 0.05, "util_7d": 0.2,
        "resets_5h_iso": reset_b.isoformat(),
        "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x",
    })
    series, _ = metrics.reported_util_series(_mk(rows), "5h", gap_break_minutes=60)
    ys = series["util_pct"].to_list()
    # exactly one None (the real reset), none among the jittered window-A samples
    assert ys.count(None) == 1
    assert ys[3] is None                              # the break is the real reset
    assert ys[:3] == pytest.approx([10.0, 20.0, 30.0])  # window A continuous (float-tolerant)


def test_sampling_gap_breaks():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc).isoformat()
    rows = [
        {"sampled_at": base, "util_5h": 0.1, "util_7d": 0.0, "resets_5h_iso": reset,
         "resets_7d_iso": None, "rate_limit_tier": "default_claude_max_5x"},
        {"sampled_at": base + timedelta(hours=40), "util_5h": 0.2, "util_7d": 0.0,
         "resets_5h_iso": reset, "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},
    ]
    series, _ = metrics.reported_util_series(_mk(rows), "5h", gap_break_minutes=15)
    assert series["util_pct"].to_list().count(None) == 1


def test_pro_rows_dropped_and_cap_hits():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc).isoformat()
    rows = [
        {"sampled_at": base, "util_5h": 0.99, "util_7d": 0.0, "resets_5h_iso": reset,
         "resets_7d_iso": None, "rate_limit_tier": "default_claude_ai"},          # Pro -> dropped
        {"sampled_at": base + timedelta(minutes=5), "util_5h": 1.0, "util_7d": 0.0,
         "resets_5h_iso": reset, "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                             # Max5x, cap hit
    ]
    series, cap_hits = metrics.reported_util_series(_mk(rows), "5h")
    assert series.height == 1                # Pro row dropped
    assert cap_hits.height == 1
    assert cap_hits["util_pct"][0] == 100.0


def test_windows_over_threshold_and_peak():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset_a = datetime(2026, 5, 23, 12, 0, tzinfo=timezone.utc)
    reset_b = datetime(2026, 5, 23, 17, 0, tzinfo=timezone.utc)
    rows = [
        {"sampled_at": base, "util_5h": 0.30, "util_7d": 0.0,
         "resets_5h_iso": reset_a.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                     # window A peak 0.30 > 0.20
        {"sampled_at": base + timedelta(minutes=20), "util_5h": 0.10, "util_7d": 0.0,
         "resets_5h_iso": reset_b.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"},                     # window B peak 0.10 < 0.20
    ]
    log = _mk(rows)
    assert metrics.windows_over_threshold(log, "5h", 0.20) == (1, 2)
    assert metrics.peak_reported(log, "5h") == 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_reported_util.py -v`
Expected: FAIL — `module 'metrics' has no attribute 'reported_util_series'`.

- [ ] **Step 3: Implement in `metrics.py`**

```python
MAX5X_TIER = "default_claude_max_5x"
_RESET_JUMP = {"5h": timedelta(minutes=30), "weekly": timedelta(days=1)}
_DEFAULT_GAP_MIN = {"5h": 15, "weekly": 60}


def _parse_reset(s):
    if not s:
        return None
    try:
        d = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _window_ids(ts_list, reset_list, kind: str) -> list[int]:
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
    if log.is_empty() or "rate_limit_tier" not in log.columns:
        return log.head(0)
    return log.filter(pl.col("rate_limit_tier") == MAX5X_TIER)


def reported_util_series(log: pl.DataFrame, kind: str, gap_break_minutes: int | None = None):
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
    util_col = "util_5h" if kind == "5h" else "util_7d"
    mx = _max5x(log)
    if mx.is_empty() or util_col not in mx.columns:
        return None
    vals = mx.drop_nulls(util_col)
    return float(vals[util_col].max()) if not vals.is_empty() else None


def windows_over_threshold(log: pl.DataFrame, kind: str, threshold: float) -> tuple[int, int]:
    util_col = "util_5h" if kind == "5h" else "util_7d"
    reset_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    mx = _max5x(log).drop_nulls(["sampled_at", util_col]).sort("sampled_at")
    if mx.is_empty():
        return 0, 0
    reset_list = mx[reset_col].to_list() if reset_col in mx.columns else [None] * mx.height
    wids = _window_ids(mx["sampled_at"].to_list(), reset_list, kind)
    grouped = (
        mx.with_columns(pl.Series("_wid", wids))
        .group_by("_wid")
        .agg(pl.col(util_col).max().alias("peak"))
    )
    n_total = grouped.height
    n_over = grouped.filter(pl.col("peak") > threshold).height
    return n_over, n_total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_reported_util.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_reported_util.py
git commit -m "feat(metrics): reported_util_series + window helpers (jitter-tolerant)"
```

---

### Task 4: Live projection `project_time_to_cap`

**Files:**
- Modify: `metrics.py` (add after the Task 3 helpers)
- Test: `tests/test_projection.py` (create)

**Interfaces:**
- Consumes: `metrics._max5x`, `metrics._window_ids`, `metrics._parse_reset` (Task 3).
- Produces:
  - `metrics.CapProjection` dataclass: `eta: timedelta | None`, `before_reset: bool`.
  - `metrics.project_time_to_cap(log, now, kind="5h") -> CapProjection` — burn-rate slope over the current (jitter-tolerant) window; `eta=None` when flat/declining or <2 in-window samples; `before_reset=False` when 100% would arrive only after the parsed reset.

- [ ] **Step 1: Write the failing test**

Create `tests/test_projection.py`:
```python
from datetime import datetime, timedelta, timezone
import polars as pl
import metrics
from tests.test_reported_util import LOG_SCHEMA, _mk  # reuse fixtures


def _rows(utils, base, reset):
    return [
        {"sampled_at": base + timedelta(minutes=10 * i), "util_5h": u, "util_7d": 0.0,
         "resets_5h_iso": reset.isoformat(), "resets_7d_iso": None,
         "rate_limit_tier": "default_claude_max_5x"}
        for i, u in enumerate(utils)
    ]


def test_rising_util_returns_finite_eta_before_reset():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 18, 0, tzinfo=timezone.utc)
    log = _mk(_rows([0.2, 0.4], base, reset))   # +0.2 over 10 min -> 0.02/min
    now = base + timedelta(minutes=10)
    proj = metrics.project_time_to_cap(log, now, "5h")
    # 0.6 util remaining at 0.02/min = 30 min
    assert proj.eta is not None
    assert abs(proj.eta.total_seconds() - 30 * 60) < 90
    assert proj.before_reset is True


def test_flat_util_returns_none():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = datetime(2026, 5, 23, 18, 0, tzinfo=timezone.utc)
    log = _mk(_rows([0.5, 0.5], base, reset))
    proj = metrics.project_time_to_cap(log, base + timedelta(minutes=10), "5h")
    assert proj.eta is None


def test_eta_past_reset_flags_before_reset_false():
    base = datetime(2026, 5, 23, 8, 0, tzinfo=timezone.utc)
    reset = base + timedelta(minutes=15)  # window resets very soon
    log = _mk(_rows([0.2, 0.25], base, reset))  # slow slope -> 100% long after reset
    proj = metrics.project_time_to_cap(log, base + timedelta(minutes=10), "5h")
    assert proj.before_reset is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_projection.py -v`
Expected: FAIL — `module 'metrics' has no attribute 'project_time_to_cap'`.

- [ ] **Step 3: Implement in `metrics.py`**

```python
from dataclasses import dataclass  # add to imports at top if not already present


@dataclass
class CapProjection:
    eta: timedelta | None       # time from `now` until util reaches 1.0
    before_reset: bool          # whether 100% arrives before the window resets


def project_time_to_cap(log: pl.DataFrame, now: datetime, kind: str = "5h") -> CapProjection:
    util_col = "util_5h" if kind == "5h" else "util_7d"
    reset_col = "resets_5h_iso" if kind == "5h" else "resets_7d_iso"
    mx = _max5x(log).drop_nulls(["sampled_at", util_col]).sort("sampled_at")
    if mx.height < 2:
        return CapProjection(None, True)
    reset_list = mx[reset_col].to_list() if reset_col in mx.columns else [None] * mx.height
    wids = _window_ids(mx["sampled_at"].to_list(), reset_list, kind)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_projection.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_projection.py
git commit -m "feat(metrics): project_time_to_cap for live 'time to 100%'"
```

---

### Task 5: Rewrite `metrics.daily_stacked` (USD, by dimension)

**Files:**
- Modify: `metrics.py:117-140` (`daily_stacked`)
- Test: `tests/test_daily_stacked.py` (create)

**Interfaces:**
- Consumes: `dollar_cost` column (Task 2); optional `machine` column.
- Produces: `metrics.daily_stacked(df, by="is_subagent", value_col="dollar_cost") -> pl.DataFrame`:
  - `by="is_subagent"` → columns `date, main, subagent`.
  - `by="machine"` → columns `date, <machine values...>`; when no `machine` column, a single `local` column.

- [ ] **Step 1: Write the failing test**

Create `tests/test_daily_stacked.py`:
```python
from datetime import datetime, date, timezone
import polars as pl
import metrics

# NOTE: use real tz-aware datetime objects — `pl.datetime(...)` inside a dict is a
# Polars EXPRESSION, not a datetime, and yields an Object column that breaks `.dt.*`.

def _dt(day, hour):
    return datetime(2026, 5, day, hour, 0, tzinfo=timezone.utc)


BASE = [
    {"ts": _dt(23, 10), "is_subagent": False, "dollar_cost": 1.0, "machine": "laptop"},
    {"ts": _dt(23, 11), "is_subagent": True,  "dollar_cost": 2.0, "machine": "server"},
    {"ts": _dt(24, 10), "is_subagent": False, "dollar_cost": 4.0, "machine": "laptop"},
]


def test_by_subagent():
    out = metrics.daily_stacked(pl.DataFrame(BASE), by="is_subagent")
    row = out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)
    assert row["main"] == 1.0 and row["subagent"] == 2.0


def test_by_machine():
    out = metrics.daily_stacked(pl.DataFrame(BASE), by="machine")
    row = out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)
    assert row["laptop"] == 1.0 and row["server"] == 2.0


def test_by_machine_fallback_when_absent():
    df = pl.DataFrame([{k: v for k, v in r.items() if k != "machine"} for r in BASE])
    out = metrics.daily_stacked(df, by="machine")
    assert "local" in out.columns
    assert out.filter(pl.col("date") == date(2026, 5, 23)).row(0, named=True)["local"] == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_daily_stacked.py -v`
Expected: FAIL — current `daily_stacked` takes no `by` arg / has no `machine`/USD support (TypeError or wrong columns).

- [ ] **Step 3: Replace `daily_stacked` in `metrics.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_daily_stacked.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/test_daily_stacked.py
git commit -m "feat(metrics): daily_stacked in USD, pivot by machine or is_subagent"
```

---

### Task 6: Trim `caps.py` to a live snapshot

**Files:**
- Modify: `caps.py` (delete derivation code; trim dataclass)
- Modify: `config.py` (delete now-dead cap constants — see Step 4b)
- Test: `tests/test_caps_snapshot.py` (create)

**Interfaces:**
- Produces:
  - `caps.DerivedCaps` reduced to snapshot fields: `sampled_at, sample_util_5h, sample_util_7d, subscription_type, resets_5h_iso, resets_7d_iso, rate_limit_tier` (all `Optional`, defaulting to `None`).
  - `caps.load_caps() -> DerivedCaps`, `caps.save_caps(caps) -> None` (unchanged behavior; `load_caps` already filters JSON to known fields so old files load).
- Removes: `derive_from_reading`, `implied_cap_series`, `derive_continuous_caps`, `cap_series`, `attach_time_varying_cap`, `effective_caps`, `hour_of_day_cap_series`, `hour_of_day_sample_counts`, `_per_hour_medians`, `_smooth_rolling_circular`, `_interpolate_empty_circular`, `global_cap_from_anchors`, `calibrate_hourly_to_log`, `attach_hour_of_day_cap`, and the now-unused module constants (`MIN_UTILIZATION_FOR_CALIBRATION`, `CONTINUOUS_*`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_caps_snapshot.py`:
```python
import caps


def test_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(caps, "CAPS_PATH", tmp_path / "caps.json")
    snap = caps.DerivedCaps(
        sampled_at="2026-05-23T10:00:00+00:00",
        sample_util_5h=0.4, sample_util_7d=0.5,
        subscription_type="max", resets_5h_iso="2026-05-23T12:00:00+00:00",
        resets_7d_iso="2026-05-25T07:00:00+00:00", rate_limit_tier="default_claude_max_5x",
    )
    caps.save_caps(snap)
    loaded = caps.load_caps()
    assert loaded.sample_util_5h == 0.4
    assert loaded.rate_limit_tier == "default_claude_max_5x"


def test_legacy_json_extra_keys_are_ignored(tmp_path, monkeypatch):
    p = tmp_path / "caps.json"
    p.write_text('{"sample_util_5h": 0.3, "max5x_5h": 9999999, "pro_5h": 123}', encoding="utf-8")
    monkeypatch.setattr(caps, "CAPS_PATH", p)
    loaded = caps.load_caps()           # must not crash on the removed max5x_5h/pro_5h keys
    assert loaded.sample_util_5h == 0.3


def test_derivation_functions_removed():
    for gone in ("global_cap_from_anchors", "derive_continuous_caps", "derive_from_reading",
                 "effective_caps", "hour_of_day_cap_series", "implied_cap_series"):
        assert not hasattr(caps, gone), f"{gone} should be deleted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_caps_snapshot.py -v`
Expected: FAIL on `test_derivation_functions_removed` (functions still present), and `DerivedCaps(...)` call fails because the current dataclass requires positional `max5x_5h` etc.

- [ ] **Step 3: Replace the top of `caps.py`** — keep only the snapshot. Replace `caps.py:1-167` (everything from the imports through `derive_continuous_caps`) and then delete the remaining functions (`cap_series` through `attach_hour_of_day_cap`, `caps.py:169-611`). The whole file becomes:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_caps_snapshot.py -v`
Expected: PASS (3 tests). (Other suites referencing deleted caps functions will fail to import — that's expected and fixed in Task 13; do not run the whole suite yet.)

- [ ] **Step 4b: Delete the now-dead cap constants in `config.py`**

`effective_caps` (just deleted) was the only reader of `config.PRO_CAP_5H_COST_WEIGHTED`,
`PRO_CAP_WEEKLY_COST_WEIGHTED`, `MAX5X_CAP_5H_COST_WEIGHTED`, `MAX5X_CAP_WEEKLY_COST_WEIGHTED`
(`config.py:35-39`). First confirm with the Grep tool that no other `.py` references them, then
delete those four lines. (Leave `COST_WEIGHTS`, `MODEL_CONTEXT_WINDOWS`, `MODEL_PRICING`, the
reset/TZ/night constants, and `FIVE_HOUR_WINDOW_HOURS` — still used by `app.py`'s burn-window
computation.)

- [ ] **Step 5: Commit**

```bash
git add caps.py config.py tests/test_caps_snapshot.py
git commit -m "refactor(caps): reduce to live snapshot; delete cap-derivation machinery"
```

---

### Task 7: New reported-usage chart renderer + delete old chart renderers

**Files:**
- Modify: `render.py` (add `build_reported_figure` + `render_reported_usage_chart`; delete `render_5h_chart`, `render_weekly_chart`, `_add_cumulative_traces`, `_peak_for_decomposition`)
- Test: `tests/test_render_reported.py` (create)

**Interfaces:**
- Consumes: `metrics.reported_util_series` (Task 3), `render._rangeselector_xaxis`, `render.add_calendar_bands`.
- Produces:
  - `render.build_reported_figure(series, cap_hits, kind, data_start_ts, data_end_ts) -> go.Figure` (pure, testable).
  - `render.render_reported_usage_chart(log, kind, data_start_ts, data_end_ts) -> None` (st wrapper).

- [ ] **Step 1: Write the failing test** (tests the pure figure builder only)

Create `tests/test_render_reported.py`:
```python
from datetime import datetime, timezone
import polars as pl
import render


def test_build_reported_figure_has_line_caphits_and_two_reflines():
    series = pl.DataFrame(
        {"ts": [datetime(2026, 5, 23, 8, tzinfo=timezone.utc),
                datetime(2026, 5, 23, 9, tzinfo=timezone.utc)],
         "util_pct": [10.0, 100.0]},
        schema={"ts": pl.Datetime("ms", "UTC"), "util_pct": pl.Float64},
    )
    cap_hits = pl.DataFrame(
        {"ts": [datetime(2026, 5, 23, 9, tzinfo=timezone.utc)], "util_pct": [100.0]},
        schema={"ts": pl.Datetime("ms", "UTC"), "util_pct": pl.Float64},
    )
    fig = render.build_reported_figure(
        series, cap_hits, "5h",
        datetime(2026, 5, 23, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 24, 0, tzinfo=timezone.utc),
    )
    names = [t.name for t in fig.data]
    assert any("reported" in (n or "").lower() for n in names)   # the util line
    assert any("cap" in (n or "").lower() or "hit" in (n or "").lower() for n in names)
    # two horizontal reference lines (20% Pro, 100% Max 5x) added as shapes/annotations
    assert len(fig.layout.shapes) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_render_reported.py -v`
Expected: FAIL — `module 'render' has no attribute 'build_reported_figure'`.

- [ ] **Step 3: Implement in `render.py`** (add these; place near the old chart functions you will delete)

```python
def build_reported_figure(series: pl.DataFrame, cap_hits: pl.DataFrame, kind: str,
                          data_start_ts, data_end_ts) -> go.Figure:
    label = "5h" if kind == "5h" else "weekly"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["ts"].to_list(), y=series["util_pct"].to_list(),
        mode="lines", name=f"reported {label} util",
        line=dict(width=1.6, color="#4f8cff"), connectgaps=False,
    ))
    if not cap_hits.is_empty():
        fig.add_trace(go.Scatter(
            x=cap_hits["ts"].to_list(), y=cap_hits["util_pct"].to_list(),
            mode="markers", name="cap hit (≥99%)",
            marker=dict(size=8, color="red", symbol="circle"),
            hovertemplate="%{x}<br>%{y:.0f}%<extra>cap hit</extra>",
        ))
    fig.add_hline(y=20.0, line_dash="dash", line_color="red",
                  annotation_text="Pro cap (est. 20%)", annotation_position="top left")
    fig.add_hline(y=100.0, line_dash="dot", line_color="orange",
                  annotation_text="Max 5x cap (100%)", annotation_position="top left")
    if data_start_ts is not None and data_end_ts is not None:
        add_calendar_bands(fig, data_start_ts, data_end_ts)
    fig.update_layout(
        height=350, margin=dict(t=60, b=20, l=10, r=10),
        yaxis_title="% of Max 5x cap", yaxis_ticksuffix="%",
        yaxis=dict(autorange=True),
        xaxis=_rangeselector_xaxis(),
        legend=dict(orientation="h"),
    )
    return fig


def render_reported_usage_chart(log: pl.DataFrame, kind: str,
                                data_start_ts, data_end_ts) -> None:
    title = ("5h usage limit (reported)" if kind == "5h"
             else "Weekly usage limit (reported)")
    st.subheader(title)
    st.caption("Anthropic's reported utilization (Max 5x plan only). Account-wide — the "
               "project/model filter does not affect this chart. Line breaks at each window "
               "reset and across sampling gaps; red dots mark where you hit the cap.")
    series, cap_hits = metrics.reported_util_series(log, kind)
    if series.is_empty():
        st.info("No Max-5x reported-usage samples yet.")
        return
    fig = build_reported_figure(series, cap_hits, kind, data_start_ts, data_end_ts)
    st.plotly_chart(fig, width="stretch")
```

Then DELETE `render_5h_chart` (`render.py:133-179`), `render_weekly_chart` (`render.py:182-221`), `_add_cumulative_traces` (`render.py:37-64`), and `_peak_for_decomposition` (`render.py:67-77`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_render_reported.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_render_reported.py
git commit -m "feat(render): reported-usage chart; remove cumulative chart renderers"
```

---

### Task 8: Daily-bar (USD, by-dimension) + new `render_kpis`

**Files:**
- Modify: `render.py` (`render_daily_bar`, `render_kpis`)
- Test: `tests/test_render_daily.py` (create)

**Interfaces:**
- Consumes: `metrics.daily_stacked` (Task 5).
- Produces:
  - `render.build_daily_figure(daily) -> go.Figure` (pure; one stacked Bar per non-`date` column).
  - `render.render_daily_bar(fdf, decomposition_key) -> None` (radio: "by machine" / "main vs sub"; computes `daily_stacked` inside).
  - `render.render_kpis(total_usd, daily_avg_usd, peak_5h, peak_weekly, windows_over_pro_5h, windows_total_5h, weeks_over_pro, weeks_total) -> None` (new signature; util KPIs are fractions 0–1 or None).

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_daily.py` (use a real `date`, not the `pl.date(...)` expression):
```python
from datetime import date
import polars as pl
import render


def test_build_daily_figure_one_bar_per_series():
    daily = pl.DataFrame({"date": [date(2026, 5, 23)], "laptop": [1.0], "server": [2.0]})
    fig = render.build_daily_figure(daily)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"
    assert fig.layout.yaxis.title.text == "USD"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_render_daily.py -v`
Expected: FAIL — `module 'render' has no attribute 'build_daily_figure'`.

- [ ] **Step 3: Implement in `render.py`** — replace `render_daily_bar` (`render.py:224-238`) and `render_kpis` (`render.py:116-130`):

```python
_DAILY_PALETTE = ["#4f8cff", "#ff8a4f", "#46b46e", "#b46edc", "#dcb446", "#46b4b4"]


def build_daily_figure(daily: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    series_cols = [c for c in daily.columns if c != "date"]
    x = daily["date"].to_list()
    for i, col in enumerate(series_cols):
        fig.add_trace(go.Bar(x=x, y=daily[col].to_list(), name=col,
                             marker_color=_DAILY_PALETTE[i % len(_DAILY_PALETTE)]))
    fig.update_layout(
        barmode="stack", height=300, margin=dict(t=60, b=20, l=10, r=10),
        yaxis_title="USD", yaxis=dict(autorange=True),
        xaxis=_rangeselector_xaxis(), legend=dict(orientation="h"),
    )
    return fig


def render_daily_bar(fdf: pl.DataFrame, decomposition_key: str) -> None:
    st.subheader("Daily burn (USD)")
    st.caption("Estimated API-equivalent cost (what this usage would cost at pay-as-you-go "
               "API rates — comparable to ccusage). Cache writes priced at 1.25× input (5m).")
    mode = st.radio("Decomposition", ["by machine", "main vs sub"],
                    index=0, horizontal=True, key=f"daily_decomp_{decomposition_key}")
    by = "machine" if mode == "by machine" else "is_subagent"
    daily = metrics.daily_stacked(fdf, by=by)
    if daily.is_empty():
        st.info("No data for the daily chart.")
        return
    st.plotly_chart(build_daily_figure(daily), width="stretch")


def render_kpis(total_usd: float, daily_avg_usd: float,
                peak_5h: float | None, peak_weekly: float | None,
                windows_over_pro_5h: int, windows_total_5h: int,
                weeks_over_pro: int, weeks_total: int):
    def pct(v):
        return f"{v*100:.0f}%" if v is not None else "—"
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total $", f"${total_usd:,.0f}")
    k2.metric("Daily avg", f"${daily_avg_usd:,.1f}/d")
    k3.metric("Peak 5h", pct(peak_5h), help="Max reported 5h utilization (Max 5x)")
    k4.metric("5h-windows over Pro", f"{windows_over_pro_5h} / {windows_total_5h}")
    k5.metric("Peak weekly", pct(peak_weekly), help="Max reported weekly utilization (Max 5x)")
    k6.metric("Weeks over Pro", f"{weeks_over_pro} / {weeks_total}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_render_daily.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_render_daily.py
git commit -m "feat(render): USD daily bar by machine + re-sourced KPI strip"
```

---

### Task 9: Live-panel projection + delete calibration-history / cost-vs-session renderers

**Files:**
- Modify: `render.py` (`render_live_panel_from_cache` gains a `log` param + projection; delete `render_calibration_history`, `render_cost_vs_session_length`, `_cost_vs_session_length_interactive`)
- Test: `tests/test_projection_format.py` (create — tests a pure formatting helper)

**Interfaces:**
- Consumes: `metrics.project_time_to_cap` (Task 4).
- Produces:
  - `render.format_projection(proj) -> str` (pure): `"—"` if `eta is None`; `"won't hit 100% before reset"` if `not before_reset`; else `"~Xh Ym"`.
  - `render.render_live_panel_from_cache(*, agent_seconds_old, log) -> None` (new `log` keyword param; renders the projection under the 5h metric).

- [ ] **Step 1: Write the failing test**

Create `tests/test_projection_format.py`:
```python
from datetime import timedelta
import render
from metrics import CapProjection


def test_format_none():
    assert render.format_projection(CapProjection(None, True)) == "—"


def test_format_after_reset():
    assert "reset" in render.format_projection(CapProjection(timedelta(hours=9), False))


def test_format_eta():
    s = render.format_projection(CapProjection(timedelta(hours=1, minutes=20), True))
    assert "1h" in s and "20m" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYRUN -m pytest tests/test_projection_format.py -v`
Expected: FAIL — `module 'render' has no attribute 'format_projection'`.

- [ ] **Step 3: Implement in `render.py`**

```python
def format_projection(proj) -> str:
    if proj is None or proj.eta is None:
        return "—"
    if not proj.before_reset:
        return "won't hit 100% before reset"
    mins = int(proj.eta.total_seconds() // 60)
    return f"~{mins // 60}h {mins % 60}m"
```

Change `render_live_panel_from_cache` signature (`render.py:393`) to:
```python
def render_live_panel_from_cache(*, agent_seconds_old: float | None, log: pl.DataFrame):
```
Inside, after the 5h metric/progress block (around `render.py:423`), add the projection. Replace the `if prev.max5x_5h or prev.max5x_weekly:` caption block (`render.py:431-437`) entirely with:
```python
    if prev.sample_util_5h is not None:
        proj = metrics.project_time_to_cap(log, now, "5h")
        cols[1].caption("5h → 100%: " + format_projection(proj))
```
Remove all references to `prev.max5x_*`, `prev.pro_*`, `prev.sample_burn_*` (those fields no longer exist).

Then DELETE `render_calibration_history` (`render.py:260-390`), `render_cost_vs_session_length` (`render.py:489-513`), and `_cost_vs_session_length_interactive` (`render.py:440-486`).

Finally, remove the now-unused module imports at the top of `render.py`: `import app_cache` (only used by the deleted cost-vs-session renderer) and `import calibration_log` (only used by the deleted calibration-history renderer). **Keep `import caps as caps_mod`** — `render_live_panel_from_cache` still calls `caps_mod.load_caps()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYRUN -m pytest tests/test_projection_format.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_projection_format.py
git commit -m "feat(render): live 'time to 100%' projection; drop calibration-history + cost-vs-session"
```

---

### Task 10: Slim `app_cache.py`

**Files:**
- Modify: `app_cache.py` (delete `Calibration`, `_calibrate`, `calibrate`; slim `FilteredCompute`/`_filtered`/`filtered_compute` to sessions only)

**Interfaces:**
- Produces: `app_cache.filtered_compute(fdf) -> FilteredCompute` where `FilteredCompute` has a single field `sessions: pl.DataFrame` (= `metrics.session_summaries(fdf)`), cached on `_sig1(fdf)`.
- Removes: everything that referenced `caps_mod.global_cap_from_anchors`, `metrics.effective_window_hours`, `metrics.five_hour_burn_since_reset`, `metrics.weekly_burn_since_reset`, `metrics.five_hour_window_totals`, `metrics.week_start_for`, `metrics.session_cost_attribution`.

- [ ] **Step 1: Replace `app_cache.py` body** (keep `_sig1`):

```python
"""Cached, filter-dependent compute shared by both apps."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import streamlit as st

import metrics


def _sig1(df: pl.DataFrame) -> tuple:
    if df.is_empty() or "ts" not in df.columns:
        return (0, None)
    return (df.height, str(df["ts"].max()))


@dataclass
class FilteredCompute:
    sessions: pl.DataFrame


@st.cache_data(show_spinner=False)
def _filtered(_key: tuple, _fdf: pl.DataFrame) -> FilteredCompute:
    return FilteredCompute(sessions=metrics.session_summaries(_fdf))


def filtered_compute(fdf: pl.DataFrame) -> FilteredCompute:
    return _filtered(_sig1(fdf), fdf)
```

Delete the `caps_mod`/`config` imports if now unused.

- [ ] **Step 2: Verify it imports**

Run: `PYRUN -c "import app_cache; print('ok')"`
Expected: prints `ok` (no import error).

- [ ] **Step 3: Commit**

```bash
git add app_cache.py
git commit -m "refactor(app_cache): drop calibration; filtered_compute returns sessions only"
```

---

### Task 11: Rewrite `app.py` (local app)

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: Tasks 1-10 (`metrics.*`, `render.*`, `app_cache.filtered_compute`, `caps.DerivedCaps`).

- [ ] **Step 1: Rewrite the live fragment write-path** — in `live_usage_panel` (`app.py:95-228`), KEEP `cache.refresh_cache`/rerun (`:98-103`), `fetch_usage` + RateLimited/except handling (`:104-130`), and `calibration_log.append_sample(...)` (`:150-171`). REPLACE the cap-derivation block (`app.py:173-204`, from `snap_meta = {...}` through `caps_mod.save_caps(derived)`) with a direct snapshot save:

```python
    snapshot = caps_mod.DerivedCaps(
        sampled_at=snap.sampled_at.isoformat(),
        sample_util_5h=snap.five_hour.utilization if snap.five_hour else None,
        sample_util_7d=snap.seven_day.utilization if snap.seven_day else None,
        subscription_type=snap.subscription_type,
        resets_5h_iso=snap.five_hour.resets_at.isoformat() if snap.five_hour and snap.five_hour.resets_at else None,
        resets_7d_iso=snap.seven_day.resets_at.isoformat() if snap.seven_day and snap.seven_day.resets_at else None,
        rate_limit_tier=snap.rate_limit_tier,
    )
    caps_mod.save_caps(snapshot)
```

KEEP the Supabase upload block (`app.py:206-216`) unchanged. The `_eff_5h`/`window_start_*`/`burn_*`/`agg_*` computations (`app.py:132-152`) feed `append_sample`'s burn/agg columns; they reference deleted `metrics.effective_window_hours`. Replace them with simple null/zero fills so the schema is preserved:

```python
    df_all = load_data()
    window_start_5h = (snap.five_hour.resets_at - timedelta(hours=config.FIVE_HOUR_WINDOW_HOURS)
                       if snap.five_hour and snap.five_hour.resets_at else None)
    window_start_weekly = (snap.seven_day.resets_at - timedelta(days=7)
                           if snap.seven_day and snap.seven_day.resets_at else None)
    burn_5h = calibration_log.cost_weighted_sum_in_window(df_all, window_start_5h, snap.sampled_at)
    burn_weekly = calibration_log.cost_weighted_sum_in_window(df_all, window_start_weekly, snap.sampled_at)
    agg_5h = calibration_log.window_aggregates(df_all, window_start_5h, snap.sampled_at)
    agg_weekly = calibration_log.window_aggregates(df_all, window_start_weekly, snap.sampled_at)
```

(This drops the `effective_window_hours` dependency while keeping the log schema populated. `calibration_log.cost_weighted_sum_in_window` and `window_aggregates` still exist and are unchanged.)

- [ ] **Step 2: Update `_render_usage_view`** (`app.py:47-92`) — add the projection and drop the derived-cap caption. Change its signature to also accept `log` and remove the `derived` param. Replace the `bits = [...]` cap-caption block (`app.py:74-92`) with a projection line:

```python
    if util_5h is not None:
        proj = metrics.project_time_to_cap(log, now, "5h")
        cols[1].caption("5h → 100%: " + render.format_projection(proj))
    age_s = (now - sampled_at).total_seconds()
    age_str = (f"{int(age_s)}s ago" if age_s < 90
               else f"{int(age_s/60)}m ago" if age_s < 5400 else f"{age_s/3600:.1f}h ago")
    prefix = f"Updated {age_str}" + (" · stale (endpoint rate-limited)" if stale else "")
    st.caption(prefix + f" · sub `{sub_type}` tier `{rate_limit_tier}`")
```

Update the two call sites of `_render_usage_view` (the RateLimited branch `app.py:116-126` and the live branch `app.py:218-228`) to pass `log=calibration_log.load_log()` and drop `derived=`.

- [ ] **Step 3: Rewrite the main body** — replace the cap/share block (`app.py:302-322`) and the KPI + chart blocks (`app.py:325-375`). New main body after `fdf`/`data_start_ts`/`data_end_ts` are set (`app.py:297-299`):

```python
    log = calibration_log.load_log()

    # ---- KPIs ----
    total_usd = float(df["dollar_cost"].sum())
    span_days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0, 1.0)
    daily_avg_usd = total_usd / span_days
    peak_5h = metrics.peak_reported(log, "5h")
    peak_weekly = metrics.peak_reported(log, "weekly")
    w5_over, w5_total = metrics.windows_over_threshold(log, "5h", 0.20)
    wk_over, wk_total = metrics.windows_over_threshold(log, "weekly", 0.20)
    render.render_kpis(total_usd, daily_avg_usd, peak_5h, peak_weekly,
                       w5_over, w5_total, wk_over, wk_total)

    # ---- Charts ----
    render.render_reported_usage_chart(log, "5h", data_start_ts, data_end_ts)
    render.render_reported_usage_chart(log, "weekly", data_start_ts, data_end_ts)
    render.render_daily_bar(fdf, decomposition_key="app")

    # ---- Session table ----
    fc = app_cache.filtered_compute(fdf)
    sessions = fc.sessions
```

Keep the session-table block (`app.py:378-397`) but source `sessions` from `fc.sessions` above; delete `render.render_cost_vs_session_length(...)` (`app.py:375`) and `render.render_calibration_history(...)` (`app.py:234`). Remove the `df_with_caps`/`share_*` columns entirely.

- [ ] **Step 4: Sidebar cleanup** — delete the "Plan caps" subheader/caption + `show_max5x` checkbox (`app.py:267-270`). Keep project/model/sort/session-table-filter controls.

- [ ] **Step 5: Verify the app imports and the suite still green for the new modules**

Run: `PYRUN -c "import app" 2>&1 | head -20`
Expected: no `ImportError`/`AttributeError` (Streamlit may warn about bare-mode `st.*` calls — that's fine; there must be no import-time crash).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(app): reported charts + USD daily + snapshot save; remove calibration"
```

---

### Task 12: Rewrite `app_cloud.py` (cloud viewer)

**Files:**
- Modify: `app_cloud.py`

**Interfaces:**
- Consumes: Tasks 1-10; mirrors Task 11 for the read-only cloud app.

- [ ] **Step 1: Pass the log into the live panel** — in `refresh_data_panel` (`app_cloud.py:69-108`), change the call (`app_cloud.py:106`) to:
```python
        render.render_live_panel_from_cache(
            agent_seconds_old=seconds_old, log=calibration_log.load_log())
```

- [ ] **Step 2: Rewrite the main body** — replace the calibration/cap/share block (`app_cloud.py:175-208`) and the KPI + chart calls (`app_cloud.py:207-233`) with the same KPI + chart structure as Task 11 Step 3:

```python
log = calibration_log.load_log()

total_usd = float(df["dollar_cost"].sum())
span_days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0, 1.0)
daily_avg_usd = total_usd / span_days
peak_5h = metrics.peak_reported(log, "5h")
peak_weekly = metrics.peak_reported(log, "weekly")
w5_over, w5_total = metrics.windows_over_threshold(log, "5h", 0.20)
wk_over, wk_total = metrics.windows_over_threshold(log, "weekly", 0.20)

render.render_kpis(total_usd, daily_avg_usd, peak_5h, peak_weekly,
                   w5_over, w5_total, wk_over, wk_total)
render.render_reported_usage_chart(log, "5h", data_start_ts, data_end_ts)
render.render_reported_usage_chart(log, "weekly", data_start_ts, data_end_ts)
render.render_daily_bar(fdf, decomposition_key="cloud")

fc = app_cache.filtered_compute(fdf)
sessions = fc.sessions
```

Keep the session-table block (`app_cloud.py:235-249`) sourcing `sessions` from `fc.sessions`. Delete `render.render_cost_vs_session_length(...)` (`app_cloud.py:233`) and `render.render_calibration_history(df)` (`app_cloud.py:251`). Remove `df_with_caps`/`share_*`.

- [ ] **Step 3: Sidebar cleanup** — delete the "Plan caps (read-only)" subheader/caption + `show_max5x` checkbox (`app_cloud.py:147-149`).

- [ ] **Step 4: Verify import**

Run: `PYRUN -c "import app_cloud" 2>&1 | head -20`
Expected: no import-time crash (Streamlit secrets warning is fine).

- [ ] **Step 5: Commit**

```bash
git add app_cloud.py
git commit -m "feat(app_cloud): reported charts + USD daily; remove calibration"
```

---

### Task 13: Delete dead `metrics.py` code + dead tests; green the whole suite

**Files:**
- Modify: `metrics.py` (delete now-unused functions)
- Modify/Delete: `tests/` (remove tests for deleted code)

**Interfaces:**
- Removes from `metrics.py`: `observed_window_lengths`, `effective_window_hours`, `week_start_for`, `_build_week_boundaries`, `five_hour_burn_since_reset`, `five_hour_window_totals`, `weekly_burn_since_reset`, `downsample_cumulative`, `bin_sessions`, `session_cost_attribution`. (Keep `rolling_burn`, `fraction_time_over_cap`, `cap_crossings` only if still referenced — grep first; they are not used by the apps and may be deleted too.)

- [ ] **Step 1: Grep for any remaining callers before deleting**

Run:
```bash
PYRUN -c "print('manual grep step')"
```
Then use the Grep tool for each function name across `*.py` (excluding `tests/`): `effective_window_hours`, `week_start_for`, `five_hour_burn_since_reset`, `five_hour_window_totals`, `weekly_burn_since_reset`, `downsample_cumulative`, `bin_sessions`, `session_cost_attribution`, `observed_window_lengths`. Expected: only definitions in `metrics.py` and references in `tests/` remain (apps were rewritten in Tasks 10-12). If any app reference remains, fix it before deleting.

- [ ] **Step 2: Delete the dead functions** from `metrics.py` (the ones listed above, plus `_inject_gap_probes`/`_parse_window` if `rolling_burn` is also removed).

- [ ] **Step 3: Delete dead test files / cases**

- Delete file `tests/test_calibration_characterization.py`.
- In `tests/test_metrics.py`, delete every test referencing the removed functions (`test_weekly_*`, `test_five_hour_*`, `test_downsample_cumulative_*`, `week_start_for`, `bin_sessions`). If nothing meaningful remains, delete the file.
- Delete `tests/test_session_cost_attribution.py` (its functions are gone).

Use Grep to confirm no test still imports a deleted symbol:
```bash
# via Grep tool: search tests/ for: effective_window_hours|week_start_for|five_hour_burn_since_reset|weekly_burn_since_reset|downsample_cumulative|bin_sessions|session_cost_attribution|global_cap_from_anchors|derive_continuous_caps|implied_cap_series
```
Expected: no matches after deletion.

- [ ] **Step 4: Run the FULL suite green**

Run: `PYRUN -m pytest tests/ -v`
Expected: PASS — all remaining tests (the new ones from Tasks 1-9 plus `test_cache_merge.py`, `test_supabase_sync.py`). No collection errors.

- [ ] **Step 5: Commit**

```bash
git add metrics.py tests/
git commit -m "chore: delete dead calibration code and tests; suite green"
```

---

### Task 14: Manual smoke test of the local app

**Files:** none (verification)

- [ ] **Step 1: Launch the app** (data files `cache.parquet`, `caps.json`, `calibration_log.parquet` are present)

Run:
```bash
cd "C:/Users/borgi/Documents/claude-usage-tracker" && PYTHONPATH=".venv/Lib/site-packages" "/c/Users/borgi/AppData/Local/Programs/Python/Python312/python.exe" -m streamlit run app.py --server.headless true --server.port 8765
```

- [ ] **Step 2: Verify in the browser at http://localhost:8765**
  - "5h usage limit (reported)" shows a continuous sawtooth (teeth, not isolated dots) with ~4 red cap-hit dots, and 20% + 100% reference lines.
  - "Weekly usage limit (reported)" shows the reported weekly % (peak ~92%, no red dots).
  - "Daily burn (USD)" is in dollars, stacked by machine by default; the toggle switches to main-vs-sub.
  - KPI strip shows Total $, Daily avg $, Peak 5h/weekly %, windows/weeks over Pro.
  - Live panel shows "5h → 100%: …" projection and no "calibrated cap ≈ XM" caption.
  - The calibration-history expander and the cost-vs-session chart are gone.

- [ ] **Step 3:** If anything is wrong, fix in the relevant task's files and re-run. No commit for this task unless a fix was needed.

---

## Self-Review (completed during planning)

- **Spec coverage:** reported 5h chart (Task 7), reported weekly chart (Task 7), Max-5x filter (Task 3 `_max5x`), reset/gap line breaks with jitter tolerance (Task 3), red cap-hit dots (Tasks 3+7), 20%/100% reference lines (Task 7), USD daily by machine + toggle (Tasks 2/5/8), `MODEL_PRICING`/`price_for` longest-prefix (Task 1), KPI re-source (Tasks 3+8+11+12), live projection with jitter-tolerant window + before-reset cap (Tasks 4+9+11), caps.py trim (Task 6), full deletion lists incl. orphans `effective_caps`/`cap_series`/`week_start_for`/`five_hour_window_totals`/`session_cost_attribution` (Tasks 6/10/13), test deletions enumerated (Task 13), broken-venv workaround (Task 0 + Global Constraints). All covered.
- **Placeholder scan:** no TBD/TODO; every code step shows real code.
- **Type consistency:** `reported_util_series` returns `(series, cap_hits)` consumed by `build_reported_figure`; `CapProjection(eta, before_reset)` produced in Task 4 and formatted in Task 9; `windows_over_threshold` returns `(int, int)` consumed by `render_kpis`; `daily_stacked(df, by=...)` consumed by `render_daily_bar`. Consistent across tasks.
