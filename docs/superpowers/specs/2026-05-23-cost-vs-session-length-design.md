# Cost vs Session Length — Design

**Date:** 2026-05-23
**Status:** Approved (design), pending implementation plan

## Motivation

Working hypothesis: the dominant driver of how much of the plan cap a stretch of
work consumes is **how long a single session runs**, not time of day. As a session
grows, every turn re-feeds a larger context, so later turns are more expensive. We
want a chart that tests whether longer/bigger sessions burn a disproportionate share
of the cap.

The ground truth for "cost" is the **API utilization percentage** Anthropic returns
(`util_5h`, `util_7d`), not a token count or our derived cap. We have been logging
these every poll in `calibration_log.parquet`, so we can correlate them against
per-session token data from the cache.

## The attribution problem

The API percentage is **window-cumulative**: a single `util_5h` reading reflects
everything burned in the current 5h window, which may contain several sessions. We
need a principled way to turn that into a per-session percentage **for all sessions**
(excluding overlapping sessions loses too much data).

### Chosen method: Δ-sum split by output share

The local agent samples the API every 5 minutes. Each calibration-log row carries
`sampled_at`, `util_5h`, `util_7d` (stored as 0.0–1.0 fractions), `resets_5h_iso`,
and `resets_7d_iso`.

For a given window kind (`5h` or `weekly`):

1. Sort the log by `sampled_at`. Walk consecutive sample pairs `(tᵢ₋₁, tᵢ]`.
2. Only diff a pair when both samples share the **same** reset id
   (`resets_5h_iso` for 5h, `resets_7d_iso` for weekly). Pairs straddling a reset
   are skipped — at most one dropped interval per window, negligible.
3. `Δutil = max(0, util(tᵢ) − util(tᵢ₋₁))`.
4. From the cache, sum `output_tokens` per `session_id` over all turns
   (main **and** subagent — subagent rows carry the parent `session_id`) whose `ts`
   falls in the half-open interval `(tᵢ₋₁, tᵢ]`. Call the per-session sums
   `o_s` and the interval total `O = Σ o_s`.
5. If `O > 0`, attribute `Δutil · (o_s / O)` to each session `s`. If `O == 0` and
   `Δutil > 0`, the delta is **unattributable** (poll lag / no logged turns);
   accumulate it into a diagnostic `unattributed_pct` total and skip.
6. A session's attributed percentage = the sum of its slices across **every interval
   and window** it touched.

**Why this is sound.** `util` tracks output tokens (established in `CLAUDE.md`:
Anthropic meters by output, not by our cost-weighted mix). Splitting `Δutil` by
output share is therefore the physically correct division. When a single session owns
a window, its attributed % collapses exactly to `session_output / cap`. The method
uses the **real measured** percentages rather than our estimated cap, and it includes
every session regardless of overlap.

**Notes / edge cases.**
- A long session that spans multiple 5h windows accumulates a slice from each, so its
  5h attributed % can exceed 100% (meaningful: "consumed >1× of a 5h budget"). Weekly
  sessions almost never cross the Sunday 07:00 reset.
- Sessions with zero output in an interval receive nothing from that interval —
  correct.
- The unattributable total is surfaced as a small caption ("X% of measured burn
  could not be matched to a session") for transparency, not plotted.

## Per-session X-axis metrics (toggle)

Computed once per `session_id` over all its turns:

| Toggle option | Definition |
|---|---|
| **Prompt tokens** (default) | `sum(input + cache_creation + cache_read)` — the context re-fed each turn |
| **Requests** | `main_turns` — count of non-subagent turns ≈ number of user requests |
| **Raw total tokens** | `sum(raw_total_tokens)` |

With Y ≈ output/cap, plotting against **prompt tokens** reveals output-per-context
efficiency vs session size — an upward bend confirms the hypothesis. Against
**requests** it shows per-turn-count cost. **Raw total** is partly mechanical (it
includes output) and offered for completeness.

## Binning

- **Quantile bins**: equal session-count per bin so error bars are comparable.
- Bin count is a **UI slider** (range 4–20, default 8).
- Each bin plots **mean Y** at the bin's **median X**, with **±1 std** error bars and
  a hover/marker annotation of bin count `n`.
- Bins with `n < 2` render the marker without an error bar.

## Chart layout

A new section titled **"Cost vs session length"** in both apps:

- Controls row: X-axis `selectbox` (Prompt tokens / Requests / Raw total) + bin-count
  `slider`.
- Two charts via `st.columns(2)`: **5h** (left), **Weekly** (right). Identical shape,
  different Y source. Plotly markers + line + error bars. Y-axis labelled "% of cap
  consumed (attributed)".
- Caption below: the unattributable-burn diagnostic and a one-line read-the-chart hint.

## Components & boundaries

**`metrics.py` (pure, tested):**

- `session_cost_attribution(df: pl.DataFrame, log: pl.DataFrame) -> tuple[pl.DataFrame, dict]`
  Returns `(sessions, diagnostics)`:
  - `sessions`: one row per `session_id` with `session_id`, `attributed_pct_5h`,
    `attributed_pct_weekly`, `prompt_tokens`, `n_requests`, `raw_total_tokens`.
  - `diagnostics`: `{"unattributed_5h": float, "unattributed_7d": float}` — the summed
    `Δutil` (as a fraction) that could not be matched to any session, per window.
  - `df` is the derived cache (already has `ts`, `output_tokens`, `prompt_tokens`,
    `raw_total_tokens`, `is_subagent`, `session_id`).
  - `log` is `calibration_log.load_log()`.
- `bin_sessions(sessions: pl.DataFrame, x_col: str, y_col: str, n_bins: int) -> pl.DataFrame`
  Quantile-bins `sessions` on `x_col`, returns `bin_median_x`, `mean_y`, `std_y`, `n`.

**`render.py` (shared):**

- `render_cost_vs_session_length(df: pl.DataFrame, log: pl.DataFrame)` — owns the
  controls, calls the two metrics functions, draws the two Plotly figures. Follows the
  existing render-helper convention: takes the **derived** df (with `ts`), not raw
  cache (see `feedback-refactor-preserve-data-prep`).

**`app.py` / `app_cloud.py`:** each adds one call to the new render function in a new
section. Both already load the derived df and the calibration log.

## Data flow

```
calibration_log.parquet ──┐
                          ├─> metrics.session_cost_attribution ─> per-session df
cache (derived df) ───────┘                                          │
                                                                     ├─> metrics.bin_sessions ─> binned df ─> Plotly
                                          UI: x_col toggle, n_bins ───┘
```

## Testing (TDD)

- `session_cost_attribution`:
  - Single session owns a window → attributed % equals `Δutil` over that window.
  - Two overlapping sessions in one interval → `Δutil` split by output share.
  - Reset-straddling sample pair → that interval skipped.
  - Interval with `Δutil > 0` but no turns → counted as unattributable, not crashed.
  - Subagent output folds into the parent `session_id`.
- `bin_sessions`:
  - Known inputs → expected bin medians, means, stds, counts.
  - Bin with a single session → `std` null / no error bar.
  - Fewer distinct X values than `n_bins` → bins collapse gracefully.

## Out of scope

- No change to the calibration/cap model. This is read-only analysis over existing
  logged data.
- No new data collection. If `calibration_log.parquet` is sparse, bins are sparse;
  acceptable.
- No per-session drill-down from this chart (the existing session table/context-curve
  views already cover single-session inspection).
