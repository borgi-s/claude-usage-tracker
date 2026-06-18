# Reported usage charts + USD daily burn — drop cap calibration

**Date:** 2026-06-18
**Status:** Draft, awaiting user approval before implementation plan

## Problem / motivation

The dashboard has run long enough to accumulate real reported-usage data. Today the two
primary charts do not show what Anthropic actually reports — they **reconstruct** a curve
from `output_tokens ÷ a guessed cap`, driven by the whole anchor-based calibration model
(`caps.py`, the continuous/hour-of-day cap derivation, share columns, window-length
self-correction). That machinery exists only to estimate a cap we no longer need to guess:
the `/api/oauth/usage` endpoint already returns the true utilization percentage, and the
agent has been logging it every few minutes into `calibration_log.parquet`
(`util_5h`, `util_7d`).

This change replaces the reconstructed charts with charts that plot the **reported**
percentage directly, deletes the cap-derivation machinery, and switches the daily-burn
chart from an internal "cost-weighted tokens" unit to **real USD**.

## Decisions (settled with user)

1. **Plan basis — Max 5x only.** The log mixes plans: **Pro** for 2026-05-21→22, then
   **Max 5x** from 2026-05-23 onward (the account was upgraded). Anthropic reports util as
   a % of the *active* plan's cap, so a Pro-era reading of "80%" is 80% of the *Pro* cap
   (≈16% of Max 5x). To keep "100% = Max 5x cap" consistent, the reported charts and their
   KPIs **drop the Pro-era samples** (filter to `rate_limit_tier == "default_claude_max_5x"`).
2. **Daily burn unit — real USD.** Compute actual dollar cost from per-model token counts
   × current Anthropic prices (a new `MODEL_PRICING` table). No `ccusage` runtime dependency —
   the app already parses the same transcripts ccusage reads, so the figure is cross-checkable
   against ccusage.
3. **Keep:** KPI summary strip (re-sourced from reported util) and the per-session table.
   **Drop:** the cost-vs-session-length chart and the calibration-history expander (both are
   artifacts of the calibration model being removed).
4. **Add:** a "time to 100%" projection line in the live panel (recommendation (a)).
   Skip the plan-change shading (b) and proactive new-rate-limit alerting (c) for v1.

## Verified data realities (checked against the live `calibration_log.parquet`)

These drive several design choices below — verify again at implementation time, but they held on 2026-06-18:

- **`resets_5h_iso` / `resets_7d_iso` carry ~jitter and are NOT stable window labels.** Among
  429 Max-5x samples there are **428 distinct `resets_5h_iso` strings** — nearly one per sample —
  and 421 consecutive "changes" are <120s apart (jitter, not a real reset). **Raw string equality
  on the reset column is therefore useless** for both the chart's reset-break detection and the
  projection's in-window grouping. Both must parse the reset ISO to a datetime and treat samples
  as the same window when their reset *times* are within a tolerance (the existing
  `metrics.session_cost_attribution` already does this with a ~1h tolerance — reuse that idea).
- **Max-5x 5h cap-hits (util ≥ 0.99): ~4**, not ~20. (The earlier "20" counted the Pro era we are
  dropping.) The red-dot overlay and its smoke test should expect a handful, not twenty.
- **Max-5x weekly peak: 0.92** (no weekly cap-hit yet — the 100% weekly line is a reference only).

## Approach

Middle-ground rewrite, not a from-scratch rebuild and not a surgical patch:

- **Keep collection untouched.** The agent keeps polling and appending `util_5h`/`util_7d`/
  `resets_*_iso`/`subscription_type`/`rate_limit_tier` to `calibration_log.parquet`. That log
  *is* the new chart data. The filename stays (renaming would orphan the parquet already in
  Supabase and break continuity) — a comment notes "calibration" is now a misnomer; it's a
  reported-usage sample log.
- **Delete cap derivation.** It is dead once "reported" needs no guessed cap.
- **Replace** the two reconstructed charts with two that plot the reported numbers straight
  from the log.

## New page layout (both `app.py` and `app_cloud.py`)

1. **Live panel** *(keep, trimmed)* — current 5h % + weekly %, "resets in", plus the new
   "time to 100%" projection. Drops the "calibrated cap ≈ XM" caption.
2. **KPI strip** *(keep, re-sourced)* — see below.
3. **Chart 1 — "5h usage limit (reported)"** *(new)*.
4. **Chart 2 — "Weekly usage limit (reported)"** *(new)*.
5. **Chart 3 — "Daily burn (USD)"** *(keep, re-unit + by-machine)*.
6. **Per-session table** *(keep, unchanged)*.

## Data sources (existing plumbing, no change to download paths)

| Surface | Source | Scope |
|---|---|---|
| Reported charts (1 & 2) + their KPIs | `calibration_log.parquet` | Account-wide, from the canonical poller prefix (`CLOUD_CAPS_PREFIX`) — already downloaded by the cloud viewer |
| Daily burn + session table | merged `cache.parquet` | Both machines (laptop + server), via `merge_cache_parquets` |

The reported charts are account-wide (Anthropic does not break utilization down per project),
so the sidebar **project/model filter no longer affects charts 1 & 2** — only the daily burn
and session table. A caption on each reported chart states this.

## Chart 1 & 2 — reported usage

**Y-axis:** the real reported percentage — `util_5h` (chart 1) / `util_7d` (chart 2), plotted
as `value * 100`. Max-5x samples only.

**Line breaks (no misleading diagonals).** Plot one tooth per window. Insert a gap (Plotly
`connectgaps=False`, i.e. a `None` Y between segments) when EITHER:
- **a real reset occurred** — parse `resets_5h_iso` / `resets_7d_iso` to datetimes and break when
  the parsed reset time *jumps forward by more than a tolerance* vs. the previous sample
  (5h: > 30 min; weekly: > 1 day). **Do NOT break on raw string inequality** — jitter makes the
  raw string change on nearly every sample (see Verified data realities), which would split the
  series into single points and draw no line at all. The tolerance is far above the observed
  ≤120s jitter and far below a real 5h/7d window length, so it cleanly separates jitter from
  resets; **or**
- **the sampling gap exceeds a threshold** (no data; e.g. the observed ~34h hole). Default
  `gap_break_minutes = 15` for 5h (3 missed 5-min polls) and `60` for weekly.

**Reference lines:** dashed `20%` = "Pro cap (est.)" (Pro ≈ 1/5 of Max 5x), dotted `100%` =
"Max 5x cap". Both always shown (the old "Show Max 5x line too" toggle is removed).

**Cap-hit markers:** overlay red dots where reported util ≥ 0.99 ("we hit the cap"); hover
shows the timestamp. The 5h chart has ~4 such points in the current Max-5x data; weekly has none
yet (peak 0.92) but uses identical logic.

**Range buttons:** reuse the existing `1D/5D/14D/1M/All` `rangeselector` from
`render._rangeselector_xaxis()`. Calendar bands (`add_calendar_bands`) carry over.

## Chart 3 — Daily burn (USD)

**Y-axis:** real `$` per day. Each row's dollar cost =
`Σ (tokens_of_type × price_of_type_for_model)`, summed per local day.

**Default stack: by machine** (laptop vs server) — directly reflects "my two main uses". The
`machine` column exists ONLY on the cloud merged cache (`cache.merge_cache_parquets`); the local
single-machine `cache.parquet` / `metrics.add_derived` output has no such column. So every
machine-consuming code path (the daily chart, and any KPI) must guard `"machine" in df.columns`
and, when absent, fall back to a **single series labelled `"local"`** (no hostname lookup — the
codebase has no hostname source today and we won't add one). A toggle keeps the existing
main-vs-subagent split.

**`MODEL_PRICING`** (new, in `config.py`) — USD per 1,000,000 tokens, current Anthropic rates
(skill `claude-api`, cached 2026-06-04). Cache-write at 1.25× input (5-minute ephemeral, Claude
Code's default), cache-read at 0.1× input:

| Model prefix | input | output | cache_write (1.25×) | cache_read (0.1×) |
|---|---|---|---|---|
| `claude-opus-4-` | 5.00 | 25.00 | 6.25 | 0.50 |
| `claude-sonnet-4-` | 3.00 | 15.00 | 3.75 | 0.30 |
| `claude-haiku-4-` | 1.00 | 5.00 | 1.25 | 0.10 |
| `claude-fable-5` | 10.00 | 50.00 | 12.50 | 1.00 |

`price_for(model)` does **explicit longest-prefix** matching (the most specific prefix wins).
Note this is intentionally NOT the same as `config.context_window_for`, which returns the first
`startswith` match in dict order — do not "mirror" that function; implement longest-prefix so e.g.
`claude-fable-5` and `claude-opus-4-` can't collide. Historical 3.x models that appear in old
rows get explicit entries (`claude-3-opus` 15/75, `claude-3-5-sonnet` 3/15, `claude-3-7-sonnet`
3/15, `claude-3-5-haiku` 0.80/4, `claude-3-haiku` 0.25/1.25). Unknown / `<synthetic>` models fall
back to Sonnet-tier pricing **and log a warning** (no silent $0).

> Caveat to document inline: the parser stores a single `cache_creation_input_tokens` column,
> not split by 5m/1h TTL, so cache-write is always priced at 1.25× (the 5m default). 1h-cache
> writes (2×) would be slightly under-counted — acceptable, and noted in the chart caption.

## KPI strip — re-sourced from reported util

`render.render_kpis` is re-pointed; computed from the Max-5x-filtered log (charts) and the
cache ($):

| KPI | New definition |
|---|---|
| Total $ | Σ dollar cost across all cache rows |
| Daily avg | Total $ ÷ span-days |
| Peak 5h % | `max(util_5h)` over Max-5x samples |
| Peak weekly % | `max(util_7d)` over Max-5x samples |
| 5h windows over Pro | # distinct `resets_5h_iso` windows whose max `util_5h` > 0.20 |
| Weeks over Pro | # distinct `resets_7d_iso` windows whose max `util_7d` > 0.20 |

These are more honest than today's reconstructed `cumulative_total`-based versions.

## Live panel — "time to 100%" projection

In `render.render_live_panel_from_cache` (cloud) and `app._render_usage_view` (local):
estimate the recent burn rate `Δutil/Δt` from the last few log samples in the **in-progress
window**, then extrapolate from the current `util_5h` to `util = 1.0`.

- **Identifying the in-progress window:** group samples by parsed reset time within a ~1h
  tolerance (the same jitter-tolerant approach `session_cost_attribution` uses), NOT by raw
  `resets_5h_iso` string equality — string equality yields ~1 sample per "window" (see Verified
  data realities) and the projection would always show "—". No window-length constant is needed,
  so this does not depend on the deleted `effective_window_hours`.
- Render as *"at current 5h burn → 100% in ~1h20m"*. Cap the ETA at the time remaining until the
  parsed reset: if 100% would arrive only after the window resets, show *"won't hit 100% before
  reset"*. Show "—" when util is flat/declining or fewer than 2 samples exist in the window.
- **Cloud signature change:** `render_live_panel_from_cache` currently reads only `caps.json` and
  has no access to the log; to compute the slope it must additionally take the recent log (pass
  the already-loaded calibration_log, or the last-N rows). Flag this in the plan — it's a new
  parameter on that function.

Drops the existing "calibrated cap ≈ XM" caption entirely.

## New / changed helpers

| Location | Helper | Purpose |
|---|---|---|
| `metrics.py` (new) | `reported_util_series(log, kind, gap_break_minutes)` | Filter to Max 5x, sort, emit `(sampled_at, util)` rows with `None`-separated segments at reset/gap boundaries (reset detected via parsed-time jump > tolerance, NOT raw string change); also returns the cap-hit subset (util ≥ 0.99). |
| `metrics.py` (new) | `_window_id(log, kind)` (shared helper) | Assign a stable integer window id per sample by parsed-reset-time grouping with jitter tolerance. Used by `reported_util_series`, `windows_over_threshold`, and `project_time_to_cap` so all three agree on what a "window" is. |
| `metrics.py` (new) | `windows_over_threshold(log, kind, threshold)` | Count `_window_id` groups whose peak util exceeds `threshold` (KPI). |
| `metrics.py` (new) | `peak_reported(log, kind)` | `max(util)` over Max-5x samples (KPI). |
| `metrics.py` (new) | `project_time_to_cap(log, now, kind)` | Burn-rate extrapolation over the current `_window_id` group, for the live projection. |
| `config.py` (new) | `MODEL_PRICING`, `price_for(model)` | USD price table + explicit longest-prefix lookup (see Chart 3). |
| `metrics.py` (changed) | `dollar_cost` column in `add_derived` | Per-row `$` from token columns × `price_for(model)`. `add_derived` already row-wise-weights tokens with `COST_WEIGHTS`; the price-driven column slots in the same way. Safe on both caches (model + token columns exist in both). |
| `metrics.py` (rewrite) | `daily_stacked(df, by)` | Today returns `(date, main, subagent)` summing `cost_weighted_tokens` (pivot on `is_subagent`). Rewrite to sum `dollar_cost` and pivot on either `is_subagent` (main/sub) OR `machine` (per-machine columns). Returns long form or a dimension-tagged frame so `render_daily_bar` can build traces from arbitrary series names. |
| `render.py` (new) | `render_reported_usage_chart(log, kind, ...)` | Replaces `render_5h_chart` + `render_weekly_chart` (one function, `kind` param). |
| `render.py` (rewrite) | `render_daily_bar(daily, decomposition_key)` | Y in `$`; default stack by machine + main-vs-sub toggle. NOTE: the current trace loop hardcodes `daily["main"]`/`daily["subagent"]` — it must be rewritten to iterate arbitrary series (machine names are not fixed). |
| `render.py` (changed) | `render_kpis(...)` | New signature per the KPI table — see KPI-ripple note below. |

## Files deleted / gutted

**Deletion principle:** grep every call site before deleting. The lists below were expanded
after a code review found several now-orphaned helpers the first draft missed — delete the whole
dependency chain, don't leave broken-but-unused functions.

| File | Change |
|---|---|
| `render.py` | Delete `render_calibration_history`, `render_cost_vs_session_length`, `_cost_vs_session_length_interactive`, `_add_cumulative_traces`, `_peak_for_decomposition`, `render_5h_chart`, `render_weekly_chart` (replaced). |
| `caps.py` | **Reduce to `load_caps`/`save_caps` + a trimmed snapshot dataclass; delete everything else.** Concretely that means deleting not just `global_cap_from_anchors`, `implied_cap_series`, the hour-of-day helpers (`hour_of_day_cap_series`, `hour_of_day_sample_counts`, `attach_hour_of_day_cap`, `calibrate_hourly_to_log`), `derive_continuous_caps`, `derive_from_reading`, but ALSO the now-orphaned `effective_caps`, `cap_series`, `attach_time_varying_cap`, and the private helpers `_per_hour_medians`, `_smooth_rolling_circular`, `_interpolate_empty_circular`. The trimmed dataclass keeps the fields the live panel reads (`sample_util_5h`/`sample_util_7d`/`resets_5h_iso`/`resets_7d_iso`/`sampled_at`/`subscription_type`/`rate_limit_tier`) and drops the derived `max5x_*`/`pro_*`/`sample_burn_*` fields. Re-check whether `config.PRO_CAP_*`/`MAX5X_CAP_*` constants still have any reader after this — if not, delete them too. |
| `metrics.py` | Delete `five_hour_burn_since_reset`, `weekly_burn_since_reset`, `observed_window_lengths`, `effective_window_hours`, `downsample_cumulative`, `bin_sessions`, `five_hour_window_totals`, `session_cost_attribution` (its only caller, the cost-vs-session chart, is gone — see test note), and the now-orphaned `week_start_for` + `_build_week_boundaries` (zero callers after the weekly-burn and per-week deletions — confirm with grep), plus share-column logic. Keep `session_summaries` (powers the session table), compaction detection, `add_derived`, and `daily_stacked` (rewritten per the helper table). |
| `app_cache.py` | Delete `calibrate` and the cumulative/`five_h_window_shares`/`per_week_shares` parts of `filtered_compute` (these consumed `five_hour_window_totals` + `week_start_for`). `daily` now aggregates `$` via the rewritten `daily_stacked`. Confirm `filtered_compute`'s remaining outputs (sessions, daily) don't reference the deleted share columns. |
| `app.py` | Remove the continuous-calibration WRITE path in the live fragment (`derive_continuous_caps` + single-sample fallback + the ~30 lines that build the derived caps, `app.py:173-204`) — replace with a direct save of the trimmed snapshot dataclass. KEEP `calibration_log.append_sample` (the data source) and the Supabase upload of the three files. Rewrite the KPI computation block + the chart-render block. Remove the "Plan caps" sidebar subheader/caption + "Show Max 5x line too" toggle. Add the projection to `_render_usage_view`. |
| `app_cloud.py` | Remove cap derivation (`app_cache.calibrate`, share columns, fallbacks), the calibration-history + cost-vs-session render calls, the "Plan caps" sidebar block + "Show Max 5x line" toggle. Reported charts read the log; daily/session read the cache. Pass the log into `render_live_panel_from_cache` for the projection. |
| `calibration_log.py` | Unchanged schema (continuity with Supabase). Comment that burn/agg columns are now vestigial; only util/resets/plan columns are consumed. |
| `tests/` | See enumerated list in Test plan. |

**KPI-ripple note.** `render_kpis` has exactly two call sites (`app.py` and `app_cloud.py`), each
passing 8 positional args today. Both change together. More importantly, the *inputs* to those
KPIs currently come from `fc.*`/`app_cache` outputs that are being deleted (`five_h_window_shares`,
`per_week_shares`, `cumulative_total`). So the entire KPI-computation block in **both** apps is
rewritten to source util KPIs from the log (`peak_reported`, `windows_over_threshold`) and the
`$` KPIs from `dollar_cost` — not just the `render_kpis` signature.

## Test plan

**Tests to delete** (they exercise removed code — leaving them causes import-time collection
errors): `tests/test_calibration_characterization.py` (whole file — it's entirely
`global_cap_from_anchors`/`observed_window_lengths`/`effective_window_hours`/`*_burn_since_reset`);
the `test_weekly_*` / `test_five_hour_*` / `test_downsample_cumulative_*` cases in
`tests/test_metrics.py`; the `test_bin_sessions_*` cases in `tests/test_session_cost_attribution.py`,
plus that file's `session_cost_attribution` tests (the function is being deleted — delete them
unless we decide to keep the function). Keep `tests/test_cache_merge.py` and
`tests/test_supabase_sync.py` (unaffected).

**Tests to add:**

- **Reset-break uses tolerant matching (regression for the jitter bug).** Build a log where
  consecutive samples in one real window have `resets_5h_iso` strings differing by <120s (jitter)
  AND a later sample whose reset jumps by ~5h (real reset). Assert `reported_util_series` produces
  a CONTINUOUS segment across the jittered samples and a `None` break only at the real reset — i.e.
  it does not fragment the window. Also assert a `None` break across a 40h sampling gap, and that
  all points are Max-5x only (Pro rows dropped).
- **KPI window counts.** Synthetic log with known per-(real-)window peaks; assert
  `windows_over_threshold` groups by tolerant window id and counts only windows whose peak > 0.20.
- **Dollar cost.** Row with known token counts for each model tier; assert `dollar_cost` equals the
  hand-computed sum; assert `price_for` longest-prefix picks the most specific entry; assert an
  unknown model falls back to Sonnet pricing and logs a warning.
- **Daily by-dimension.** Cache with two machines over 3 days; assert `daily_stacked(df, by="machine")`
  groups by (date, machine) and totals match per-row `dollar_cost` sums; assert `by="is_subagent"`
  still yields main/sub; assert graceful single-`"local"`-series fallback when `machine` is absent.
- **Projection.** Window (tolerant-grouped) with steadily rising util; assert `project_time_to_cap`
  returns a finite ETA; flat/declining util or <2 in-window samples returns `None`; an ETA past the
  reset returns the "won't hit before reset" sentinel.
- **Smoke.** Run `app.py` locally; verify the two reported charts show real % with reset breaks
  (continuous teeth, not isolated dots) and ~4 red cap-hit dots on the 5h chart, the daily chart is
  in `$` stacked by machine, the KPI strip and session table render, and the calibration-history /
  cost-vs-session sections are gone. (Local venv is currently broken — see Open risks; use the
  standalone Python 3.12 + `.venv/Lib/site-packages` on `PYTHONPATH` that the data checks used.)

## Open risks / setup notes

- **The project venv is broken.** `.venv` was built from an Anaconda env (`alp_hack`) that no
  longer exists, so `.venv/Scripts/python.exe` errors on every invocation. The data checks in this
  spec ran via a standalone interpreter with the venv's packages on the path:
  `PYTHONIOENCODING=utf-8 PYTHONPATH=".venv/Lib/site-packages" <Python312>/python.exe ...`.
  Before/at implementation, either repair the venv (recreate from a present base interpreter) or
  use that workaround to run tests and the local app. This blocks the smoke test until resolved.

## Non-goals

- No plan-change shading (decision (b) skipped); no proactive rate-limit alerting (c skipped).
- No `ccusage` runtime dependency.
- No rename of `calibration_log.parquet` (Supabase continuity).
- No per-project breakdown of reported utilization (Anthropic doesn't expose it).
- No 5m-vs-1h cache-write price split (single `cache_creation` column; priced at 1.25×).
