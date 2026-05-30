# Claude Usage Tracker

Two-piece dashboard for monitoring Claude Code token usage against subscription caps.

- **Local agent** (`app.py`) — runs on the machine that has your Claude Code data. Reads JSONL transcripts, polls Anthropic's `oauth/usage` endpoint, continuously calibrates the Max-5x cap against your own 100% utilization moments, syncs three small data files to Supabase Storage every 5 minutes.
- **Cloud viewer** (`app_cloud.py`) — read-only Streamlit app deployed on Streamlit Community Cloud. Downloads from Supabase every 5 minutes, renders the same charts. Phone-accessible.

## Calibration model in one paragraph

Each row of your local JSONL is cost-weighted (`input=1, cache_write=1.25, cache_read=0.1, output=5`). The live panel polls Anthropic every 5 minutes and stores `(util, burn_in_window, resets_at)` in `calibration_log.parquet`. The 5h cap is the **median across ≥95% utilization anchors of `burn_in_chart_window / util`** — meaning the chart's cumulative at every 100% anchor moment equals 100% by construction. No hour-of-day adjustment (too few anchors per hour for that to be reliable). The weekly cap is computed the same way once you hit ≥95% on the weekly window. `FIVE_HOUR_WINDOW_HOURS` auto-corrects from observed resets once you have ≥5 of them in the log.

## Setup

### 1. Supabase

1. Create or pick a Supabase project (free tier fine).
2. Storage → New bucket → name `usage-tracker`, **private**.
3. SQL Editor → run:
   ```sql
   create policy "anon read usage-tracker objects"
   on storage.objects for select
   to anon
   using (bucket_id = 'usage-tracker');
   ```
4. Project Settings → API → copy `service_role` and `anon` keys.

### 2. Local agent

```
git clone <this repo>
cd claude-usage-tracker
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
# Edit .env with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
.venv\Scripts\streamlit run app.py
```

### 3. Cloud viewer (Streamlit Community Cloud)

1. Push to GitHub.
2. share.streamlit.io → New app → point at `app_cloud.py`.
3. Settings → Secrets:
   ```
   # Top-level keys are also exposed as environment variables.
   CLOUD_USER_PREFIXES = "borgi,borgi-linux"
   CLOUD_CAPS_PREFIX   = "borgi-linux"

   [supabase]
   url = "https://YOUR_PROJECT_REF.supabase.co"
   anon_key = "eyJ..."
   bucket = "usage-tracker"
   ```
   - `CLOUD_USER_PREFIXES` — comma-separated machine prefixes to merge into one view
     (each machine's agent uploads under its own prefix). The merged cache gets a
     `machine` column per row. The legacy single `CLOUD_USER_PREFIX` still works as a
     fallback when this is unset (it just shows one machine).
   - `CLOUD_CAPS_PREFIX` — which prefix supplies the account-wide caps/calibration
     (set it to the always-on poller, e.g. `borgi-linux`). Defaults to the first entry
     of `CLOUD_USER_PREFIXES` if unset.
4. Save. Bookmark the `*.streamlit.app` URL.

## Module map

- `app.py`, `app_cloud.py` — Streamlit entry points
- `render.py` — shared chart/panel rendering
- `parser.py`, `cache.py` — JSONL → parquet
- `metrics.py` — cost-weighting, rolling/fixed windows, share-based cumulative, window-length self-correction
- `caps.py` — global-cap-from-anchors derivation; legacy hour-of-day helpers kept for diagnostics
- `calibration_log.py` — append-only sample log with reset timestamps
- `anthropic_client.py` — OAuth + `/api/oauth/usage`
- `supabase_sync.py` — Storage upload/download
- `config.py` — knobs (TZ, window length default, cost weights, weekly reset)
