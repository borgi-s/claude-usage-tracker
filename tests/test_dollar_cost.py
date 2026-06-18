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
