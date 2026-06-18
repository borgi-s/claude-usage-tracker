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
