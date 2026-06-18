from datetime import date
import polars as pl
import render


def test_build_daily_figure_one_bar_per_series():
    daily = pl.DataFrame({"date": [date(2026, 5, 23)], "laptop": [1.0], "server": [2.0]})
    fig = render.build_daily_figure(daily)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "stack"
    assert fig.layout.yaxis.title.text == "USD"
