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


def test_format_at_cap():
    assert render.format_projection(CapProjection(timedelta(0), True)) == "at/over the cap now"
