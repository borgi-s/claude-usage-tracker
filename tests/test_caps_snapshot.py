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
