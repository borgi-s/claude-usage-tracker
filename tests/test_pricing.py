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
