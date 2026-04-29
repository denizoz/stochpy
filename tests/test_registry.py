import pytest

from stochpy.registry import ModelEntry, ModelRegistry, _AutoSelector


def make_registry(**kwargs) -> ModelRegistry:
    return ModelRegistry(tracking=True, **kwargs)


def test_model_access_by_attribute():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6", fast="openai/gpt-4o-mini")
    assert reg.quality.litellm_name == "anthropic/claude-sonnet-4-6"
    assert reg.fast.litellm_name    == "openai/gpt-4o-mini"


def test_missing_model_raises():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    with pytest.raises(AttributeError):
        _ = reg.nonexistent


def test_auto_selector_created():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    assert isinstance(reg.auto, _AutoSelector)


def test_record_and_success_rate():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    reg.record("ctx_a", "quality", True)
    reg.record("ctx_a", "quality", True)
    reg.record("ctx_a", "quality", False)
    assert abs(reg.success_rate("ctx_a", "quality") - 2/3) < 1e-9


def test_success_rate_default_optimistic():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    assert reg.success_rate("unknown_ctx", "quality") == 1.0


def test_best_for_cost_strategy():
    reg = make_registry(
        expensive = "anthropic/claude-opus",
        cheap     = "openai/gpt-4o-mini",
    )
    reg.models["expensive"].cost_per_1k = 0.015
    reg.models["cheap"].cost_per_1k     = 0.001

    best = reg.best_for("ctx", strategy="cost_optimized")
    assert best.name == "cheap"


def test_best_for_quality_strategy():
    reg = make_registry(good="a/model-good", bad="b/model-bad")
    # simulate history: good passes, bad fails
    for _ in range(10):
        reg.record("ctx", "good", True)
        reg.record("ctx", "bad",  False)

    best = reg.best_for("ctx", strategy="quality_optimized")
    assert best.name == "good"


def test_filter_poor_performers():
    reg = make_registry(poor="a/poor", ok="b/ok")
    # poor model has <60% success rate
    for _ in range(10):
        reg.record("ctx", "poor", False)
    for _ in range(10):
        reg.record("ctx", "ok",   True)

    best = reg.best_for("ctx")
    assert best.name == "ok"


def test_stats_returns_dict():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    reg.record("ctx", "quality", True)
    stats = reg.stats()
    assert "ctx" in stats
    assert stats["ctx"]["quality"]["calls"] == 1


def test_history_capped_at_200():
    reg = make_registry(quality="anthropic/claude-sonnet-4-6")
    for _ in range(300):
        reg.record("ctx", "quality", True)
    assert len(reg._scores["ctx"]["quality"]) == 200
