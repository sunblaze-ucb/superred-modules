"""Tests for prompt variant loading and rendering."""

from crescendo_optimizer.prompts import get_variant, get_variant_count


def test_get_variant_returns_string():
    variant = get_variant(0)
    assert isinstance(variant, str)
    assert len(variant) > 100


def test_get_variant_contains_objective_placeholder():
    for i in range(get_variant_count()):
        variant = get_variant(i)
        assert "{objective}" in variant, f"Variant {i} missing {{objective}} placeholder"


def test_get_variant_contains_max_turns_placeholder():
    for i in range(get_variant_count()):
        variant = get_variant(i)
        assert "{max_turns}" in variant, f"Variant {i} missing {{max_turns}} placeholder"


def test_get_variant_cycles():
    count = get_variant_count()
    assert count >= 3
    v0 = get_variant(0)
    v_wrap = get_variant(count)
    assert v0 == v_wrap


def test_variants_are_distinct():
    v0 = get_variant(0)
    v1 = get_variant(1)
    v2 = get_variant(2)
    assert v0 != v1
    assert v1 != v2
    assert v0 != v2


def test_variant_renders_with_format():
    variant = get_variant(0)
    rendered = variant.format(objective="test goal", max_turns=10)
    assert "test goal" in rendered
    assert "10" in rendered
    assert "{objective}" not in rendered
    assert "{max_turns}" not in rendered
