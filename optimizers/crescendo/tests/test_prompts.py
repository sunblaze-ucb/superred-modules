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


def test_variant_1_contains_max_turns_placeholder():
    variant = get_variant(0)
    assert "{max_turns}" in variant, "Variant 0 missing {max_turns} placeholder"


def test_get_variant_cycles():
    count = get_variant_count()
    assert count == 5
    v0 = get_variant(0)
    v_wrap = get_variant(count)
    assert v0 == v_wrap


def test_variants_are_distinct():
    count = get_variant_count()
    variants = [get_variant(i) for i in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            assert variants[i] != variants[j], f"Variant {i} == Variant {j}"


def test_variant_renders_with_format():
    variant = get_variant(0)
    rendered = variant.format(objective="test goal", max_turns=10)
    assert "test goal" in rendered
    assert "10" in rendered
    assert "{objective}" not in rendered
    assert "{max_turns}" not in rendered


def test_all_variants_render_with_format():
    """All variants should be renderable with objective and max_turns kwargs."""
    for i in range(get_variant_count()):
        variant = get_variant(i)
        rendered = variant.format(objective="test goal", max_turns=10)
        assert "test goal" in rendered
        assert "{objective}" not in rendered
