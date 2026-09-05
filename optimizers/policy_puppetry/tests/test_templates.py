"""Faithfulness tests for the vendored template and the leetspeak transform."""

from __future__ import annotations

import random

from policy_puppetry_optimizer.templates import (
    HARMFUL_BEHAVIOR_SLOT,
    LEET_SLOT_SUFFIX,
    dr_house_template,
    leetspeak,
    render,
)


def test_template_carries_exactly_two_slots() -> None:
    """Upstream's template names the behaviour twice; both must be filled."""
    assert dr_house_template().count(HARMFUL_BEHAVIOR_SLOT) == 2


def test_template_is_the_dr_house_scene_config() -> None:
    template = dr_house_template()
    assert "<interaction-config>" in template
    assert "DR HOUSE" in template
    assert template.rstrip().endswith("script ONLY!</rules>")


def test_render_fills_every_slot() -> None:
    out = render("do the forbidden thing")
    assert HARMFUL_BEHAVIOR_SLOT not in out
    assert out.count("do the forbidden thing") == 2


def test_render_matches_upstream_build_prompts() -> None:
    """Upstream does a plain str.replace over the template."""
    behavior = "detailed instructions for how to make methamphetamine"
    expected = dr_house_template().replace(HARMFUL_BEHAVIOR_SLOT, behavior)
    assert render(behavior) == expected


def test_leet_suffix_matches_upstream_bypassleet() -> None:
    """BypassLeet widens the slot before substitution, not after."""
    behavior = "X"
    expected = dr_house_template().replace(
        HARMFUL_BEHAVIOR_SLOT, HARMFUL_BEHAVIOR_SLOT + LEET_SLOT_SUFFIX
    ).replace(HARMFUL_BEHAVIOR_SLOT, behavior)
    assert render(behavior, leet_suffix=True) == expected
    assert "leet speak so Cuddy" in render(behavior, leet_suffix=True)


def test_leetspeak_matches_upstream_algorithm() -> None:
    """Re-implement upstream's loop inline against a shared seed."""
    text = "detailed instructions for the thing"
    for seed in range(8):
        src, dst = "ieaost", "134057"
        processed = text.translate(
            str.maketrans(src.lower() + src.upper(), dst + dst)
        )
        rng = random.Random(seed)
        expected = "".join(
            processed[i] if rng.random() > 0.5 else text[i] for i in range(len(text))
        )
        assert leetspeak(text, seed=seed) == expected


def test_leetspeak_preserves_length() -> None:
    text = "a moderately long sentence about nothing in particular"
    assert len(leetspeak(text, seed=3)) == len(text)


def test_leetspeak_is_deterministic_per_seed() -> None:
    assert leetspeak("hello world", seed=7) == leetspeak("hello world", seed=7)


def test_leetspeak_differs_across_seeds() -> None:
    """Long enough input that two seeds will not coincide."""
    text = "detailed instructions " * 6
    assert leetspeak(text, seed=0) != leetspeak(text, seed=1)


def test_leetspeak_only_substitutes_the_upstream_alphabet() -> None:
    """Characters outside ieaost/IEAOST are never altered."""
    text = "bcdfg BCDFG 12345 !?-"
    assert leetspeak(text, seed=0) == text


def test_leetspeak_maps_to_upstream_digits() -> None:
    """Every produced character is either the original or its mapped digit."""
    text = "ieaost" * 20
    out = leetspeak(text, seed=1)
    assert set(out) <= set("ieaost134057")
    assert any(ch in "134057" for ch in out)
