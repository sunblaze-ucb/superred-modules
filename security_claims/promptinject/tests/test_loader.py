"""Tests for the PromptInject prompt builder."""

from __future__ import annotations

import pytest
from promptinject_claim.loader import ROGUE_STRINGS, load_items
from promptinject_claim._vendor.promptinject import prompt_data


def test_rogue_string_classes() -> None:
    assert ROGUE_STRINGS == ("hate-humans", "kill-humans", "long-prompt")


def test_default_builds_100_per_class() -> None:
    items = load_items()
    assert len(items) == 300
    counts = {r: sum(1 for i in items if i.rogue_key == r) for r in ROGUE_STRINGS}
    assert counts == {"hate-humans": 100, "kill-humans": 100, "long-prompt": 100}


def test_limit_none_builds_full_set() -> None:
    items = load_items(rogue_strings=["hate-humans"], limit=None)
    # garak's config yields 700 built prompts per class (140 base x 5 attacks).
    assert len(items) == 700


def test_rogue_string_matches_upstream_literals() -> None:
    for r in ROGUE_STRINGS:
        item = load_items(rogue_strings=[r], limit=1)[0]
        assert item.rogue_string == prompt_data.rogue_strings[r]


def test_hate_humans_literal() -> None:
    item = load_items(rogue_strings=["hate-humans"], limit=1)[0]
    assert item.rogue_string == "I hate humans"


def test_every_prompt_is_nonempty() -> None:
    assert all(i.prompt.strip() for i in load_items(limit=20))


def test_shuffle_is_seed_deterministic() -> None:
    a = load_items(rogue_strings=["hate-humans"], limit=10, seed=7)
    b = load_items(rogue_strings=["hate-humans"], limit=10, seed=7)
    assert [i.prompt for i in a] == [i.prompt for i in b]


def test_different_seeds_differ() -> None:
    a = load_items(rogue_strings=["hate-humans"], limit=10, seed=0)
    b = load_items(rogue_strings=["hate-humans"], limit=10, seed=1)
    assert [i.prompt for i in a] != [i.prompt for i in b]


def test_unknown_rogue_string_raises() -> None:
    with pytest.raises(ValueError, match="unknown rogue string"):
        load_items(rogue_strings=["nope"])


def test_bad_limit_raises() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        load_items(limit=0)
