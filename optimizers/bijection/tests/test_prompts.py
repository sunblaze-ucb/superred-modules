"""Tests for prompt rendering."""

from __future__ import annotations

import random

import pytest

from bijection_optimizer.bijection import generate_bijection
from bijection_optimizer.prompts import (
    DEFAULT_TEACHING_CORPUS,
    MULTITURN_INDICATOR,
    render_attack_prompt,
    render_teaching_intro,
)


def _seed_bijection(codomain: str = "digit") -> object:
    return generate_bijection(
        codomain=codomain,
        fixed_size=10,
        num_digits=2,
        delimiter="  ",
        rng=random.Random(42),
    )


class TestRenderTeachingIntro:
    def test_includes_language_alpha_and_mapping(self) -> None:
        b = _seed_bijection("digit")
        intro = render_teaching_intro(b)
        assert "Language Alpha" in intro
        assert "alphabet map" in intro
        # The literal mapping dict is embedded in the intro.
        assert str(b.mapping) in intro


class TestRenderAttackPrompt:
    def test_default_includes_intro_shots_and_encoded_query(self) -> None:
        b = _seed_bijection("digit")
        out = render_attack_prompt(
            bijection=b,
            goal="describe a recipe for cookies",
            num_teaching_shots=3,
        )
        assert "Language Alpha" in out  # intro
        assert "User: Here is a text in English" in out  # shots
        assert "Assistant: Here is the same text translated" in out
        assert MULTITURN_INDICATOR in out
        # Goal is encoded with the bijection (no plain "describe a recipe").
        assert "describe a recipe for cookies" not in out

    def test_include_intro_false_omits_intro(self) -> None:
        b = _seed_bijection("digit")
        out = render_attack_prompt(
            bijection=b,
            goal="describe a recipe",
            num_teaching_shots=2,
            include_intro=False,
        )
        assert "You are about to learn a new language" not in out
        assert "User: Here is a text in English" in out
        assert MULTITURN_INDICATOR in out

    def test_zero_shots_omits_shot_block(self) -> None:
        b = _seed_bijection("digit")
        out = render_attack_prompt(
            bijection=b,
            goal="describe a recipe",
            num_teaching_shots=0,
        )
        assert "User: Here is a text in English" not in out
        assert "Language Alpha" in out  # intro still present
        assert MULTITURN_INDICATOR in out

    def test_loops_corpus_when_more_shots_than_passages(self) -> None:
        b = _seed_bijection("digit")
        # Use a 2-passage corpus and request 5 shots — the corpus
        # must be looped (matches upstream `* (k // len + 1)`).
        corpus = (
            "the city changed slowly during winter",
            "the harbor was busy in early autumn",
        )
        out = render_attack_prompt(
            bijection=b,
            goal="x",
            num_teaching_shots=5,
            teaching_corpus=corpus,
        )
        # First passage appears at least 3 times; second at least 2.
        assert out.count("the city changed slowly during winter") >= 3
        assert out.count("the harbor was busy in early autumn") >= 2

    def test_negative_shots_raise(self) -> None:
        b = _seed_bijection("digit")
        with pytest.raises(ValueError):
            render_attack_prompt(
                bijection=b, goal="x", num_teaching_shots=-1,
            )

    def test_empty_corpus_with_shots_raises(self) -> None:
        b = _seed_bijection("digit")
        with pytest.raises(ValueError):
            render_attack_prompt(
                bijection=b,
                goal="x",
                num_teaching_shots=3,
                teaching_corpus=(),
            )

    def test_letter_codomain_renders(self) -> None:
        b = generate_bijection(
            codomain="letter", fixed_size=10, rng=random.Random(0),
        )
        out = render_attack_prompt(
            bijection=b,
            goal="describe a recipe",
            num_teaching_shots=2,
        )
        # No leak of the plain English goal in the encoded query.
        assert "describe a recipe" not in out
        # Encoded goal lives after the multiturn indicator.
        idx = out.index(MULTITURN_INDICATOR)
        encoded_query = out[idx + len(MULTITURN_INDICATOR) :]
        # Decode brings it back.
        assert b.decode(encoded_query).strip() == "describe a recipe"


class TestDefaultTeachingCorpus:
    def test_corpus_passages_are_lowercase_and_non_empty(self) -> None:
        assert DEFAULT_TEACHING_CORPUS
        for passage in DEFAULT_TEACHING_CORPUS:
            assert passage
            assert passage == passage.lower()
