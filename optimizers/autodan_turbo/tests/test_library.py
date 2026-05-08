"""Tests for StrategyLibrary."""

from __future__ import annotations

import pytest

from autodan_turbo_optimizer.library import (
    HIGH_SCORE_THRESHOLD,
    LOW_SCORE_THRESHOLD,
    StrategyLibrary,
)


class TestAdd:
    def test_first_add_creates_entry(self) -> None:
        lib = StrategyLibrary()
        lib.add(
            strategy="Storytelling",
            definition="Use narrative",
            example="prompt-1",
            score=3.0,
        )
        assert "Storytelling" in lib
        assert len(lib) == 1
        entry = lib.all()["Storytelling"]
        assert entry["Strategy"] == "Storytelling"
        assert entry["Definition"] == "Use narrative"
        assert entry["Example"] == ["prompt-1"]
        assert entry["Score"] == [3.0]

    def test_second_add_same_name_appends(self) -> None:
        lib = StrategyLibrary()
        lib.add(
            strategy="Storytelling", definition="d1",
            example="prompt-1", score=2.0,
        )
        lib.add(
            strategy="Storytelling", definition="d-NEW",
            example="prompt-2", score=4.0,
        )
        assert len(lib) == 1
        entry = lib.all()["Storytelling"]
        # Definition stays as-is from first add (mirrors upstream merge).
        assert entry["Definition"] == "d1"
        assert entry["Example"] == ["prompt-1", "prompt-2"]
        assert entry["Score"] == [2.0, 4.0]

    def test_different_strategies_coexist(self) -> None:
        lib = StrategyLibrary()
        lib.add(strategy="A", definition="dA", example="pA", score=2.0)
        lib.add(strategy="B", definition="dB", example="pB", score=3.0)
        assert len(lib) == 2
        assert "A" in lib and "B" in lib

    def test_empty_strategy_name_raises(self) -> None:
        lib = StrategyLibrary()
        with pytest.raises(ValueError):
            lib.add(strategy="", definition="d", example="p", score=1.0)


class TestRetrieveEmpty:
    def test_empty_library_returns_valid_empty(self) -> None:
        lib = StrategyLibrary()
        valid, strategies = lib.retrieve()
        assert valid is True
        assert strategies == []

    def test_invalid_k_raises(self) -> None:
        lib = StrategyLibrary()
        with pytest.raises(ValueError):
            lib.retrieve(k=0)


class TestRetrieveTier1HighScore:
    def test_high_avg_returns_single_winner(self) -> None:
        """Avg score >= HIGH_SCORE_THRESHOLD -> single best strategy."""
        lib = StrategyLibrary()
        lib.add(
            strategy="Strong", definition="dS", example="pS",
            score=HIGH_SCORE_THRESHOLD + 1,
        )
        lib.add(
            strategy="Weak", definition="dW", example="pW",
            score=LOW_SCORE_THRESHOLD + 0.1,
        )
        valid, strategies = lib.retrieve(k=3)
        assert valid is True
        assert len(strategies) == 1
        assert strategies[0]["Strategy"] == "Strong"

    def test_high_score_picks_highest_example(self) -> None:
        lib = StrategyLibrary()
        lib.add(
            strategy="Strong", definition="dS",
            example="example-low", score=4.0,
        )
        lib.add(
            strategy="Strong", definition="dS",
            example="example-high", score=8.0,
        )
        valid, strategies = lib.retrieve()
        assert valid is True
        # Avg = 6.0 >= HIGH_SCORE_THRESHOLD; pick highest-scoring example.
        assert strategies[0]["Example"] == "example-high"


class TestRetrieveTier2MediumScore:
    def test_medium_avg_returns_up_to_k(self) -> None:
        lib = StrategyLibrary()
        lib.add(strategy="A", definition="dA", example="pA", score=3.0)
        lib.add(strategy="B", definition="dB", example="pB", score=2.5)
        lib.add(strategy="C", definition="dC", example="pC", score=4.0)
        valid, strategies = lib.retrieve(k=2)
        assert valid is True
        assert len(strategies) == 2
        # Highest avg first.
        assert strategies[0]["Strategy"] == "C"
        assert strategies[1]["Strategy"] == "A"

    def test_medium_avg_strips_score_field(self) -> None:
        lib = StrategyLibrary()
        lib.add(strategy="A", definition="dA", example="pA", score=3.0)
        valid, strategies = lib.retrieve()
        assert valid is True
        assert "Score" not in strategies[0]
        assert set(strategies[0].keys()) == {"Strategy", "Definition", "Example"}


class TestRetrieveTier3IneffectiveScore:
    def test_low_avg_returns_invalid_and_strategies_to_avoid(self) -> None:
        lib = StrategyLibrary()
        lib.add(strategy="A", definition="dA", example="pA", score=1.0)
        lib.add(strategy="B", definition="dB", example="pB", score=1.5)
        valid, strategies = lib.retrieve(k=2)
        assert valid is False
        assert len(strategies) == 2

    def test_low_avg_caps_at_k(self) -> None:
        lib = StrategyLibrary()
        for i in range(5):
            lib.add(
                strategy=f"S{i}", definition=f"d{i}",
                example=f"p{i}", score=1.0,
            )
        valid, strategies = lib.retrieve(k=2)
        assert valid is False
        assert len(strategies) == 2


class TestRetrieveMixedTiers:
    def test_high_score_outranks_medium(self) -> None:
        """A high-tier strategy short-circuits any medium-tier ones."""
        lib = StrategyLibrary()
        lib.add(strategy="Med", definition="dM", example="pM", score=3.0)
        lib.add(strategy="High", definition="dH", example="pH", score=7.0)
        valid, strategies = lib.retrieve(k=5)
        assert valid is True
        assert len(strategies) == 1
        assert strategies[0]["Strategy"] == "High"
