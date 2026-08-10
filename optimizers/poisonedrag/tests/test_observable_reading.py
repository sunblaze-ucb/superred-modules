"""Regression tests for observable-name matching and per-run retrieval counts."""

from __future__ import annotations

from poisonedrag_optimizer.optimizer import (
    _CORRECT_ANSWER_HINTS,
    _TARGET_ANSWER_HINTS,
    PoisonedRAGOptimizer,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

_TAG = SecurityDomainTag(name="t")


def _obs(name: str, content: str) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name=name, security_domain=_TAG), content=content
    )


def test_incorrect_answer_is_not_read_as_the_correct_answer() -> None:
    """A task exposing only ``incorrect_answer`` has no ground truth to read.

    Hints match as substrings, so the bare ``"answer"`` in
    ``_CORRECT_ANSWER_HINTS`` also matched ``incorrect_answer``. Both fields
    then held the poison target, the generation prompt was told the target was
    the correct answer, and the attack aimed at some third answer the claim's
    judge never checks for.
    """
    observables = [_obs("incorrect_answer", "Lyon")]

    correct = PoisonedRAGOptimizer._read_answer_observable(
        observables, _CORRECT_ANSWER_HINTS, exclude=_TARGET_ANSWER_HINTS
    )
    target = PoisonedRAGOptimizer._read_answer_observable(
        observables, _TARGET_ANSWER_HINTS
    )

    assert correct is None
    assert target == "Lyon"


def test_correct_answer_still_read_when_exposed_alongside_the_target() -> None:
    observables = [_obs("correct_answer", "Paris"), _obs("incorrect_answer", "Lyon")]

    correct = PoisonedRAGOptimizer._read_answer_observable(
        observables, _CORRECT_ANSWER_HINTS, exclude=_TARGET_ANSWER_HINTS
    )
    target = PoisonedRAGOptimizer._read_answer_observable(
        observables, _TARGET_ANSWER_HINTS
    )

    assert correct == "Paris"
    assert target == "Lyon"
