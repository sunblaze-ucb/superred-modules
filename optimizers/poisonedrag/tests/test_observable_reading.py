"""Regression tests for observable-name matching and per-run retrieval counts.

Both go through the real callers.  Testing ``_read_answer_observable`` directly
with an explicit ``exclude=`` proves nothing: the defect was that ``initialize``
omitted that argument, so a helper-level test stays green with the bug restored.
"""

from __future__ import annotations

import asyncio

import pytest
from superred.core.channel import EventEnvelope
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import RunStartEvent
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from poisonedrag_optimizer.optimizer import PoisonBatch, PoisonedRAGOptimizer

_TAG = SecurityDomainTag("rag")


def _obs(name: str, content: str) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name=name, security_domain=_TAG), content=content
    )


async def _dispatch(opt: PoisonedRAGOptimizer, event: Event) -> EventResponse:
    """Go through the envelope path, which is what sets ``current_trajectory``.

    Calling ``on_event`` directly skips the base class's lifecycle hook, so the
    run-end tally would see ``current_trajectory is None`` and return early --
    a test written that way passes no matter what the tally does.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    await opt._dispatch(EventEnvelope(event=event, future=future, loop=loop))
    await future
    return await future


def _ctrl(name: str) -> Controllable:
    return Controllable(name=name, security_domain=_TAG)


async def _init(opt: PoisonedRAGOptimizer, observables: list[ObservableValue]) -> None:
    await opt.initialize(
        Goal(description="What is the capital of France?"),
        [_ctrl("knowledge_base_insert")],
        observables,
        LLMClient._make_noop(),
    )


@pytest.mark.asyncio
async def test_incorrect_answer_is_not_read_as_the_ground_truth() -> None:
    """A task exposing only ``incorrect_answer`` has no ground truth to read.

    Hints match as substrings, so the bare ``"answer"`` in
    ``_CORRECT_ANSWER_HINTS`` also matched ``incorrect_answer``.  Both fields
    then held the poison target, and the generation prompt was told the target
    was the correct answer.
    """
    opt = PoisonedRAGOptimizer(max_attempts=1)
    await _init(opt, [_obs("incorrect_answer", "Lyon")])

    assert opt._correct_answer is None
    assert opt._target_answer == "Lyon"


@pytest.mark.asyncio
async def test_correct_answer_still_read_when_exposed_alongside_the_target() -> None:
    opt = PoisonedRAGOptimizer(max_attempts=1)
    await _init(
        opt, [_obs("correct_answer", "Paris"), _obs("incorrect_answer", "Lyon")]
    )

    assert opt._correct_answer == "Paris"
    assert opt._target_answer == "Lyon"


@pytest.mark.asyncio
async def test_a_postcall_delivered_count_survives_run_end() -> None:
    """Run end must not overwrite a count recorded during the run.

    A poison batch delivered through a PostCall is counted as it arrives.  There
    may be no context observable on the trajectory to rediscover it from, so a
    run-end pass that starts its tally at zero reports 0 for the whole run.
    """
    opt = PoisonedRAGOptimizer(
        target_answer="Lyon", poison_texts=["Poison: Lyon."], max_attempts=2
    )
    await _init(opt, [])
    await _dispatch(opt, RunStartEvent(trajectory=Trajectory()))

    opt._current_batch = PoisonBatch(
        question="q", target_answer="Lyon", documents=("Poison: Lyon.",)
    )
    opt._injected_this_run = True
    opt._record_retrieval_metrics_from_content("retrieved: q.Poison: Lyon.")
    assert opt._last_retrieved_poison_count == 1

    assert opt.current_trajectory is not None, (
        "test setup: the tally would return early"
    )
    opt._record_retrieval_metrics_from_trajectory()  # trajectory has no context observable
    assert opt._last_retrieved_poison_count == 1, (
        "run end clobbered the count recorded in-run"
    )


@pytest.mark.asyncio
async def test_the_count_does_not_carry_across_runs() -> None:
    """The figure describes this run's batch, not the best batch so far.

    ``_last_retrieved_poison_count`` was reset only in ``initialize`` (once per
    task), so once a run retrieved poison every later run inherited the number.
    """
    opt = PoisonedRAGOptimizer(
        target_answer="Lyon", poison_texts=["Poison: Lyon."], max_attempts=3
    )
    await _init(opt, [])

    await _dispatch(opt, RunStartEvent(trajectory=Trajectory()))
    opt._current_batch = PoisonBatch(
        question="q", target_answer="Lyon", documents=("Poison: Lyon.",)
    )
    opt._injected_this_run = True
    opt._record_retrieval_metrics_from_content("retrieved: q.Poison: Lyon.")
    assert opt._last_retrieved_poison_count == 1

    await _dispatch(opt, RunStartEvent(trajectory=Trajectory()))
    assert opt._last_retrieved_poison_count == 0, "the count carried into the next run"
