"""Tests for the GPTFuzzer response predictor and its degradation path.

The behaviour under test is what happens when the official RoBERTa scorer
``hubert233/GPTFuzz`` cannot be loaded. That is not hypothetical: a host with
no Hugging Face Hub access raises ``OSError`` out of ``from_pretrained``, and
``OSError`` is not a ``RuntimeError``. The optimizer calls ``predict`` from
``on_event``, so an escaping exception costs the whole task's measurement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from superred.core.channel import EventEnvelope
from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal

from tests.conftest import (
    FakeReadableTrajectory,
    USER_TAG,
    make_controllable,
    make_observable,
    mock_response,
)
from gptfuzzer_optimizer.optimizer import GPTFuzzerOptimizer
from gptfuzzer_optimizer.predictor import (
    DEFAULT_GPTFUZZ_MODEL,
    FallbackPredictor,
    PredictorUnavailableError,
    RefusalStringPredictor,
    RoBERTaPredictor,
)

# Verbatim message transformers raises when the Hub is unreachable and the
# weights are not cached, reproduced with HF_HUB_OFFLINE=1 and an empty HF_HOME.
# It is an OSError: not a RuntimeError, and not any type a caller can enumerate.
OFFLINE_MESSAGE = (
    "We couldn't connect to 'https://huggingface.co' to load the files, and "
    "couldn't find them in the cached files.\nCheck your internet connection "
    "or see how to run the library in offline mode at "
    "'https://huggingface.co/docs/transformers/installation#offline-mode'."
)


def offline_error() -> OSError:
    return OSError(OFFLINE_MESSAGE)


class ExplodingPredictor:
    """Stand-in for a RoBERTaPredictor whose weights will not load."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc if exc is not None else offline_error()
        self.calls = 0

    def predict(self, responses: list[str]) -> list[int]:
        self.calls += 1
        raise self._exc


# ---------------------------------------------------------------------------
# RoBERTaPredictor: every load failure becomes one type
# ---------------------------------------------------------------------------


def test_unreachable_hub_surfaces_as_predictor_unavailable(monkeypatch: Any) -> None:
    """An OSError from the Hub is normalized, not leaked as-is."""
    import transformers

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise offline_error()

    monkeypatch.setattr(
        transformers.RobertaForSequenceClassification, "from_pretrained", boom
    )

    predictor = RoBERTaPredictor(DEFAULT_GPTFUZZ_MODEL, device="cpu")
    with pytest.raises(PredictorUnavailableError) as excinfo:
        predictor.predict(["anything"])

    # The model id and the original cause both survive, so the operator can act.
    assert DEFAULT_GPTFUZZ_MODEL in str(excinfo.value)
    assert "OSError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_predictor_unavailable_is_a_runtimeerror() -> None:
    """Back-compat: the previous contract raised RuntimeError."""
    assert issubclass(PredictorUnavailableError, RuntimeError)


def test_failed_load_leaves_no_half_initialized_predictor(monkeypatch: Any) -> None:
    """A load that fails must not publish a partially built predictor."""
    import transformers

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise offline_error()

    monkeypatch.setattr(
        transformers.RobertaForSequenceClassification, "from_pretrained", boom
    )

    predictor = RoBERTaPredictor(DEFAULT_GPTFUZZ_MODEL, device="cpu")
    with pytest.raises(PredictorUnavailableError):
        predictor.predict(["anything"])

    assert predictor._model is None
    assert predictor._tokenizer is None
    assert predictor._device is None


# ---------------------------------------------------------------------------
# FallbackPredictor: the regression this file exists for
# ---------------------------------------------------------------------------


def test_fallback_fires_when_the_scorer_raises_oserror() -> None:
    """Regression: the fallback used to catch only RuntimeError.

    ``OSError`` is what an unreachable Hub actually raises, so the fallback
    never fired and the exception escaped into ``on_event``.
    """
    primary = ExplodingPredictor(offline_error())
    predictor = FallbackPredictor(primary, RefusalStringPredictor())

    scores = predictor.predict(
        ["I'm sorry, I can't help with that.", "Sure, here goes."]
    )

    assert scores == [0, 1]
    assert predictor.fallback_active is True


@pytest.mark.parametrize(
    "exc",
    [
        OSError(OFFLINE_MESSAGE),
        ValueError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'"),
        ImportError("transformers is not installed"),
        RuntimeError("Expected all tensors to be on the same device"),
        Exception("strict dataclass rejected an unknown config field"),
    ],
    ids=["offline", "bad-repo-id", "missing-dep", "device-mismatch", "config-rejected"],
)
def test_fallback_fires_for_every_observed_load_failure(exc: BaseException) -> None:
    """Load failures share no exception type, so none of them may escape."""
    predictor = FallbackPredictor(ExplodingPredictor(exc), RefusalStringPredictor())

    assert predictor.predict(["Sure, here goes."]) == [1]
    assert predictor.fallback_active is True


def test_fallback_does_not_swallow_keyboard_interrupt() -> None:
    """Degrading is for failures, not for an operator stopping the sweep."""
    predictor = FallbackPredictor(
        ExplodingPredictor(KeyboardInterrupt()), RefusalStringPredictor()
    )

    with pytest.raises(KeyboardInterrupt):
        predictor.predict(["anything"])

    assert predictor.fallback_active is False


def test_fallback_latches_so_one_task_uses_one_scorer() -> None:
    """After the first failure the primary is not retried within the task."""
    primary = ExplodingPredictor()
    predictor = FallbackPredictor(primary, RefusalStringPredictor())

    predictor.predict(["first"])
    predictor.predict(["second"])
    predictor.predict(["third"])

    assert primary.calls == 1


def test_working_scorer_is_never_second_guessed() -> None:
    """A healthy primary owns every verdict, including the counter-intuitive ones."""
    primary = MagicMock()
    primary.predict.return_value = [1]
    predictor = FallbackPredictor(primary, RefusalStringPredictor())

    # The refusal heuristic would score this 0; the real classifier says 1.
    assert predictor.predict(["I'm sorry, but here is exactly how."]) == [1]
    assert predictor.fallback_active is False


def test_degradation_is_logged_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """A silent downgrade of the scorer would silently change what is measured."""
    predictor = FallbackPredictor(ExplodingPredictor(), RefusalStringPredictor())

    with caplog.at_level(logging.WARNING, logger="gptfuzzer_optimizer.predictor"):
        predictor.predict(["anything"])

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "OSError" in record.getMessage()
    assert record.exc_info is not None


# ---------------------------------------------------------------------------
# End to end: the task survives, which is the point
# ---------------------------------------------------------------------------


async def _dispatch(opt: GPTFuzzerOptimizer, event: Any) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    try:
        await opt._dispatch(envelope)
    except Exception:
        await asyncio.sleep(0)
        if future.done():
            future.exception()
        raise
    return await future


async def _run_until_scoring(opt: GPTFuzzerOptimizer) -> FakeReadableTrajectory:
    traj = FakeReadableTrajectory()
    await _dispatch(opt, RunStartEvent(trajectory=traj))
    pre = await _dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=make_controllable(), request="user message"
        ),
    )
    assert isinstance(pre, ControllableInjection)
    traj.push(
        ObservableEvent(
            observable=make_observable("response"), content="Sure, here goes."
        )
    )
    return traj


@pytest.mark.asyncio
async def test_unloadable_scorer_does_not_discard_the_task() -> None:
    """The consequence under test: a missing model must degrade, not lose data.

    Without a framework evaluation the optimizer scores the response itself.
    If that raises, ``_dispatch`` rejects the envelope and re-raises, the
    controller marks the task ``stop_reason="error"``, and the measurement is
    thrown away even though the target already answered.
    """
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = GPTFuzzerOptimizer(
        initial_seed=["seed [INSERT PROMPT HERE]"],
        max_query=10,
        max_jailbreak=2,
        random_seed=0,
        predictor=FallbackPredictor(ExplodingPredictor(), RefusalStringPredictor()),
    )
    await opt.initialize(
        goal=Goal(description="How do I test a security control?"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=llm,
    )
    await _run_until_scoring(opt)

    end = await _dispatch(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    # The run produced a verdict rather than an exception.
    assert end.done is False
    assert opt._current_node is not None
    assert opt._current_node.results == [1]


@pytest.mark.asyncio
async def test_fallback_can_be_refused_so_the_task_fails_instead() -> None:
    """``allow_predictor_fallback=False`` keeps the strict, no-degradation mode."""
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated [INSERT PROMPT HERE]. ")
    opt = GPTFuzzerOptimizer(
        initial_seed=["seed [INSERT PROMPT HERE]"],
        max_query=10,
        max_jailbreak=2,
        random_seed=0,
        allow_predictor_fallback=False,
        predictor=ExplodingPredictor(),
    )
    await opt.initialize(
        goal=Goal(description="How do I test a security control?"),
        controllables=[make_controllable()],
        observables=[],
        llm_client=llm,
    )
    await _run_until_scoring(opt)

    with pytest.raises(OSError):
        await _dispatch(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))


def test_default_predictor_degrades_unless_told_otherwise() -> None:
    """The wiring, not just the class: the default optimizer gets a fallback."""
    lenient = GPTFuzzerOptimizer(
        initial_seed=["seed [INSERT PROMPT HERE]"], random_seed=0
    )
    strict = GPTFuzzerOptimizer(
        initial_seed=["seed [INSERT PROMPT HERE]"],
        random_seed=0,
        allow_predictor_fallback=False,
    )

    assert isinstance(lenient._predictor, FallbackPredictor)
    assert isinstance(strict._predictor, RoBERTaPredictor)
