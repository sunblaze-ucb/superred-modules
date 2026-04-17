"""Tests for ManyShotOptimizer event-driven state machine."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from many_shot_optimizer.optimizer import ManyShotOptimizer

DOMAIN = SecurityDomainTag(name="test")


def _mock_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_controllable():
    return Controllable(name="input", security_domain=DOMAIN, description="test")


def _make_run_start():
    return RunStartEvent(trajectory=Trajectory())


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(
        controllable=ctrl, request="user input", answer=answer
    )


FAKE_EXAMPLES = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(50)]


async def _init_optimizer(**kwargs) -> ManyShotOptimizer:
    defaults = dict(
        example_counts=[5, 10],
        success_threshold=0.8,
        custom_examples=FAKE_EXAMPLES,
        generate_extra=False,
    )
    defaults.update(kwargs)
    opt = ManyShotOptimizer(**defaults)
    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=[_make_controllable()],
        observables=[],
        llm_client=mock_llm,
    )
    return opt


@pytest.mark.asyncio
async def test_run_start_returns_event_response():
    opt = await _init_optimizer()
    resp = await opt.on_event(_make_run_start())
    assert isinstance(resp, EventResponse)


@pytest.mark.asyncio
async def test_pre_call_injects_many_shot_prompt():
    opt = await _init_optimizer(example_counts=[3])
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert "You are a helpful assistant" in resp.value
    assert "test objective" in resp.value


@pytest.mark.asyncio
async def test_prompt_contains_correct_example_count():
    opt = await _init_optimizer(example_counts=[5])
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert resp.value.count("Assistant:") == 5


@pytest.mark.asyncio
async def test_success_triggers_done():
    opt = await _init_optimizer(success_threshold=0.8)
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = _mock_response(
        '{"score": 0.9, "rationale": "High"}'
    )
    await opt.on_event(_make_post_call(ctrl, "full compliance"))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True


@pytest.mark.asyncio
async def test_escalation_through_schedule():
    opt = await _init_optimizer(example_counts=[3, 6])
    ctrl = _make_controllable()

    # First attempt: 3 examples
    await opt.on_event(_make_run_start())
    resp1 = await opt.on_event(_make_pre_call(ctrl))
    assert resp1.value.count("Assistant:") == 3

    opt.llm.complete.return_value = _mock_response(
        '{"score": 0.2, "rationale": "Low"}'
    )
    await opt.on_event(_make_post_call(ctrl, "refused"))
    end_resp = await opt.on_event(_make_run_end())
    assert end_resp.done is False

    # Second attempt: 6 examples
    await opt.on_event(_make_run_start())
    resp2 = await opt.on_event(_make_pre_call(ctrl))
    assert resp2.value.count("Assistant:") == 6


@pytest.mark.asyncio
async def test_all_attempts_exhausted():
    opt = await _init_optimizer(example_counts=[3])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = _mock_response(
        '{"score": 0.1, "rationale": "Failed"}'
    )
    await opt.on_event(_make_post_call(ctrl, "refused"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_custom_example_counts():
    opt = await _init_optimizer(example_counts=[7, 14, 21])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert resp.value.count("Assistant:") == 7


@pytest.mark.asyncio
async def test_secondary_controllable_ignored():
    opt = await _init_optimizer()
    ctrl1 = _make_controllable()
    ctrl2 = Controllable(name="other", security_domain=DOMAIN, description="secondary")

    await opt.on_event(_make_run_start())
    resp1 = await opt.on_event(_make_pre_call(ctrl1))
    assert isinstance(resp1, ControllableInjection)

    resp2 = await opt.on_event(_make_pre_call(ctrl2))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_evaluator_failure_gives_low_score():
    opt = await _init_optimizer(example_counts=[3, 6])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.side_effect = Exception("LLM down")
    await opt.on_event(_make_post_call(ctrl, "some response"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is False


@pytest.mark.asyncio
async def test_llm_generation_triggered_when_count_exceeds_bundled():
    small_examples = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(5)]
    opt = await _init_optimizer(
        example_counts=[10],
        custom_examples=small_examples,
        generate_extra=True,
    )
    ctrl = _make_controllable()

    # Mock LLM for generator (returns 5 generated examples)
    generated = [{"user": f"GQ{i}", "assistant": f"GA{i}"} for i in range(5)]
    opt.llm.complete.return_value = _mock_response(json.dumps(generated))

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert resp.value.count("Assistant:") == 10
