"""Tests for FlipAttackOptimizer event-driven state machine."""

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

from flip_attack_optimizer.optimizer import FlipAttackOptimizer

DOMAIN = SecurityDomainTag(name="test")


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _make_controllable():
    return Controllable(name="input", security_domain=DOMAIN, description="test")


def _make_run_start():
    traj = Trajectory()
    return RunStartEvent(trajectory=traj)


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(
        controllable=ctrl, request="user input", answer=answer
    )


async def _init_optimizer(**kwargs) -> FlipAttackOptimizer:
    defaults = dict(
        flip_modes=["FWO", "FCW"],
        use_cot=False,
        use_few_shot=False,
        success_threshold=0.8,
    )
    defaults.update(kwargs)
    opt = FlipAttackOptimizer(**defaults)
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
async def test_pre_call_injects_flipped_task():
    opt = await _init_optimizer(flip_modes=["FWO"])
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert "TASK" in resp.value
    # FWO reverses word order: "test objective" -> "objective test"
    assert "objective test" in resp.value


@pytest.mark.asyncio
async def test_post_call_scores_and_returns_no_injection():
    opt = await _init_optimizer()
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = _mock_response(
        '{"score": 0.3, "rationale": "Low"}'
    )
    resp = await opt.on_event(_make_post_call(ctrl, "some response"))
    assert isinstance(resp, ControllableNoInjection)


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
async def test_strategy_rotation():
    opt = await _init_optimizer(flip_modes=["FWO", "FCW"])
    ctrl = _make_controllable()

    # First attempt (FWO)
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = _mock_response(
        '{"score": 0.2, "rationale": "Low"}'
    )
    await opt.on_event(_make_post_call(ctrl, "refused"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is False

    # Second attempt (FCW)
    await opt.on_event(_make_run_start())
    assert opt._attempt == 1
    resp2 = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp2, ControllableInjection)
    # FCW reverses chars in each word: "test" -> "tset", "objective" -> "evitcejbo"
    assert "tset" in resp2.value or "evitcejbo" in resp2.value


@pytest.mark.asyncio
async def test_all_strategies_exhausted():
    opt = await _init_optimizer(flip_modes=["FWO"])
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
async def test_system_prompt_exposed():
    opt = await _init_optimizer(flip_modes=["FWO"], prompt_style="vanilla")
    await opt.on_event(_make_run_start())
    assert opt.system_prompt is not None
    assert isinstance(opt.system_prompt, str)
    assert len(opt.system_prompt) > 50
