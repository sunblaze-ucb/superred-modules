"""Tests for GEPAAgenticOptimizer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superred.core.channel import EventEnvelope
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from gepa_agentic_optimizer.optimizer import GEPAAgenticOptimizer
from gepa_agentic_optimizer.reflector import ReflectionResult


USER_TAG = SecurityDomainTag("user")
TOOL_TAG = SecurityDomainTag("tools")
SYSTEM_TAG = SecurityDomainTag("system_prompt")
TRACE_TAG = SecurityDomainTag("agent_trace")


class _FakeReadableTrajectory:
    def __init__(self) -> None:
        self._items: list[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def snapshot(self) -> list[Any]:
        return list(self._items)

    def drain(self) -> list[Any]:
        items = list(self._items)
        self._items = []
        return items


def _empty_llm() -> MagicMock:
    llm = MagicMock()

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("tests patch reflector.propose directly")

    llm.complete.side_effect = fail
    return llm


def _user_ctrl(name: str = "user_prompt") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG)


def _system_ctrl() -> Controllable:
    return Controllable(name="system_prompt", security_domain=SYSTEM_TAG)


def _read_ctrl(name: str = "read__slack__get_webpage") -> Controllable:
    return Controllable(
        name=name,
        security_domain=TOOL_TAG,
        description=(
            "Per-read injection point. ControllablePostCallEvent answer carries "
            "the legitimate tool return."
        ),
        value_type="json",
    )


def _opaque_content_ctrl(name: str = "opaque_surface") -> Controllable:
    return Controllable(
        name=name,
        security_domain=TOOL_TAG,
        description="External document content returned to the agent.",
        value_type="json",
    )


def _response_observable(name: str = "agent_trace_tool_response_0000") -> Observable:
    return Observable(name=name, security_domain=TRACE_TAG)


def _failure_eval(score: float = 0.1) -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="not yet",
    )


def _refl(text: str) -> ReflectionResult:
    return ReflectionResult(new_instruction=text, raw_output=text, prompt="")


async def _dispatch_event(opt: GEPAAgenticOptimizer, event: Any) -> Any:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    try:
        await opt._dispatch(envelope)
    except Exception:
        await asyncio.sleep(0)
        if future.done():
            future.exception()
        raise
    return await future


async def _init_optimizer(
    *,
    controllables: list[Controllable] | None = None,
    max_attempts: int = 3,
    target_controllable_name: str | None = None,
    content_controllable_names: list[str] | None = None,
    observables: list[ObservableValue] | None = None,
) -> GEPAAgenticOptimizer:
    opt = GEPAAgenticOptimizer(
        max_attempts=max_attempts,
        target_controllable_name=target_controllable_name,
        content_controllable_names=content_controllable_names,
    )
    await opt.initialize(
        goal=Goal(description="achieve target X"),
        controllables=controllables if controllables is not None else [_read_ctrl()],
        observables=observables if observables is not None else [],
        llm_client=_empty_llm(),
    )
    return opt


class TestSurfaceClassification:
    @pytest.mark.asyncio
    async def test_discovers_agentdojo_read_surfaces_from_initialize(self) -> None:
        opt = await _init_optimizer(
            controllables=[_system_ctrl(), _user_ctrl(), _read_ctrl()],
        )

        assert opt._content_surface_names == ["read__slack__get_webpage"]

    @pytest.mark.asyncio
    async def test_discovers_opaque_content_surface_from_metadata(self) -> None:
        opt = await _init_optimizer(controllables=[_opaque_content_ctrl()])

        assert opt._content_surface_names == ["opaque_surface"]

    @pytest.mark.asyncio
    async def test_constructor_names_mark_opaque_content_surface(self) -> None:
        ctrl = Controllable(name="custom_result", security_domain=TOOL_TAG)
        opt = await _init_optimizer(
            controllables=[ctrl],
            content_controllable_names=["custom_result"],
        )

        assert opt._content_surface_names == ["custom_result"]


class TestPostCallSelection:
    @pytest.mark.asyncio
    async def test_plans_one_deterministic_content_surface_not_first_seen(self) -> None:
        alpha = _read_ctrl("read__alpha")
        beta = _read_ctrl("read__beta")
        opt = await _init_optimizer(controllables=[alpha, beta])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        beta_resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=beta,
                request="beta",
                answer="beta legitimate content",
            ),
        )
        alpha_resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=alpha,
                request="alpha",
                answer="alpha legitimate content",
            ),
        )

        assert opt._planned_content_surface == "read__alpha"
        assert isinstance(beta_resp, ControllableNoInjection)
        assert isinstance(alpha_resp, ControllableInjection)
        assert "alpha legitimate content" in alpha_resp.value
        assert "[GEPA AGENTIC INDIRECT PROMPT]" in alpha_resp.value
        assert "achieve target X" in alpha_resp.value

    @pytest.mark.asyncio
    async def test_missed_planned_surface_rotates_to_next_surface(self) -> None:
        alpha = _read_ctrl("read__alpha")
        beta = _read_ctrl("read__beta")
        opt = await _init_optimizer(controllables=[alpha, beta], max_attempts=3)
        propose = AsyncMock(return_value=_refl("next"))

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._planned_content_surface == "read__alpha"
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._planned_content_surface == "read__beta"

    @pytest.mark.asyncio
    async def test_runtime_discovery_handles_dynamic_tool_call_surface(self) -> None:
        opt = await _init_optimizer(controllables=[], max_attempts=2)
        tool_call = _read_ctrl("tool_call:attacker_lookup")
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=tool_call,
                request="{}",
                answer="fake return",
            ),
        )

        assert isinstance(resp, ControllableInjection)
        assert opt._selected_surface is not None
        assert opt._selected_surface["mode"] == "runtime_discovery"


class TestPromptFallback:
    @pytest.mark.asyncio
    async def test_prompt_fallback_when_no_agentic_content_surface_exists(self) -> None:
        opt = await _init_optimizer(controllables=[_system_ctrl(), _user_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
        )

        assert isinstance(resp, ControllableInjection)
        assert resp.value == "achieve target X"
        assert opt._selected_surface == {
            "name": "system_prompt",
            "type": "system_prompt",
            "event_kind": "pre",
            "mode": "fallback",
        }

    @pytest.mark.asyncio
    async def test_prompt_channels_skipped_when_content_surface_is_planned(self) -> None:
        opt = await _init_optimizer(controllables=[_user_ctrl(), _read_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="hello"),
        )

        assert isinstance(resp, ControllableNoInjection)


class TestRolloutContext:
    @pytest.mark.asyncio
    async def test_reflection_receives_surface_metadata_and_tool_return(self) -> None:
        read_ctrl = _read_ctrl()
        opt = await _init_optimizer(controllables=[read_ctrl], max_attempts=2)
        propose = AsyncMock(return_value=None)
        traj = _FakeReadableTrajectory()

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=read_ctrl,
                request="read",
                answer="legitimate webpage",
            ),
        )
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert rollout.selected_surface["name"] == "read__slack__get_webpage"
        assert rollout.selected_surface["type"] == "content_postcall"
        assert rollout.tool_return == "legitimate webpage"
        assert rollout.observed_surfaces[0]["name"] == "read__slack__get_webpage"

    @pytest.mark.asyncio
    async def test_dynamic_agent_observables_are_included(self) -> None:
        opt = await _init_optimizer(controllables=[_read_ctrl()], max_attempts=2)
        propose = AsyncMock(return_value=None)
        traj = _FakeReadableTrajectory()

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content={"value": "tool response visible to agent"},
            )
        )
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert rollout.response == '{"value": "tool response visible to agent"}'
        assert rollout.agent_observations is not None
        assert "agent_trace_tool_response_0000" in rollout.agent_observations[0]

    @pytest.mark.asyncio
    async def test_static_observables_still_surface(self) -> None:
        observables = [
            ObservableValue(
                observable=Observable(name="model_identity", security_domain=TRACE_TAG),
                content="gpt-4o",
            )
        ]
        opt = await _init_optimizer(observables=observables, max_attempts=2)
        propose = AsyncMock(return_value=None)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert rollout.target_observables == {"model_identity": "gpt-4o"}


class TestDoneSemantics:
    @pytest.mark.asyncio
    async def test_no_feedback_no_response_runs_to_budget(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        propose = AsyncMock(return_value=None)

        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
            end_1 = await _dispatch_event(
                opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
            end_2 = await _dispatch_event(
                opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert isinstance(end_1, RunEndResponse)
        assert isinstance(end_2, RunEndResponse)
        assert end_1.done is False
        assert end_2.done is True
