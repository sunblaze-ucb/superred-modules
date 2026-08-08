"""Tests for GEPAAgenticOptimizer."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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
from gepa_agentic_optimizer.reflector import ReflectionResult, RolloutRecord


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


def _classifying_llm(roles: dict[str, str]) -> MagicMock:
    """An LLM whose ``complete`` returns a fixed surface->role classification.

    Lets a test exercise the LLM-driven surface interpretation path
    (``surface_llm.classify_controllables``) deterministically.
    """
    llm = MagicMock()

    async def complete(_messages: Any, **_kwargs: Any) -> Any:
        content = json.dumps(roles)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    llm.complete.side_effect = complete
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
        # A read/content surface consumes the injected value as a raw string
        # (see agentdojo runtime_wrapper), so its truthful value_type is text.
        value_type="text",
    )


def _opaque_content_ctrl(name: str = "opaque_surface") -> Controllable:
    return Controllable(
        name=name,
        security_domain=TOOL_TAG,
        description="External document content returned to the agent.",
        value_type="json",
    )


def _response_observable(name: str = "agent_trace_message_0000") -> Observable:
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


async def _deliver_run(
    opt: GEPAAgenticOptimizer,
    score: float,
    *,
    ctrl: Controllable | None = None,
    propose: AsyncMock | None = None,
) -> None:
    """Drive one run in which the content surface actually fires.

    A run that injects nothing is deliberately left unscored, so any test about
    scoring or candidate acceptance has to deliver the payload first.
    """
    ctrl = ctrl if ctrl is not None else _read_ctrl()
    await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
    await _dispatch_event(
        opt,
        ControllablePostCallEvent(
            controllable=ctrl, request="read", answer="legitimate content"
        ),
    )
    with patch.object(
        opt._reflector,
        "propose",
        new=propose if propose is not None else AsyncMock(return_value=None),
    ):
        await _dispatch_event(
            opt,
            RunEndEvent(evaluation=_failure_eval(score), security_domain=USER_TAG),
        )


async def _init_optimizer(
    *,
    controllables: list[Controllable] | None = None,
    max_attempts: int = 3,
    target_controllable_name: str | None = None,
    content_controllable_names: list[str] | None = None,
    observables: list[ObservableValue] | None = None,
    max_content_injections_per_run: int = 3,
    max_pool_size: int = 8,
    response_observable_names: list[str] | None = None,
    llm: MagicMock | None = None,
) -> GEPAAgenticOptimizer:
    opt = GEPAAgenticOptimizer(
        max_attempts=max_attempts,
        target_controllable_name=target_controllable_name,
        content_controllable_names=content_controllable_names,
        max_content_injections_per_run=max_content_injections_per_run,
        max_pool_size=max_pool_size,
        response_observable_names=response_observable_names,
    )
    await opt.initialize(
        goal=Goal(description="achieve target X"),
        controllables=controllables if controllables is not None else [_read_ctrl()],
        observables=observables if observables is not None else [],
        llm_client=llm if llm is not None else _empty_llm(),
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
    async def test_discovers_opaque_content_surface_via_llm(self) -> None:
        # An opaque, non-canonically-named surface is discovered by the attacker's
        # LLM reading its description (not by a hard-coded keyword table).
        opt = await _init_optimizer(
            controllables=[_opaque_content_ctrl()],
            llm=_classifying_llm({"opaque_surface": "content-injection"}),
        )

        assert opt._content_surface_names == ["opaque_surface"]

    @pytest.mark.asyncio
    async def test_opaque_surface_ignored_without_llm_signal(self) -> None:
        # No name/prefix match and no LLM classification -> not a content surface.
        opt = await _init_optimizer(controllables=[_opaque_content_ctrl()])

        assert opt._content_surface_names == []

    @pytest.mark.asyncio
    async def test_constructor_names_mark_opaque_content_surface(self) -> None:
        ctrl = Controllable(name="custom_result", security_domain=TOOL_TAG)
        opt = await _init_optimizer(
            controllables=[ctrl],
            content_controllable_names=["custom_result"],
        )

        assert opt._content_surface_names == ["custom_result"]

    @pytest.mark.asyncio
    async def test_discovers_inspect_agent_tool_return_surface(self) -> None:
        ctrl = _read_ctrl("tool:get_balance")
        opt = await _init_optimizer(controllables=[ctrl])

        assert opt._content_surface_names == ["tool:get_balance"]

    @pytest.mark.asyncio
    async def test_precall_only_surface_not_armed_as_postcall_content(self) -> None:
        # Content is a ROLE; PostCall firing is a TIMING fact about the target. The
        # LLM answers only the role question, so it labels a PreCall-only surface
        # (dtap_scaffold's filesystem: attacker files the agent later reads) as
        # content. That role is real, but the surface never returns a value the
        # agent reads back, so it must not be armed as a PostCall content surface.
        # Timing is read from the target's own description ("PreCall").
        precall = Controllable(
            name="filesystem",
            security_domain=TOOL_TAG,
            description=(
                "Attacker write access to the target machine's filesystem BEFORE "
                "the run: place files the agent later reads. PreCall, fired once "
                "during run setup."
            ),
            value_type="json",
        )
        postcall = Controllable(
            name="env_tool:gmail",
            security_domain=TOOL_TAG,
            description=(
                "Replace the value any gmail MCP tool returns to the agent "
                "(indirect prompt injection). PostCall, once per tool call."
            ),
            value_type="text",
        )
        llm = _classifying_llm(
            {"filesystem": "content-injection", "env_tool:gmail": "content-injection"}
        )
        opt = await _init_optimizer(controllables=[precall, postcall], llm=llm)

        # The PreCall surface is excluded despite its content role; the PostCall
        # surface with the identical role is armed.
        assert opt._content_surface_names == ["env_tool:gmail"]
        assert opt._can_fire_postcall(postcall) is True
        assert opt._can_fire_postcall(precall) is False

    @pytest.mark.asyncio
    async def test_discovers_html_environment_surface_at_runtime(self) -> None:
        ctrl = Controllable(name="opaque_surface", security_domain=TOOL_TAG)
        opt = await _init_optimizer(controllables=[], max_attempts=2)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=ctrl,
                request="read",
                answer="<html><body><input aria-label='email'></body></html>",
            ),
        )

        assert isinstance(resp, ControllableInjection)
        assert opt._selected_surface is not None
        assert "HTML/environment" in opt._selected_surface["classification_reason"]

    @pytest.mark.asyncio
    async def test_json_content_surface_routes_through_fill_value(self) -> None:
        # A schema-typed (json) content surface is LLM-classified as content, then
        # its value is produced by fill_value (valid JSON), not silently declined.
        ctrl = _opaque_content_ctrl("env_inject:gmail")  # value_type="json"
        fill = '{"injection_mcp_tool": "gmail:inject_email", "kwargs": {"body": "PWN"}}'

        def _dispatch(messages: list[dict[str, str]], **_kw: Any) -> Any:
            system = messages[0]["content"]
            content = (
                fill
                if "crafting the exact value" in system
                else ('{"env_inject:gmail": "environment-write"}')
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        llm = MagicMock()

        async def complete(messages: Any, **kwargs: Any) -> Any:
            return _dispatch(messages, **kwargs)

        llm.complete.side_effect = complete

        opt = await _init_optimizer(controllables=[ctrl], llm=llm)
        assert opt._content_surface_names == ["env_inject:gmail"]
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=ctrl, request="read", answer="genuine content"
            ),
        )

        assert isinstance(resp, ControllableInjection)
        assert json.loads(resp.value)["kwargs"]["body"] == "PWN"

    @pytest.mark.asyncio
    async def test_json_content_surface_declines_when_fill_fails(self) -> None:
        # When fill_value cannot produce valid JSON, the surface is declined
        # (no malformed value emitted, no crash).
        ctrl = _opaque_content_ctrl("env_inject:gmail")

        def _dispatch(messages: list[dict[str, str]], **_kw: Any) -> Any:
            system = messages[0]["content"]
            content = (
                "no json here"
                if "crafting the exact value" in system
                else ('{"env_inject:gmail": "environment-write"}')
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        llm = MagicMock()

        async def complete(messages: Any, **kwargs: Any) -> Any:
            return _dispatch(messages, **kwargs)

        llm.complete.side_effect = complete

        opt = await _init_optimizer(controllables=[ctrl], llm=llm)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=ctrl, request="read", answer="genuine content"
            ),
        )

        assert isinstance(resp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_agentdojo_contract_shapes_are_classified(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        agentdojo_src = repo_root / "targets" / "agentdojo" / "src"
        sys.path.insert(0, str(agentdojo_src))
        try:
            from agentdojo_target.controllables import (  # type: ignore[import-not-found]
                READ_CTRLS,
                TOOL_CATALOG_REWRITE_DOC_CTRL,
            )
            from agentdojo_target.observables import (  # type: ignore[import-not-found]
                TOOL_CATALOG_LISTING_OBS,
            )
        finally:
            try:
                sys.path.remove(str(agentdojo_src))
            except ValueError:
                pass

        read_ctrl = READ_CTRLS["slack__get_webpage"]
        opt = await _init_optimizer(
            controllables=[read_ctrl, TOOL_CATALOG_REWRITE_DOC_CTRL],
            observables=[
                ObservableValue(
                    observable=TOOL_CATALOG_LISTING_OBS,
                    content=[
                        {
                            "name": "slack__get_webpage",
                            "description": "Fetch a web page.",
                        }
                    ],
                )
            ],
            max_attempts=2,
        )

        assert "read__slack__get_webpage" in opt._content_surface_names
        assert opt._tool_catalog == [
            {"name": "slack__get_webpage", "description": "Fetch a web page."}
        ]

        traj = _FakeReadableTrajectory()
        traj.push(
            ControllablePostCallEvent(
                controllable=read_ctrl,
                request="read",
                answer={"value": "agent saw this"},
            )
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(
            opt,
            RunEndEvent(evaluation=_failure_eval(0.1), security_domain=USER_TAG),
        )

        assert opt._pool[0].rollouts[-1].response == '{"value": "agent saw this"}'

    @pytest.mark.asyncio
    async def test_inspect_agent_contract_fixture_is_classified(self) -> None:
        inspect_tool_ctrl = Controllable(
            name="tool:get_balance",
            security_domain=TOOL_TAG,
            description=(
                "Replace the value the 'get_balance' tool returns to the agent "
                "(indirect prompt injection). Fired once per call to this tool; "
                "the event's answer carries the legitimate output."
            ),
            value_type="text",
        )
        catalog_obs = Observable(
            name="tool_catalog_listing",
            security_domain=TRACE_TAG,
            observable_type="json",
        )
        opt = await _init_optimizer(
            controllables=[inspect_tool_ctrl],
            observables=[
                ObservableValue(
                    observable=catalog_obs,
                    content=[
                        {
                            "name": "get_balance",
                            "description": "Read account balance.",
                        }
                    ],
                )
            ],
            max_attempts=2,
        )

        assert opt._content_surface_names == ["tool:get_balance"]
        assert opt._tool_catalog == [
            {"name": "get_balance", "description": "Read account balance."}
        ]


class TestPostCallSelection:
    @pytest.mark.asyncio
    async def test_injects_multiple_compatible_content_surfaces_with_cap(self) -> None:
        alpha = _read_ctrl("read__alpha")
        beta = _read_ctrl("read__beta")
        gamma = _read_ctrl("read__gamma")
        opt = await _init_optimizer(
            controllables=[alpha, beta, gamma],
            max_content_injections_per_run=2,
        )
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
        gamma_resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=gamma,
                request="gamma",
                answer="gamma legitimate content",
            ),
        )

        assert opt._planned_content_surface == "read__alpha"
        assert isinstance(beta_resp, ControllableInjection)
        assert isinstance(alpha_resp, ControllableInjection)
        assert "alpha legitimate content" in alpha_resp.value
        assert "[GEPA AGENTIC INDIRECT PROMPT]" not in alpha_resp.value
        assert "achieve target X" in alpha_resp.value
        assert alpha_resp.value == "alpha legitimate content\n\nachieve target X"
        assert isinstance(gamma_resp, ControllableNoInjection)

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

    @pytest.mark.asyncio
    async def test_multiple_injections_are_reflected_in_rollout_metadata(self) -> None:
        alpha = _read_ctrl("read__alpha")
        beta = _read_ctrl("read__beta")
        opt = await _init_optimizer(controllables=[alpha, beta], max_attempts=2)
        propose = AsyncMock(return_value=None)

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=alpha,
                request="alpha",
                answer="alpha legitimate content",
            ),
        )
        await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=beta,
                request="beta",
                answer="beta legitimate content",
            ),
        )
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert len(rollout.selected_surface["all_injected_surfaces"]) == 2


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
    async def test_prompt_channels_skipped_when_content_surface_is_planned(
        self,
    ) -> None:
        opt = await _init_optimizer(controllables=[_user_ctrl(), _read_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="hello"),
        )

        assert isinstance(resp, ControllableNoInjection)


class TestSurfaceLadder:
    """The prompt channels reopen, monotonically, when nothing was delivered.

    A planned content surface is a deferred channel: a PostCall that only fires if
    the agent calls the corresponding tool. Whether it will fire is unknowable when
    the prompt PreCall is decided, so the only evidence is the previous runs. Each
    run whose planned content surfaces all missed deepens the ladder by one, and the
    depth never shrinks.
    """

    @pytest.mark.asyncio
    async def test_prompt_channel_reopens_after_the_content_surface_never_fires(
        self,
    ) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _read_ctrl()], max_attempts=4
        )

        # Run 1: content surface planned, prompt declined, PostCall never fires.
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        first = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="book a flight"
            ),
        )
        assert isinstance(first, ControllableNoInjection)
        with patch.object(opt._reflector, "propose", new=AsyncMock(return_value=None)):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )
        assert opt._ladder_depth == 1

        # Run 2: the content surface missed, so the prompt is eligible again.
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        second = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="book a flight"
            ),
        )

        assert isinstance(second, ControllableInjection)
        assert second.value == "achieve target X"

    @pytest.mark.asyncio
    async def test_reopening_is_monotone_across_a_later_delivery(self) -> None:
        # The resetting predecessor closed the prompt channel again as soon as one
        # content delivery landed, so a flaky surface could re-blind the optimizer
        # every other run. Depth only grows.
        read_ctrl = _read_ctrl()
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), read_ctrl], max_attempts=5
        )

        # Run 1: nothing fires -> depth 1.
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._reflector, "propose", new=AsyncMock(return_value=None)):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )
        assert opt._ladder_depth == 1

        # Run 2: prompt is written AND the content surface fires this time.
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        prompt_resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="book a flight"
            ),
        )
        content_resp = await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=read_ctrl, request="read", answer="legitimate page"
            ),
        )
        assert isinstance(prompt_resp, ControllableInjection)
        assert isinstance(content_resp, ControllableInjection)
        with patch.object(opt._reflector, "propose", new=AsyncMock(return_value=None)):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        # Run 3: the delivery did not close the channel again.
        assert opt._ladder_depth == 1
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        third = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="book a flight"
            ),
        )

        assert isinstance(third, ControllableInjection)

    @pytest.mark.asyncio
    async def test_prompt_channel_stays_closed_while_the_content_surface_delivers(
        self,
    ) -> None:
        # No miss has occurred, so the ladder never leaves rung 0 and the run keeps
        # exactly one attributable surface.
        read_ctrl = _read_ctrl()
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), read_ctrl], max_attempts=3
        )

        await _deliver_run(opt, 0.2, ctrl=read_ctrl)

        assert opt._ladder_depth == 0
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="hello"),
        )

        assert isinstance(resp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_precall_only_scope_eventually_injects_instead_of_declining(
        self,
    ) -> None:
        # The s5 shape: the scope grants the system prompt plus surfaces that are
        # only ever written before the run, so no PostCall can ever fire. When such
        # a surface is armed as content (a target that does not annotate timing in
        # its description), rung 0 declines the system prompt. The ladder must not
        # leave it declined for the whole 20-run budget.
        filesystem = Controllable(
            name="filesystem",
            security_domain=TOOL_TAG,
            description="Place files on the target machine that the agent reads.",
            value_type="text",
        )
        skill = Controllable(
            name="skill",
            security_domain=TOOL_TAG,
            description="Install a skill document the agent loads.",
            value_type="text",
        )
        llm = _classifying_llm(
            {"filesystem": "content-injection", "skill": "content-injection"}
        )
        opt = await _init_optimizer(
            controllables=[_system_ctrl(), filesystem, skill],
            max_attempts=4,
            llm=llm,
        )
        assert opt._content_surface_names == ["filesystem", "skill"]

        injected: list[bool] = []
        for _ in range(3):
            await _dispatch_event(
                opt, RunStartEvent(trajectory=_FakeReadableTrajectory())
            )
            # Only PreCall events ever arrive at this scope.
            resp = await _dispatch_event(
                opt,
                ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
            )
            await _dispatch_event(
                opt,
                ControllablePreCallEvent(controllable=filesystem, request="{}"),
            )
            injected.append(isinstance(resp, ControllableInjection))
            with patch.object(
                opt._reflector, "propose", new=AsyncMock(return_value=None)
            ):
                await _dispatch_event(
                    opt,
                    RunEndEvent(
                        evaluation=_failure_eval(0.0), security_domain=USER_TAG
                    ),
                )

        assert injected == [False, True, True]


class TestEmptyUserTurn:
    """A user prompt reached with an empty request is the agent's only task.

    On a DTAP direct task the scaffold has no benign instruction to put in the
    user turn, so it reaches the user_prompt controllable with ``request=""`` and
    passes whatever comes back to the agent container as its message. Declining
    it launches an agent with no task, which scores zero for a harness reason
    rather than a defence, so this surface is always filled.
    """

    @pytest.mark.asyncio
    async def test_empty_user_turn_is_filled_while_content_surface_delivers(
        self,
    ) -> None:
        opt = await _init_optimizer(controllables=[_user_ctrl(), _read_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request=""),
        )

        assert isinstance(resp, ControllableInjection)
        assert resp.value == "achieve target X"

    @pytest.mark.asyncio
    async def test_empty_user_turn_is_filled_after_the_system_prompt_took_the_run(
        self,
    ) -> None:
        # Targets offer the system prompt before the user prompt, so the first-wins
        # lock settles on the system prompt. The empty user turn is still filled.
        opt = await _init_optimizer(controllables=[_system_ctrl(), _user_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        system_resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
        )
        user_resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request=""),
        )

        assert isinstance(system_resp, ControllableInjection)
        assert isinstance(user_resp, ControllableInjection)
        assert user_resp.value == "achieve target X"

    @pytest.mark.asyncio
    async def test_benign_user_turn_is_still_declined_once_a_surface_holds_the_run(
        self,
    ) -> None:
        # The exemption is only for an empty request. A user turn that carries a
        # benign instruction is left alone, so an indirect task keeps the task the
        # injected content is supposed to subvert.
        opt = await _init_optimizer(controllables=[_system_ctrl(), _user_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
        )
        user_resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="book a flight"
            ),
        )

        assert isinstance(user_resp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_empty_user_turn_is_filled_on_every_run(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _read_ctrl()], max_attempts=3
        )

        for _ in range(2):
            await _dispatch_event(
                opt, RunStartEvent(trajectory=_FakeReadableTrajectory())
            )
            resp = await _dispatch_event(
                opt,
                ControllablePreCallEvent(controllable=_user_ctrl(), request=""),
            )
            assert isinstance(resp, ControllableInjection)
            with patch.object(
                opt._reflector, "propose", new=AsyncMock(return_value=None)
            ):
                await _dispatch_event(
                    opt,
                    RunEndEvent(
                        evaluation=_failure_eval(0.0), security_domain=USER_TAG
                    ),
                )

    @pytest.mark.asyncio
    async def test_empty_user_turn_does_not_adopt_an_unrelated_empty_post_call(
        self,
    ) -> None:
        # Filling the empty user turn records `request=""` as the run's PreCall
        # request. An unrelated PostCall that also carries an empty request is not
        # the same channel, so it must not be adopted as the response surface.
        opt = await _init_optimizer(controllables=[_user_ctrl(), _read_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request=""),
        )

        await _dispatch_event(
            opt,
            ControllablePostCallEvent(
                controllable=_read_ctrl("weather_tool"),
                request="",
                answer="SUNNY 25C",
            ),
        )

        assert opt._primary_post_controllable is None
        assert opt._pending_post_answer is None


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
            ControllablePostCallEvent(
                controllable=_read_ctrl(),
                request="read",
                answer={"value": "tool response visible to agent"},
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
        assert any(
            "tool response visible to agent" in item
            for item in rollout.agent_observations
        )

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

    @pytest.mark.asyncio
    async def test_explicit_response_allowlist_disables_response_heuristics(
        self,
    ) -> None:
        opt = await _init_optimizer(
            controllables=[_read_ctrl()],
            max_attempts=2,
            response_observable_names=["final_answer"],
        )
        propose = AsyncMock(return_value=None)
        traj = _FakeReadableTrajectory()
        traj.push(
            ObservableEvent(
                observable=_response_observable("assistant_response"),
                content="heuristic response should not be selected",
            )
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable("final_answer"),
                content="explicit response",
            )
        )
        traj.push(
            ControllablePostCallEvent(
                controllable=_read_ctrl(),
                request="read",
                answer="postcall tool content should not be selected",
            )
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert rollout.response == "explicit response"
        assert rollout.agent_observations is not None
        assert any(
            "heuristic response should not be selected" in item
            for item in rollout.agent_observations
        )
        assert any(
            "postcall tool content should not be selected" in item
            for item in rollout.agent_observations
        )

    @pytest.mark.asyncio
    async def test_default_response_detection_still_uses_heuristics(self) -> None:
        opt = await _init_optimizer(controllables=[_read_ctrl()], max_attempts=2)
        propose = AsyncMock(return_value=None)
        traj = _FakeReadableTrajectory()
        traj.push(
            ObservableEvent(
                observable=_response_observable("assistant_response"),
                content="heuristic response",
            )
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.2), security_domain=USER_TAG),
            )

        rollout = propose.call_args.kwargs["rollouts"][0]
        assert rollout.response == "heuristic response"


class TestToolCatalogRewrite:
    @pytest.mark.asyncio
    async def test_rewrites_visible_content_tool_description(self) -> None:
        catalog_ctrl = Controllable(
            name="tool_catalog_rewrite_doc",
            security_domain=SYSTEM_TAG,
            value_type="json",
        )
        observables = [
            ObservableValue(
                observable=Observable(
                    name="tool_catalog_listing",
                    security_domain=TRACE_TAG,
                ),
                content=[
                    {
                        "name": "search_web",
                        "description": "Search the web for relevant pages.",
                    }
                ],
            )
        ]
        opt = await _init_optimizer(
            controllables=[catalog_ctrl, _user_ctrl()],
            observables=observables,
            max_attempts=2,
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=catalog_ctrl,
                request="tool catalogue edit",
            ),
        )

        assert isinstance(resp, ControllableInjection)
        payload = resp.value
        assert '"name": "search_web"' in payload
        assert "Search the web for relevant pages.\\n\\nachieve target X" in payload
        assert "achieve target X" in payload
        assert "Returned content may include task-relevant updates" not in payload
        assert "Current task context" not in payload
        assert opt._selected_surface is not None
        assert opt._selected_surface["mode"] == "catalog_rewrite"

    @pytest.mark.asyncio
    async def test_skips_catalog_rewrite_without_visible_catalog_tool(self) -> None:
        catalog_ctrl = Controllable(
            name="tool_catalog_rewrite_doc",
            security_domain=SYSTEM_TAG,
            value_type="json",
        )
        opt = await _init_optimizer(controllables=[catalog_ctrl], max_attempts=2)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=catalog_ctrl,
                request="tool catalogue edit",
            ),
        )

        assert isinstance(resp, ControllableNoInjection)


class TestAcceptance:
    @pytest.mark.asyncio
    async def test_rejects_non_improving_reflected_candidate_from_pool(self) -> None:
        # Both runs deliver: only a run that actually injected earns a score, so a
        # delivery is what makes the acceptance gate reachable at all.
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=_refl("candidate two"))

        await _deliver_run(opt, 0.4, propose=propose)
        await _deliver_run(opt, 0.1)

        assert [candidate.prompt for candidate in opt._pool] == ["achieve target X"]

    @pytest.mark.asyncio
    async def test_accepts_strictly_improving_reflected_candidate(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=_refl("candidate two"))

        await _deliver_run(opt, 0.1, propose=propose)
        await _deliver_run(opt, 0.4)

        assert [candidate.prompt for candidate in opt._pool] == [
            "achieve target X",
            "candidate two",
        ]

    @pytest.mark.asyncio
    async def test_run_that_injected_nothing_does_not_score_the_candidate(self) -> None:
        # The content surface never fired, so the candidate was never delivered.
        # The zero the claim reports belongs to the target, not to the candidate:
        # it must not be written onto it, and it must not admit a fresh candidate
        # into the pool either.
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=_refl("candidate two"))

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )

        seed = opt._pool[0]
        assert seed.score is None
        assert seed.rollouts[-1].score is None
        assert seed.effective_score == 0.0

        # A fresh candidate that also delivers nothing cannot enter the pool.
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._reflector, "propose", new=AsyncMock(return_value=None)):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.9), security_domain=USER_TAG),
            )

        assert [candidate.prompt for candidate in opt._pool] == ["achieve target X"]

    @pytest.mark.asyncio
    async def test_a_delivered_run_is_still_scored(self) -> None:
        opt = await _init_optimizer(max_attempts=3)

        await _deliver_run(opt, 0.3)

        assert opt._pool[0].score == 0.3
        assert opt._pool[0].effective_score == 0.3

    @pytest.mark.asyncio
    async def test_does_not_grow_pool_without_score_signal(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=_refl("candidate two"))
        traj = _FakeReadableTrajectory()
        traj.push(
            ObservableEvent(
                observable=_response_observable("assistant_response"),
                content="visible response",
            )
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.1), security_domain=USER_TAG),
            )

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        with patch.object(opt._reflector, "propose", new=AsyncMock(return_value=None)):
            await _dispatch_event(
                opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert [candidate.prompt for candidate in opt._pool] == ["achieve target X"]

    @pytest.mark.asyncio
    async def test_prunes_pool_to_configured_size(self) -> None:
        opt = await _init_optimizer(max_attempts=5, max_pool_size=2)
        for idx, score in enumerate((0.1, 0.2, 0.3), start=1):
            candidate = type(opt._pool[0])(prompt=f"candidate {idx}", rolled_out=True)
            candidate.rollouts.append(
                RolloutRecord(
                    goal="achieve target X", prompt=candidate.prompt, score=score
                )
            )
            opt._pool.append(candidate)

        opt._prune_pool()

        assert [candidate.prompt for candidate in opt._pool] == [
            "achieve target X",
            "candidate 3",
        ]


class TestDoneSemantics:
    @pytest.mark.asyncio
    async def test_no_delivery_no_feedback_no_response_does_not_spend_attempt(
        self,
    ) -> None:
        opt = await _init_optimizer(max_attempts=2)
        propose = AsyncMock(return_value=None)

        with patch.object(opt._reflector, "propose", new=propose):
            await _dispatch_event(
                opt, RunStartEvent(trajectory=_FakeReadableTrajectory())
            )
            end_1 = await _dispatch_event(
                opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            await _dispatch_event(
                opt, RunStartEvent(trajectory=_FakeReadableTrajectory())
            )
            end_2 = await _dispatch_event(
                opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert isinstance(end_1, RunEndResponse)
        assert isinstance(end_2, RunEndResponse)
        assert end_1.done is False
        assert end_2.done is False
        assert opt._attempt == 0
        propose.assert_not_called()

    @pytest.mark.asyncio
    async def test_framework_feedback_is_authoritative_even_if_trace_looks_successful(
        self,
    ) -> None:
        opt = await _init_optimizer(max_attempts=2)
        propose = AsyncMock(return_value=None)
        traj = _FakeReadableTrajectory()
        traj.push(
            ObservableEvent(
                observable=_response_observable("assistant_response"),
                content="this looks like a success locally",
            )
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        with patch.object(opt._reflector, "propose", new=propose):
            end = await _dispatch_event(
                opt,
                RunEndEvent(evaluation=_failure_eval(0.0), security_domain=USER_TAG),
            )

        assert isinstance(end, RunEndResponse)
        assert end.done is False
        assert opt._succeeded is False
        assert opt._attempt == 1
