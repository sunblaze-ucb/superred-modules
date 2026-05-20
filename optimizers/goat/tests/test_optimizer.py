"""Tests for the GOATOptimizer event-driven state machine.

Covers:
* RunStart / PreCall / PostCall / RunEnd dispatch and state.
* Multi-turn within a single run (one run = one ``K``-turn conversation).
* Trajectory-first response recovery; PostCall fallback; empty-M fallback.
* Cross-attempt: fresh attacker per run.
* Early-stop on ``RunEndEvent.evaluation.success``.
* writable ``system_prompt`` PreCall is used without stealing the primary lock.
* All four adversarial information-access settings.
* Benign fallback on attacker parse failure.
"""

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

from goat_optimizer.attacker import AttackerParseError, AttackerTurn
from goat_optimizer.optimizer import GOATOptimizer

USER_TAG = SecurityDomainTag("user")
RESPONSE_TAG = SecurityDomainTag("response_readable")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_ctrl(name: str = "user_message") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG)


def _response_ctrl(name: str = "response") -> Controllable:
    return Controllable(name=name, security_domain=RESPONSE_TAG)


def _system_prompt_ctrl() -> Controllable:
    return Controllable(name="system_prompt", security_domain=SYSTEM_PROMPT_TAG)


def _response_observable(name: str = "response") -> Observable:
    return Observable(name=name, security_domain=RESPONSE_TAG)


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
    """Stand-in LLMClient that errors if called.

    All optimizer tests patch ``opt._attacker.next_turn`` so the
    underlying LLM should never be invoked.
    """
    llm = MagicMock()

    async def fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(
            "Underlying LLM should not be called; tests patch next_turn"
        )

    llm.complete.side_effect = fail
    return llm


async def _init_optimizer(
    *,
    max_turns: int = 5,
    max_attempts: int = 1,
    attacks=None,
    max_no_response_runs: int = 0,
    controllables: list[Controllable] | None = None,
    observables: list[ObservableValue] | None = None,
) -> GOATOptimizer:
    opt = GOATOptimizer(
        max_turns=max_turns,
        max_attempts=max_attempts,
        attacks=attacks,
        max_no_response_runs=max_no_response_runs,
    )
    await opt.initialize(
        goal=Goal(description="test goal"),
        controllables=controllables if controllables is not None else [_user_ctrl()],
        observables=observables if observables is not None else [],
        llm_client=_empty_llm(),
    )
    return opt


def _success_eval(score: float = 0.95) -> EvaluationResult:
    return EvaluationResult(
        success=True,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="judge says success",
    )


def _failure_eval(score: float = 0.1) -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="judge says no",
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


async def _dispatch_event(opt, event):
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


class TestConstruction:
    def test_rejects_invalid_max_turns(self) -> None:
        with pytest.raises(ValueError):
            GOATOptimizer(max_turns=0)

    def test_rejects_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            GOATOptimizer(max_attempts=0)

    def test_rejects_empty_attacks(self) -> None:
        with pytest.raises(ValueError):
            GOATOptimizer(attacks=())


# ---------------------------------------------------------------------------
# Run start
# ---------------------------------------------------------------------------


class TestRunStart:
    @pytest.mark.asyncio
    async def test_run_start_constructs_attacker(self) -> None:
        opt = await _init_optimizer()
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._attacker is not None

    @pytest.mark.asyncio
    async def test_static_observables_are_added_to_attacker_prompt(self) -> None:
        model_observable = Observable(
            name="target_model",
            security_domain=SYSTEM_PROMPT_TAG,
            description="Target model identifier",
        )
        opt = await _init_optimizer(
            observables=[ObservableValue(model_observable, "gpt-4.1-mini")]
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        assert opt._attacker is not None
        assert "SUPERRED STATIC TARGET CONTEXT" in opt._attacker.system_prompt
        assert "target_model" in opt._attacker.system_prompt
        assert "gpt-4.1-mini" in opt._attacker.system_prompt

    @pytest.mark.asyncio
    async def test_static_context_is_bounded(self) -> None:
        model_observable = Observable(
            name="target_model",
            security_domain=SYSTEM_PROMPT_TAG,
            description="Target model identifier",
        )
        opt = GOATOptimizer(static_context_max_chars=64)
        await opt.initialize(
            goal=Goal(description="test goal"),
            controllables=[_user_ctrl()],
            observables=[ObservableValue(model_observable, "x" * 200)],
            llm_client=_empty_llm(),
        )

        assert opt._static_context is not None
        assert len(opt._static_context) <= 64
        assert "[truncated]" in opt._static_context

    @pytest.mark.asyncio
    async def test_run_start_resets_per_attempt_state(self) -> None:
        opt = await _init_optimizer()
        opt._turn = 5
        opt._attempt_done = True
        opt._last_injected_value = "stale"

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        assert opt._turn == 0
        assert opt._attempt_done is False
        assert opt._last_injected_value is None


# ---------------------------------------------------------------------------
# Pre-call: injection + skip + lock
# ---------------------------------------------------------------------------


class TestPreCall:
    @pytest.mark.asyncio
    async def test_pre_call_injects_attacker_response(self) -> None:
        opt = await _init_optimizer()
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "ATTACK PROMPT"),
            ),
        ):
            resp = await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
            )

        assert isinstance(resp, ControllableInjection)
        assert resp.value == "ATTACK PROMPT"
        assert opt._turn == 1
        assert opt._awaiting_target_response is True

    @pytest.mark.asyncio
    async def test_locks_to_first_user_controllable(self) -> None:
        """After locking, other (non-system_prompt) controllables are ignored."""
        opt = await _init_optimizer()
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl_user = _user_ctrl()
        ctrl_other = Controllable(
            name="other_user_channel", security_domain=USER_TAG,
        )

        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "first payload"),
            ),
        ):
            first = await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl_user, request="r1"),
            )
        assert isinstance(first, ControllableInjection)

        second = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl_other, request="r2"),
        )
        assert isinstance(second, ControllableNoInjection)
        assert opt._primary_pre_controllable == ctrl_user

    @pytest.mark.asyncio
    async def test_writable_system_prompt_injects_without_stealing_primary_lock(self) -> None:
        """A pre-loop system_prompt PreCall should use the capability when scoped.

        ChatbotTarget shape: emits ``system_prompt`` PreCall before the
        ``user_message`` loop. GOAT injects a target-side system prompt
        extension when writable, without locking that slot, so the
        first user_message PreCall can still become the primary
        injection channel.
        """
        opt = await _init_optimizer(controllables=[_system_prompt_ctrl(), _user_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="seed",
            ),
        )
        assert isinstance(sp_resp, ControllableInjection)
        assert sp_resp.value.startswith("seed")
        assert "authorized red-team evaluation" in sp_resp.value
        assert opt._primary_pre_controllable is None

        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "PAYLOAD"),
            ),
        ):
            user_resp = await _dispatch_event(opt,
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request="seed",
                ),
            )
        assert isinstance(user_resp, ControllableInjection)
        assert user_resp.value == "PAYLOAD"

    @pytest.mark.asyncio
    async def test_unscoped_system_prompt_pre_call_is_skipped_without_locking(self) -> None:
        opt = await _init_optimizer(controllables=[_user_ctrl()])
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await _dispatch_event(
            opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="seed",
            ),
        )

        assert isinstance(sp_resp, ControllableNoInjection)
        assert opt._primary_pre_controllable is None

    @pytest.mark.asyncio
    async def test_pre_call_after_max_turns_returns_no_injection(self) -> None:
        opt = await _init_optimizer(max_turns=1)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl = _user_ctrl()

        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "p1"),
            ),
        ):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )

        resp = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="r2"),
        )
        assert isinstance(resp, ControllableNoInjection)
        assert opt._attempt_done is True


# ---------------------------------------------------------------------------
# Multi-turn conversation behaviour
# ---------------------------------------------------------------------------


class TestMultiTurn:
    @pytest.mark.asyncio
    async def test_multi_turn_within_one_run_feeds_response_to_next_turn(
        self,
    ) -> None:
        opt = await _init_optimizer(max_turns=3)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        ctrl = _user_ctrl()
        post_ctrl = _response_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
                AttackerTurn("o", "t", "s", "p3"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="seed"),
            )
            await _dispatch_event(opt,
                ControllablePostCallEvent(
                    controllable=post_ctrl, request="p1", answer="reply-1",
                ),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="seed"),
            )
            assert (
                attacker_mock.call_args_list[1].kwargs["prev_prompt"] == "p1"
            )
            assert (
                attacker_mock.call_args_list[1].kwargs["prev_response"] == "reply-1"
            )

            await _dispatch_event(opt,
                ControllablePostCallEvent(
                    controllable=post_ctrl, request="p2", answer="reply-2",
                ),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="seed"),
            )

            # Turn 4 hits max_turns.
            resp4 = await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="seed"),
            )
            assert isinstance(resp4, ControllableNoInjection)
            assert opt._attempt_done is True


# ---------------------------------------------------------------------------
# Response recovery
# ---------------------------------------------------------------------------


class TestResponseRecovery:
    @pytest.mark.asyncio
    async def test_trajectory_observable_preferred_over_postcall(self) -> None:
        opt = await _init_optimizer(max_turns=3)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        ctrl = _user_ctrl()
        post_ctrl = _response_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
            await _dispatch_event(opt,
                ControllablePostCallEvent(
                    controllable=post_ctrl, request="p1", answer="from-postcall",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(),
                    content="from-trajectory",
                ),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r2"),
            )

        assert (
            attacker_mock.call_args_list[1].kwargs["prev_response"]
            == "from-trajectory"
        )

    @pytest.mark.asyncio
    async def test_postcall_used_when_no_trajectory_observable(self) -> None:
        opt = await _init_optimizer(max_turns=3)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        ctrl = _user_ctrl()
        post_ctrl = _response_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
            await _dispatch_event(opt,
                ControllablePostCallEvent(
                    controllable=post_ctrl, request="p1", answer="postcall-reply",
                ),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r2"),
            )

        assert (
            attacker_mock.call_args_list[1].kwargs["prev_response"]
            == "postcall-reply"
        )

    @pytest.mark.asyncio
    async def test_response_observable_names_are_case_normalized(self) -> None:
        opt = GOATOptimizer(
            max_turns=2,
            max_attempts=1,
            response_observable_names=["Assistant_Response"],
        )
        await opt.initialize(
            goal=Goal(description="test goal"),
            controllables=[_user_ctrl()],
            observables=[],
            llm_client=_empty_llm(),
        )
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
            ],
        )
        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(
                opt,
                ControllablePreCallEvent(controllable=_user_ctrl(), request="r1"),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable("assistant_response"),
                    content="case-normalized-response",
                ),
            )
            await _dispatch_event(
                opt,
                ControllablePreCallEvent(controllable=_user_ctrl(), request="r2"),
            )

        assert (
            attacker_mock.call_args_list[1].kwargs["prev_response"]
            == "case-normalized-response"
        )

    @pytest.mark.asyncio
    async def test_empty_M_fallback_when_neither_visible(self) -> None:
        """Paper "if M is empty" branch when nothing about the response is in scope."""
        opt = await _init_optimizer(max_turns=3)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r2"),
            )

        prev_resp = attacker_mock.call_args_list[1].kwargs["prev_response"]
        assert not prev_resp


# ---------------------------------------------------------------------------
# Run end + cross-attempt
# ---------------------------------------------------------------------------


class TestRunEnd:
    @pytest.mark.asyncio
    async def test_success_eval_signals_done(self) -> None:
        opt = await _init_optimizer(max_attempts=10)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        end_resp = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )

        assert isinstance(end_resp, RunEndResponse)
        assert end_resp.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_no_evaluation_continues_until_max_attempts(self) -> None:
        opt = await _init_optimizer(max_attempts=3)

        for expected_done in (False, False, True):
            await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
            end = await _dispatch_event(opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            assert end.done is expected_done

    @pytest.mark.asyncio
    async def test_failure_eval_does_not_signal_done(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_failure_eval(), security_domain=USER_TAG),
        )
        assert end.done is False

    @pytest.mark.asyncio
    async def test_fresh_attacker_per_attempt(self) -> None:
        opt = await _init_optimizer(max_attempts=2)

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker_a = opt._attacker
        await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker_b = opt._attacker

        assert attacker_a is not attacker_b


# ---------------------------------------------------------------------------
# Adversarial information-access settings
# ---------------------------------------------------------------------------


class TestAdversarialSettings:
    """The four settings the project requires.

    The optimizer naturally operates in all four — there is no
    setting knob. The framework's scope filter and ``include_feedback``
    flag select which information surfaces are visible.
    """

    @pytest.mark.asyncio
    async def test_setting_1_user_query_only(self) -> None:
        """No responses, no feedback. K turns still fire; every M is empty."""
        opt = await _init_optimizer(max_turns=3, max_attempts=1)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", f"p{i}") for i in range(3)
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            for i in range(3):
                resp = await _dispatch_event(opt,
                    ControllablePreCallEvent(controllable=ctrl, request=f"r{i}"),
                )
                assert isinstance(resp, ControllableInjection)

        assert attacker_mock.call_count == 3
        for call in attacker_mock.call_args_list:
            assert not call.kwargs["prev_response"]

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end.done is True  # max_attempts=1 exhausted

    @pytest.mark.asyncio
    async def test_setting_2_user_query_plus_feedback(self) -> None:
        """Feedback only. Per-turn behaviour matches setting 1; success early-stops."""
        opt = await _init_optimizer(max_turns=3, max_attempts=10)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "p"),
            ),
        ):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True

    @pytest.mark.asyncio
    async def test_setting_3_user_query_plus_responses(self) -> None:
        """Responses, no feedback. Full GOAT loop, no early-stop."""
        opt = await _init_optimizer(max_turns=2, max_attempts=2)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        ctrl = _user_ctrl()
        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "p1"),
                AttackerTurn("o", "t", "s", "p2"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(),
                    content="dangerous reply",
                ),
            )
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r2"),
            )

        assert (
            attacker_mock.call_args_list[1].kwargs["prev_response"]
            == "dangerous reply"
        )

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end.done is False

    @pytest.mark.asyncio
    async def test_setting_4_full_access(self) -> None:
        """Responses + feedback. Full GOAT plus early-stop on success."""
        opt = await _init_optimizer(max_turns=2, max_attempts=10)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        ctrl = _user_ctrl()
        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                return_value=AttackerTurn("o", "t", "s", "p"),
            ),
        ):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(),
                    content="bad reply",
                ),
            )

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True

    @pytest.mark.asyncio
    async def test_max_no_response_runs_terminates_blind_optimization(self) -> None:
        opt = await _init_optimizer(
            max_turns=2,
            max_attempts=10,
            max_no_response_runs=2,
        )

        ctrl = _user_ctrl()
        attacker_mock = AsyncMock(
            return_value=AttackerTurn("o", "t", "s", "p"),
        )

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
        end1 = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end1.done is False

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )
        end2 = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end2.done is True


# ---------------------------------------------------------------------------
# Benign fallback on attacker parse failure
# ---------------------------------------------------------------------------


class TestBenignFallback:
    @pytest.mark.asyncio
    async def test_attacker_parse_failure_uses_benign_fallback(self) -> None:
        opt = await _init_optimizer(max_turns=2, max_attempts=1)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        with patch.object(
            opt._attacker, "next_turn", new=AsyncMock(
                side_effect=AttackerParseError("simulated"),
            ),
        ):
            resp = await _dispatch_event(opt,
                ControllablePreCallEvent(controllable=ctrl, request="r1"),
            )

        assert isinstance(resp, ControllableInjection)
        assert "test goal" not in resp.value.lower()
        assert ("neutral" in resp.value.lower()
                or "high-level" in resp.value.lower())
        assert opt._turn == 1


# ---------------------------------------------------------------------------
# End-to-end target-loop integration
# ---------------------------------------------------------------------------


class TestTargetRunIntegration:
    @pytest.mark.asyncio
    async def test_optimizer_drives_chatbot_style_target_loop(self) -> None:
        """Simulate ChatbotTarget's [system_prompt][user_message_loop] shape."""
        opt = await _init_optimizer(max_turns=2, max_attempts=1)
        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        async def send_event(event):
            return await _dispatch_event(opt, event)

        attacker_mock = AsyncMock(
            side_effect=[
                AttackerTurn("o", "t", "s", "user-turn-1"),
                AttackerTurn("o", "t", "s", "user-turn-2"),
            ],
        )

        with patch.object(opt._attacker, "next_turn", new=attacker_mock):
            sp_resp = await send_event(
                ControllablePreCallEvent(
                    controllable=_system_prompt_ctrl(),
                    request="default-system-prompt",
                ),
            )
            assert isinstance(sp_resp, ControllableNoInjection)

            user_ctrl = _user_ctrl()
            response_ctrl = _response_ctrl()

            pre1 = await send_event(
                ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
            )
            assert isinstance(pre1, ControllableInjection)
            assert pre1.value == "user-turn-1"

            await send_event(
                ControllablePostCallEvent(
                    controllable=response_ctrl,
                    request="user-turn-1",
                    answer="assistant said something benign",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(),
                    content="assistant said something benign",
                ),
            )

            pre2 = await send_event(
                ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
            )
            assert isinstance(pre2, ControllableInjection)
            assert pre2.value == "user-turn-2"

            pre3 = await send_event(
                ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
            )
            assert isinstance(pre3, ControllableNoInjection)

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True

        assert (
            attacker_mock.call_args_list[1].kwargs["prev_response"]
            == "assistant said something benign"
        )
