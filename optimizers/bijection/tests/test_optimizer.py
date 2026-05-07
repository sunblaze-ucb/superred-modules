"""Tests for BijectionOptimizer event-driven state machine.

Covers:
* RunStart / PreCall / PostCall / RunEnd dispatch and state.
* Single-turn injection (one PreCall per run, one PostCall pairing).
* Two-channel split when ``system_prompt`` is also in controllable scope.
* ``target_controllable_name`` override locks injection.
* Trajectory-first response observation; PostCall fallback.
* Cross-attempt: fresh bijection per run, deterministic via seed.
* Early-stop on ``RunEndEvent.evaluation.success``.
* All four adversarial information-access settings.
* ``max_no_signal_runs`` blind-loop guard.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

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
from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from bijection_optimizer.optimizer import BijectionOptimizer
from bijection_optimizer.prompts import MULTITURN_INDICATOR

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
    """Stand-in LLMClient. Bijection never calls the LLM.

    The optimizer's design is response-blind on the construction path,
    so the LLM client is provided only to satisfy the abstract
    ``Optimizer.initialize`` signature.
    """
    llm = MagicMock()

    async def fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("Bijection must not call the LLM client")

    llm.complete.side_effect = fail
    return llm


async def _init_optimizer(
    *,
    controllables: list[Controllable] | None = None,
    bijection_type: str = "digit",
    fixed_size: int = 10,
    num_digits: int = 2,
    digit_delimiter: str = "  ",
    num_teaching_shots: int = 3,
    max_attempts: int = 1,
    target_controllable_name: str | None = None,
    max_no_signal_runs: int = 0,
    seed: int = 0,
) -> BijectionOptimizer:
    opt = BijectionOptimizer(
        bijection_type=bijection_type,
        fixed_size=fixed_size,
        num_digits=num_digits,
        digit_delimiter=digit_delimiter,
        num_teaching_shots=num_teaching_shots,
        max_attempts=max_attempts,
        target_controllable_name=target_controllable_name,
        max_no_signal_runs=max_no_signal_runs,
        seed=seed,
    )
    await opt.initialize(
        goal=Goal(description="describe a recipe"),
        controllables=controllables if controllables is not None else [_user_ctrl()],
        observables=[],
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


class TestConstruction:
    def test_rejects_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            BijectionOptimizer(max_attempts=0)

    def test_rejects_negative_teaching_shots(self) -> None:
        with pytest.raises(ValueError):
            BijectionOptimizer(num_teaching_shots=-1)

    def test_rejects_unknown_bijection_type(self) -> None:
        with pytest.raises(ValueError):
            BijectionOptimizer(bijection_type="word")

    @pytest.mark.parametrize("fixed_size", [-1, 27])
    def test_rejects_out_of_range_fixed_size(self, fixed_size: int) -> None:
        with pytest.raises(ValueError):
            BijectionOptimizer(fixed_size=fixed_size)


# ---------------------------------------------------------------------------
# Run start
# ---------------------------------------------------------------------------


class TestRunStart:
    @pytest.mark.asyncio
    async def test_run_start_prepares_bijection_and_prompt(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._current_bijection is not None
        assert opt._current_user_message
        # Default scope (user_message only) keeps the intro inside the
        # user-message; system_prompt slot is empty.
        assert opt._current_system_prompt == ""

    @pytest.mark.asyncio
    async def test_run_start_resets_per_run_state(self) -> None:
        opt = await _init_optimizer()
        opt._injected_this_run = True
        opt._awaiting_post_call = True
        opt._pending_post_answer = "stale"
        opt._primary_pre_controllable = _user_ctrl()

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        assert opt._injected_this_run is False
        assert opt._awaiting_post_call is False
        assert opt._pending_post_answer is None
        assert opt._primary_pre_controllable is None


# ---------------------------------------------------------------------------
# Pre-call: injection + skip + lock
# ---------------------------------------------------------------------------


class TestPreCall:
    @pytest.mark.asyncio
    async def test_pre_call_injects_bijection_prompt(self) -> None:
        opt = await _init_optimizer(num_teaching_shots=2)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        resp = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
        )
        assert isinstance(resp, ControllableInjection)
        # Plain English goal must NOT appear; encoded text must.
        assert "describe a recipe" not in resp.value
        assert MULTITURN_INDICATOR in resp.value
        assert "Language Alpha" in resp.value
        assert opt._injected_this_run is True

    @pytest.mark.asyncio
    async def test_second_pre_call_in_same_run_returns_no_injection(self) -> None:
        """Single-turn: only the first user-message PreCall is injected."""
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl = _user_ctrl()
        first = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r1"),
        )
        assert isinstance(first, ControllableInjection)

        second = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r2"),
        )
        assert isinstance(second, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_locks_to_first_user_controllable(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl_user = _user_ctrl()
        ctrl_other = Controllable(
            name="other_user_channel", security_domain=USER_TAG,
        )

        first = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl_user, request="r1"),
        )
        assert isinstance(first, ControllableInjection)

        second = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl_other, request="r2"),
        )
        assert isinstance(second, ControllableNoInjection)
        assert opt._primary_pre_controllable == ctrl_user

    @pytest.mark.asyncio
    async def test_system_prompt_pre_call_when_not_in_scope_is_skipped(
        self,
    ) -> None:
        """ChatbotTarget shape: system_prompt PreCall before user_message loop."""
        opt = await _init_optimizer()  # default: user_message only in scope
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="seed",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)
        # Lock has not been claimed by system_prompt — user_message can still
        # become the primary.
        assert opt._primary_pre_controllable is None

        user_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="seed",
            ),
        )
        assert isinstance(user_resp, ControllableInjection)


# ---------------------------------------------------------------------------
# Two-channel split (system_prompt in scope)
# ---------------------------------------------------------------------------


class TestTwoChannelSplit:
    @pytest.mark.asyncio
    async def test_intro_goes_to_system_prompt_when_in_scope(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="default-sp",
            ),
        )
        assert isinstance(sp_resp, ControllableInjection)
        assert "Language Alpha" in sp_resp.value
        assert "alphabet map" in sp_resp.value

        user_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="seed",
            ),
        )
        assert isinstance(user_resp, ControllableInjection)
        # When intro goes to system_prompt, user_message must NOT include
        # the intro again — only teaching shots + encoded query.
        assert "You are about to learn a new language" not in user_resp.value
        assert MULTITURN_INDICATOR in user_resp.value
        assert "User: Here is a text in English" in user_resp.value


# ---------------------------------------------------------------------------
# target_controllable_name override
# ---------------------------------------------------------------------------


class TestTargetControllableNameOverride:
    @pytest.mark.asyncio
    async def test_override_locks_to_named_controllable(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
            target_controllable_name="prompt_text",
        )
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        wrong_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="x",
            ),
        )
        assert isinstance(wrong_resp, ControllableNoInjection)

        target = Controllable(name="prompt_text", security_domain=USER_TAG)
        right_resp = await opt.on_event(
            ControllablePreCallEvent(controllable=target, request="x"),
        )
        assert isinstance(right_resp, ControllableInjection)
        # Override forces single-channel mode: full prompt incl. intro
        # goes into ``prompt_text``; system_prompt is not used.
        assert "Language Alpha" in right_resp.value


# ---------------------------------------------------------------------------
# Post-call: pairing
# ---------------------------------------------------------------------------


class TestPostCall:
    @pytest.mark.asyncio
    async def test_post_call_records_answer(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="seed"),
        )
        injected_value = pre.value  # type: ignore[attr-defined]

        post_ctrl = _response_ctrl()
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=post_ctrl,
                request=injected_value,
                answer="dangerous reply",
            ),
        )
        assert opt._pending_post_answer == "dangerous reply"

    @pytest.mark.asyncio
    async def test_post_call_unrelated_controllable_ignored(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="seed",
            ),
        )

        # Distinct controllable + request that doesn't match the pre's
        # ``seed`` and isn't the injected value -> all 3 pairing rules
        # fail and the answer must be ignored.
        unrelated = Controllable(name="other", security_domain=USER_TAG)
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=unrelated,
                request="totally-unrelated",
                answer="ignore me",
            ),
        )
        assert opt._pending_post_answer is None


# ---------------------------------------------------------------------------
# Run end + cross-attempt
# ---------------------------------------------------------------------------


class TestRunEnd:
    @pytest.mark.asyncio
    async def test_success_eval_signals_done(self) -> None:
        opt = await _init_optimizer(max_attempts=10)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        end_resp = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert isinstance(end_resp, RunEndResponse)
        assert end_resp.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_failure_eval_does_not_signal_done(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end = await opt.on_event(
            RunEndEvent(evaluation=_failure_eval(), security_domain=USER_TAG),
        )
        assert end.done is False

    @pytest.mark.asyncio
    async def test_no_evaluation_continues_until_max_attempts(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        for expected_done in (False, False, True):
            await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
            end = await opt.on_event(
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            assert end.done is expected_done

    @pytest.mark.asyncio
    async def test_fresh_bijection_per_attempt(self) -> None:
        opt = await _init_optimizer(max_attempts=2, seed=2024)

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        bj_a = opt._current_bijection
        assert bj_a is not None
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        bj_b = opt._current_bijection
        assert bj_b is not None
        # Same seed feeds an RNG that advances between draws → different
        # mappings (high probability for digit codomain).
        assert bj_a.mapping != bj_b.mapping

    @pytest.mark.asyncio
    async def test_deterministic_with_same_seed(self) -> None:
        opt_a = await _init_optimizer(seed=7)
        opt_b = await _init_optimizer(seed=7)
        await opt_a.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt_b.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt_a._current_bijection is not None
        assert opt_b._current_bijection is not None
        assert opt_a._current_bijection.mapping == opt_b._current_bijection.mapping


# ---------------------------------------------------------------------------
# Adversarial information-access settings
# ---------------------------------------------------------------------------


class TestAdversarialSettings:
    """Four settings the project requires.

    The optimizer naturally operates in all four — there is no
    setting knob. Bijection is response-blind on the construction
    path; the framework's scope filter and ``include_feedback`` flag
    decide which information is visible.
    """

    @pytest.mark.asyncio
    async def test_setting_1_user_query_only(self) -> None:
        """No responses, no feedback. Each run injects, no early-stop."""
        opt = await _init_optimizer(max_attempts=2)
        for run in range(2):
            await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
            resp = await opt.on_event(
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{run}",
                ),
            )
            assert isinstance(resp, ControllableInjection)
            end = await opt.on_event(
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            assert end.done is (run == 1)  # only the last attempt is "done"

    @pytest.mark.asyncio
    async def test_setting_2_user_query_plus_feedback_early_stops(self) -> None:
        opt = await _init_optimizer(max_attempts=10)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_setting_3_user_query_plus_responses_no_eval(self) -> None:
        """Responses present on trajectory, no eval. No early-stop."""
        opt = await _init_optimizer(max_attempts=2)
        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="encoded reply",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end.done is False

    @pytest.mark.asyncio
    async def test_setting_4_full_access(self) -> None:
        opt = await _init_optimizer(max_attempts=10)
        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="encoded reply",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True


# ---------------------------------------------------------------------------
# Blind-loop guard
# ---------------------------------------------------------------------------


class TestNoSignalGuard:
    @pytest.mark.asyncio
    async def test_max_no_signal_runs_terminates_blind_optimization(self) -> None:
        opt = await _init_optimizer(
            max_attempts=10,
            max_no_signal_runs=2,
        )

        # Run 1: nothing visible.
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        end1 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end1.done is False

        # Run 2: still nothing visible -> hit the guard.
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        end2 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end2.done is True

    @pytest.mark.asyncio
    async def test_no_signal_counter_resets_when_response_seen(self) -> None:
        opt = await _init_optimizer(
            max_attempts=10,
            max_no_signal_runs=2,
        )

        # Run 1: blind.
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        # Run 2: response present -> counter resets.
        traj2 = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj2))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r2",
            ),
        )
        traj2.push(
            ObservableEvent(
                observable=_response_observable(), content="seen",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        # Run 3: blind again -> counter is back at 1, not 3.
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r3",
            ),
        )
        end3 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end3.done is False


# ---------------------------------------------------------------------------
# End-to-end target-loop integration
# ---------------------------------------------------------------------------


class TestTargetRunIntegration:
    @pytest.mark.asyncio
    async def test_chatbot_style_loop_with_user_only_scope(self) -> None:
        """ChatbotTarget shape: system_prompt PreCall, then user_message PreCall."""
        opt = await _init_optimizer(max_attempts=1, seed=1)
        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))

        # 1. system_prompt PreCall before the loop (out of scope) -> skipped.
        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)

        user_ctrl = _user_ctrl()
        response_ctrl = _response_ctrl()

        # 2. First user_message PreCall -> bijection prompt.
        pre1 = await opt.on_event(
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre1, ControllableInjection)
        bijection_prompt = pre1.value
        assert MULTITURN_INDICATOR in bijection_prompt
        assert "Language Alpha" in bijection_prompt

        # 3. PostCall delivers (encoded) reply.
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=response_ctrl,
                request=bijection_prompt,
                answer="encoded assistant reply",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="encoded assistant reply",
            ),
        )

        # 4. Target tries another user_message PreCall -> NoInjection (single-turn).
        pre2 = await opt.on_event(
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre2, ControllableNoInjection)

        # 5. RunEnd with success.
        end = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_chatbot_style_loop_with_system_prompt_in_scope(self) -> None:
        """When system_prompt is in scope, intro is split into the system channel."""
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
            max_attempts=1,
            seed=2,
        )
        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableInjection)
        assert "Language Alpha" in sp_resp.value
        # Intro carries the alphabet map; user-message must not duplicate it.

        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
        )
        assert isinstance(pre, ControllableInjection)
        assert "You are about to learn a new language" not in pre.value
        assert MULTITURN_INDICATOR in pre.value

        end = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True

    @pytest.mark.asyncio
    async def test_drains_response_through_real_filtered_trajectory(self) -> None:
        """Wire up an actual ``Trajectory(filtered_scope=...).filtered``
        and push a real ``ObservableEvent`` through it.

        Other tests use ``_FakeReadableTrajectory`` for focus; this one
        guards against drift in real ``Trajectory`` / ``FilteredTrajectory``
        ``emit`` / ``drain`` semantics.
        """
        trajectory = Trajectory(filtered_scope=frozenset({RESPONSE_TAG}))
        opt = await _init_optimizer(max_attempts=2, max_no_signal_runs=0, seed=7)

        await opt.on_event(RunStartEvent(trajectory=trajectory.filtered))

        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )

        trajectory.emit(
            ObservableEvent(
                observable=_response_observable(),
                content="real-filtered response",
            ),
        )

        end1 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end1.done is False
        assert opt._consecutive_no_signal_runs == 0
        # The drain consumed the response; nothing left over.
        assert trajectory.filtered.drain() == []
