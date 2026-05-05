"""Tests for the GEPAOptimizer event-driven state machine.

Covers:

* RunStart / PreCall / PostCall / RunEnd dispatch and per-run state.
* Seed-candidate first run; freshly-proposed candidate next run.
* ``system_prompt`` PreCall passes through without locking.
* Trajectory-first response recovery; PostCall fallback; empty M.
* Pool growth: only freshly-proposed candidates enter the pool.
* Best-scoring parent selection with latest-wins tie-breaking.
* Early-stop on ``RunEndEvent.evaluation.success``.
* ``max_attempts`` exhaustion.
* All four adversarial information-access settings.
* Reflection LM parse failure leaves the next run with no pending
  proposal (parent re-rolls instead of crashing).
* End-to-end ChatbotTarget-style integration shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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

from gepa_optimizer.optimizer import GEPAOptimizer
from gepa_optimizer.reflector import ReflectionResult


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
    """LLMClient stub that errors if .complete is called.

    All optimizer tests patch ``opt._reflector.propose`` directly, so
    the underlying LLM should never be invoked.
    """
    llm = MagicMock()

    async def fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(
            "Underlying LLM should not be called; tests patch reflector.propose"
        )

    llm.complete.side_effect = fail
    return llm


async def _init_optimizer(
    *,
    max_attempts: int = 5,
    max_no_signal_runs: int = 0,
) -> GEPAOptimizer:
    opt = GEPAOptimizer(
        max_attempts=max_attempts,
        max_no_signal_runs=max_no_signal_runs,
    )
    await opt.initialize(
        goal=Goal(description="achieve target X"),
        controllables=[_user_ctrl()],
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


def _failure_eval(score: float = 0.1, rationale: str = "judge says no") -> EvaluationResult:
    return EvaluationResult(
        success=False,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale=rationale,
    )


def _refl(text: str) -> ReflectionResult:
    return ReflectionResult(new_instruction=text, raw_output=text, prompt="")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            GEPAOptimizer(max_attempts=0)

    def test_clamps_max_no_signal_runs_to_non_negative(self) -> None:
        opt = GEPAOptimizer(max_no_signal_runs=-3)
        assert opt._max_no_signal_runs == 0


# ---------------------------------------------------------------------------
# Initialization seeds the pool
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_seeds_pool_with_goal_description(self) -> None:
        opt = await _init_optimizer()
        assert len(opt._pool) == 1
        assert opt._pool[0].prompt == "achieve target X"
        assert opt._pool[0].rolled_out is False

    @pytest.mark.asyncio
    async def test_constructs_reflector(self) -> None:
        opt = await _init_optimizer()
        assert opt._reflector is not None


# ---------------------------------------------------------------------------
# RunStart selects current candidate
# ---------------------------------------------------------------------------


class TestRunStart:
    @pytest.mark.asyncio
    async def test_first_run_picks_seed(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._current is opt._pool[0]
        assert opt._current_is_fresh is False

    @pytest.mark.asyncio
    async def test_pending_proposal_wins_when_present(self) -> None:
        opt = await _init_optimizer()
        # Manually plant a pending mutation to verify selection rule.
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        # Roll out the seed first.
        await _roll_out_one(opt, eval_=_failure_eval(0.2))

        # After RunEnd, reflection should have set up a pending proposal,
        # but tests patch reflection — emulate by setting it directly.
        from gepa_optimizer.optimizer import _Candidate

        opt._pending = _Candidate(prompt="proposed", parent_idx=0)

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._current is not None
        assert opt._current.prompt == "proposed"
        assert opt._current_is_fresh is True
        assert opt._pending is None


# ---------------------------------------------------------------------------
# PreCall: injection + skip + lock
# ---------------------------------------------------------------------------


class TestPreCall:
    @pytest.mark.asyncio
    async def test_pre_call_injects_current_candidate_prompt(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
        )
        assert isinstance(resp, ControllableInjection)
        assert resp.value == "achieve target X"

    @pytest.mark.asyncio
    async def test_only_first_pre_call_per_run_gets_injected(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl = _user_ctrl()

        first = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r1"),
        )
        second = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r2"),
        )

        assert isinstance(first, ControllableInjection)
        assert isinstance(second, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_locks_to_first_user_controllable(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl_a = _user_ctrl("user_message")
        ctrl_b = Controllable(name="another_user_channel", security_domain=USER_TAG)

        first = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl_a, request="r1"),
        )
        second = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl_b, request="r2"),
        )
        assert isinstance(first, ControllableInjection)
        assert isinstance(second, ControllableNoInjection)
        assert opt._primary_pre_controllable is ctrl_a

    @pytest.mark.asyncio
    async def test_system_prompt_pre_call_passes_through_without_locking(
        self,
    ) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="seed",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)
        assert opt._primary_pre_controllable is None

        user_resp = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
        )
        assert isinstance(user_resp, ControllableInjection)
        assert user_resp.value == "achieve target X"


# ---------------------------------------------------------------------------
# PostCall: 3-way pairing
# ---------------------------------------------------------------------------


class TestPostCall:
    @pytest.mark.asyncio
    async def test_post_call_pairs_by_controllable_identity(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl = _user_ctrl()

        await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r1"),
        )
        resp = await opt.on_event(
            ControllablePostCallEvent(
                controllable=ctrl, request="r1", answer="hello",
            ),
        )

        assert isinstance(resp, ControllableNoInjection)
        assert opt._pending_post_answer == "hello"
        assert opt._primary_post_controllable is ctrl

    @pytest.mark.asyncio
    async def test_post_call_pairs_by_request_match(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="r1"),
        )
        # PostCall on a different controllable but with matching pre-request.
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(), request="r1", answer="hi",
            ),
        )
        assert opt._pending_post_answer == "hi"

    @pytest.mark.asyncio
    async def test_post_call_pairs_by_injected_value_match(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="seed"),
        )
        # PostCall whose request equals the injected value (request after rewrite).
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request="achieve target X",
                answer="hi",
            ),
        )
        assert opt._pending_post_answer == "hi"

    @pytest.mark.asyncio
    async def test_post_call_without_match_is_ignored(self) -> None:
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="r1"),
        )
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request="something_else",
                answer="hi",
            ),
        )
        assert opt._pending_post_answer is None


# ---------------------------------------------------------------------------
# RunEnd: pool growth, scoring, reflection wiring
# ---------------------------------------------------------------------------


async def _roll_out_one(
    opt: GEPAOptimizer,
    *,
    eval_: EvaluationResult | None = None,
    response_text: str | None = None,
    user_ctrl: Controllable | None = None,
    response_ctrl: Controllable | None = None,
) -> RunEndResponse:
    """Drive opt through one full run with the current/pending candidate.

    Always sends a fresh ``RunStartEvent`` so per-run state is reset
    and any pending mutation is consumed by the start of this run.
    """
    if user_ctrl is None:
        user_ctrl = _user_ctrl()
    if response_ctrl is None:
        response_ctrl = _response_ctrl()

    traj = _FakeReadableTrajectory()
    await opt.on_event(RunStartEvent(trajectory=traj))

    await opt.on_event(
        ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
    )

    if response_text is not None:
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content=response_text,
            ),
        )

    end = await opt.on_event(
        RunEndEvent(evaluation=eval_, security_domain=USER_TAG),
    )
    assert isinstance(end, RunEndResponse)
    return end


class TestRunEndPoolGrowth:
    @pytest.mark.asyncio
    async def test_seed_first_rollout_does_not_duplicate_in_pool(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.3), response_text="r")
        assert len(opt._pool) == 1
        assert opt._pool[0].rolled_out is True
        assert opt._pool[0].score == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_freshly_proposed_candidate_enters_pool(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        # Seed run + reflection -> sets pending = "M1"
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.3))
        assert opt._pending is not None
        # Run 2: pending becomes current, gets evaluated, enters pool.
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M2")),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.5))
        assert len(opt._pool) == 2
        prompts = [c.prompt for c in opt._pool]
        assert "achieve target X" in prompts
        assert "M1" in prompts

    @pytest.mark.asyncio
    async def test_best_scoring_parent_is_used_for_reflection(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        # Run 1: seed scores 0.7
        propose = AsyncMock(side_effect=[_refl("M1"), _refl("M2")])
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.7))
            # Run 2: M1 scores 0.3 (worse than seed)
            await _roll_out_one(opt, eval_=_failure_eval(0.3))

        # Second reflection call should have used the seed (0.7) as parent.
        second_call_kwargs = propose.call_args_list[1].kwargs
        assert second_call_kwargs["current_instruction"] == "achieve target X"


class TestRunEndDoneSemantics:
    @pytest.mark.asyncio
    async def test_success_evaluation_signals_done(self) -> None:
        opt = await _init_optimizer(max_attempts=10)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            end = await _roll_out_one(opt, eval_=_success_eval(0.95))
        assert end.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_failure_evaluation_continues(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            end = await _roll_out_one(opt, eval_=_failure_eval(0.4))
        assert end.done is False

    @pytest.mark.asyncio
    async def test_max_attempts_terminates(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            end_1 = await _roll_out_one(opt, eval_=_failure_eval(0.3))
            assert end_1.done is False
            end_2 = await _roll_out_one(opt, eval_=_failure_eval(0.4))
        assert end_2.done is True

    @pytest.mark.asyncio
    async def test_max_no_signal_runs_terminates_when_blind(self) -> None:
        opt = await _init_optimizer(max_attempts=10, max_no_signal_runs=2)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            end_1 = await _roll_out_one(opt, eval_=None, response_text=None)
            assert end_1.done is False
            end_2 = await _roll_out_one(opt, eval_=None, response_text=None)
        assert end_2.done is True


# ---------------------------------------------------------------------------
# Adversarial information-access settings
# ---------------------------------------------------------------------------


class TestAdversarialSettings:
    """All four settings are driven by what's in scope, no setting knob."""

    @pytest.mark.asyncio
    async def test_setting_1_user_query_only(self) -> None:
        """No responses, no feedback. Seed scores 0; reflection still runs."""
        opt = await _init_optimizer(max_attempts=2)
        propose = AsyncMock(return_value=_refl("M1"))
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=None, response_text=None)
        assert opt._pool[0].score is None
        assert opt._pool[0].response is None
        assert propose.called
        # The reflective rollout passed to propose has no response/score.
        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].response is None
        assert rollouts[0].score is None

    @pytest.mark.asyncio
    async def test_setting_2_user_query_plus_feedback(self) -> None:
        """Score drives selection; success early-stops."""
        opt = await _init_optimizer(max_attempts=10)
        propose = AsyncMock(return_value=_refl("M1"))
        with patch.object(opt._reflector, "propose", new=propose):
            end = await _roll_out_one(
                opt, eval_=_success_eval(0.9), response_text=None
            )
        assert end.done is True
        assert opt._pool[0].score == pytest.approx(0.9)
        assert opt._pool[0].response is None

    @pytest.mark.asyncio
    async def test_setting_3_user_query_plus_responses(self) -> None:
        """Response feeds reflection; no early-stop without feedback."""
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=_refl("M1"))
        with patch.object(opt._reflector, "propose", new=propose):
            end = await _roll_out_one(
                opt, eval_=None, response_text="dangerous reply"
            )
        assert end.done is False
        assert opt._pool[0].response == "dangerous reply"
        assert opt._pool[0].score is None

        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].response == "dangerous reply"
        assert rollouts[0].score is None

    @pytest.mark.asyncio
    async def test_setting_4_full_access(self) -> None:
        """Full GEPA loop with response, score, rationale, and early-stop."""
        opt = await _init_optimizer(max_attempts=10)
        propose = AsyncMock(return_value=_refl("M1"))
        with patch.object(opt._reflector, "propose", new=propose):
            end = await _roll_out_one(
                opt, eval_=_success_eval(0.92), response_text="r"
            )
        assert end.done is True
        assert opt._pool[0].response == "r"
        assert opt._pool[0].score == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Reflection failure handling
# ---------------------------------------------------------------------------


class TestReflectionFailure:
    @pytest.mark.asyncio
    async def test_propose_returning_none_leaves_no_pending(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.3))
        assert opt._pending is None

    @pytest.mark.asyncio
    async def test_reflection_exception_does_not_crash_run_end(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        with patch.object(
            opt._reflector, "propose",
            new=AsyncMock(side_effect=RuntimeError("LM blew up")),
        ):
            end = await _roll_out_one(opt, eval_=_failure_eval(0.3))
        assert isinstance(end, RunEndResponse)
        assert end.done is False
        assert opt._pending is None

    @pytest.mark.asyncio
    async def test_next_run_re_rolls_seed_when_no_pending(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.3))
            await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._current is opt._pool[0]
        assert opt._current_is_fresh is False


# ---------------------------------------------------------------------------
# End-to-end ChatbotTarget-style integration
# ---------------------------------------------------------------------------


class TestTargetRunIntegration:
    @pytest.mark.asyncio
    async def test_drives_chatbot_style_target_loop(self) -> None:
        """ChatbotTarget shape: [system_prompt PreCall][user_message loop]."""
        opt = await _init_optimizer(max_attempts=2)
        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)

        user_ctrl = _user_ctrl()
        response_ctrl = _response_ctrl()

        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre, ControllableInjection)
        assert pre.value == "achieve target X"

        await opt.on_event(
            ControllablePostCallEvent(
                controllable=response_ctrl,
                request="achieve target X",
                answer="assistant reply",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="assistant reply",
            ),
        )

        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=_refl("M1")),
        ):
            end = await opt.on_event(
                RunEndEvent(evaluation=_success_eval(0.95), security_domain=USER_TAG),
            )
        assert end.done is True
        assert opt._pool[0].response == "assistant reply"
        assert opt._pool[0].score == pytest.approx(0.95)
