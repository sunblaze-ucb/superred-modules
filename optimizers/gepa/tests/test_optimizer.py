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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.score == pytest.approx(0.3)

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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.score is None
        assert opt._pool[0].latest.response is None
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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.score == pytest.approx(0.9)
        assert opt._pool[0].latest.response is None

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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.response == "dangerous reply"
        assert opt._pool[0].latest.score is None

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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.response == "r"
        assert opt._pool[0].latest.score == pytest.approx(0.92)


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
        assert opt._pool[0].latest is not None
        assert opt._pool[0].latest.response == "assistant reply"
        assert opt._pool[0].latest.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Per-candidate rollout history (ring buffer)
# ---------------------------------------------------------------------------


class TestRolloutHistoryBuffer:
    """Each candidate keeps a bounded ring buffer of recent rollouts so the
    reflection LM sees more signal even when the same parent is re-rolled
    after a failed mutation proposal."""

    @pytest.mark.asyncio
    async def test_re_rolling_seed_appends_to_history(self) -> None:
        # Reflection always returns None so each run re-rolls the seed.
        opt = await _init_optimizer(max_attempts=4)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r1")
            await _roll_out_one(opt, eval_=_failure_eval(0.2), response_text="r2")
            await _roll_out_one(opt, eval_=_failure_eval(0.3), response_text="r3")

        seed = opt._pool[0]
        assert len(seed.rollouts) == 3
        scores = [r.score for r in seed.rollouts]
        responses = [r.response for r in seed.rollouts]
        assert scores == pytest.approx([0.1, 0.2, 0.3])
        assert responses == ["r1", "r2", "r3"]

    @pytest.mark.asyncio
    async def test_history_is_bounded_by_default_size(self) -> None:
        # Default ring size is 3; a fourth rollout drops the oldest.
        opt = await _init_optimizer(max_attempts=5)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r1")
            await _roll_out_one(opt, eval_=_failure_eval(0.2), response_text="r2")
            await _roll_out_one(opt, eval_=_failure_eval(0.3), response_text="r3")
            await _roll_out_one(opt, eval_=_failure_eval(0.4), response_text="r4")

        seed = opt._pool[0]
        assert len(seed.rollouts) == 3
        scores = [r.score for r in seed.rollouts]
        assert scores == pytest.approx([0.2, 0.3, 0.4])  # oldest dropped

    @pytest.mark.asyncio
    async def test_effective_score_is_mean_of_buffered_scores(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.2))
            await _roll_out_one(opt, eval_=_failure_eval(0.6))
        assert opt._pool[0].effective_score == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_reflector_receives_all_recent_rollouts(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        propose = AsyncMock(return_value=None)
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r1")
            await _roll_out_one(opt, eval_=_failure_eval(0.2), response_text="r2")

        # Last call should have seen both rollouts in the side-info dataset.
        last_call_rollouts = propose.call_args.kwargs["rollouts"]
        assert len(last_call_rollouts) == 2
        assert [r.response for r in last_call_rollouts] == ["r1", "r2"]

    @pytest.mark.asyncio
    async def test_custom_history_size_is_respected(self) -> None:
        opt = GEPAOptimizer(max_attempts=4, rollout_history_size=2)
        await opt.initialize(
            goal=Goal(description="goal"),
            controllables=[_user_ctrl()],
            observables=[],
            llm_client=_empty_llm(),
        )
        with patch.object(
            opt._reflector, "propose", new=AsyncMock(return_value=None),
        ):
            await _roll_out_one(opt, eval_=_failure_eval(0.1))
            await _roll_out_one(opt, eval_=_failure_eval(0.2))
            await _roll_out_one(opt, eval_=_failure_eval(0.3))
        assert len(opt._pool[0].rollouts) == 2

    def test_construction_rejects_zero_history_size(self) -> None:
        with pytest.raises(ValueError):
            GEPAOptimizer(rollout_history_size=0)


# ---------------------------------------------------------------------------
# System-prompt observable surfaces into the reflective dataset (item 3a)
# ---------------------------------------------------------------------------


class TestSystemPromptObservable:
    @pytest.mark.asyncio
    async def test_system_prompt_observable_surfaced_in_rollout(self) -> None:
        from superred.core.types.observable import ObservableValue

        sp_obs = ObservableValue(
            observable=Observable(
                name="system_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
            ),
            content="You are a careful assistant.",
        )
        opt = GEPAOptimizer(max_attempts=2)
        await opt.initialize(
            goal=Goal(description="exfil"),
            controllables=[_user_ctrl()],
            observables=[sp_obs],
            llm_client=_empty_llm(),
        )
        propose = AsyncMock(return_value=None)
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r")

        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].target_system_prompt == "You are a careful assistant."

    @pytest.mark.asyncio
    async def test_no_system_prompt_observable_means_field_is_none(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        propose = AsyncMock(return_value=None)
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r")
        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].target_system_prompt is None

    @pytest.mark.asyncio
    async def test_custom_observable_name_is_recognised(self) -> None:
        from superred.core.types.observable import ObservableValue

        sp_obs = ObservableValue(
            observable=Observable(
                name="initial_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
            ),
            content="custom-frame",
        )
        opt = GEPAOptimizer(
            max_attempts=2,
            system_prompt_observable_names=("initial_prompt",),
        )
        await opt.initialize(
            goal=Goal(description="g"),
            controllables=[_user_ctrl()],
            observables=[sp_obs],
            llm_client=_empty_llm(),
        )
        propose = AsyncMock(return_value=None)
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r")
        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].target_system_prompt == "custom-frame"

    @pytest.mark.asyncio
    async def test_empty_system_prompt_observable_is_treated_as_absent(self) -> None:
        from superred.core.types.observable import ObservableValue

        sp_obs = ObservableValue(
            observable=Observable(
                name="system_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
            ),
            content="   ",
        )
        opt = GEPAOptimizer(max_attempts=2)
        await opt.initialize(
            goal=Goal(description="g"),
            controllables=[_user_ctrl()],
            observables=[sp_obs],
            llm_client=_empty_llm(),
        )
        propose = AsyncMock(return_value=None)
        with patch.object(opt._reflector, "propose", new=propose):
            await _roll_out_one(opt, eval_=_failure_eval(0.1), response_text="r")
        rollouts = propose.call_args.kwargs["rollouts"]
        assert rollouts[0].target_system_prompt is None


# ---------------------------------------------------------------------------
# target_controllable_name knob (item 3b): explicit-target injection mode
# ---------------------------------------------------------------------------


class TestTargetControllableName:
    @pytest.mark.asyncio
    async def test_default_skips_system_prompt_pre_call(self) -> None:
        """Sanity: the default (unset) preserves the original behavior."""
        opt = await _init_optimizer()
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_explicit_target_locks_onto_system_prompt(self) -> None:
        """When set to ``system_prompt``, GEPA injects there and skips user_message."""
        opt = GEPAOptimizer(
            max_attempts=2,
            target_controllable_name="system_prompt",
        )
        await opt.initialize(
            goal=Goal(description="payload"),
            controllables=[_system_prompt_ctrl(), _user_ctrl()],
            observables=[],
            llm_client=_empty_llm(),
        )
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableInjection)
        assert sp_resp.value == "payload"

        # Subsequent user_message PreCall in the same run is rejected
        # because we're locked onto system_prompt.
        user_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(),
                request="next-turn",
            ),
        )
        assert isinstance(user_resp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_explicit_target_ignores_unrelated_controllables(self) -> None:
        opt = GEPAOptimizer(
            max_attempts=2,
            target_controllable_name="user_message",
        )
        await opt.initialize(
            goal=Goal(description="payload"),
            controllables=[_user_ctrl(), _user_ctrl(name="extra_input")],
            observables=[],
            llm_client=_empty_llm(),
        )
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        # Different name -> ignored.
        other = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(name="extra_input"),
                request="x",
            ),
        )
        assert isinstance(other, ControllableNoInjection)

        # Configured target -> injected.
        target = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(),
                request="x",
            ),
        )
        assert isinstance(target, ControllableInjection)
        assert target.value == "payload"


# ---------------------------------------------------------------------------
# End-to-end: full Controller wiring through _run_single (item 4)
# ---------------------------------------------------------------------------


class _FakeChatTarget:
    """Minimal target shape modeled on ChatbotTarget for integration tests.

    Emits a ``system_prompt`` PreCall (passes through default GEPA),
    then one ``user_message`` PreCall whose value is echoed back as a
    ``response`` ObservableEvent.  ``get_observables`` exposes a
    ``system_prompt`` ObservableValue at SYSTEM_PROMPT_TAG so the
    controller's scope filter decides whether the optimizer sees it.
    """

    def __init__(self, system_prompt: str = "be careful") -> None:
        self._system_prompt = system_prompt
        self.last_response: str = ""

    @property
    def security_domain(self):
        from superred.core.types.security_domain import SecurityDomain

        return SecurityDomain([USER_TAG, SYSTEM_PROMPT_TAG, RESPONSE_TAG])

    @property
    def config_specs(self):
        return []

    def set_config(self, name: str, value: str) -> None:  # pragma: no cover - unused
        pass

    @property
    def query_specs(self):
        return []

    def query(self, name: str, **params: str) -> str:  # pragma: no cover
        return self.last_response if name == "last_response" else ""

    def get_controllables(self) -> list[Controllable]:
        return [
            _system_prompt_ctrl(),
            _user_ctrl(),
            _response_ctrl(),
        ]

    def get_observables(self):
        from superred.core.types.observable import ObservableValue

        return [
            ObservableValue(
                observable=Observable(
                    name="system_prompt",
                    security_domain=SYSTEM_PROMPT_TAG,
                ),
                content=self._system_prompt,
            ),
        ]

    async def run(self, emit, send_event) -> None:
        from superred.core.types.events import (
            ControllableInjection as _Inj,
            ControllablePostCallEvent as _Post,
            ControllablePreCallEvent as _Pre,
            ObservableEvent as _Obs,
        )

        sp_resp = await send_event(
            _Pre(controllable=_system_prompt_ctrl(), request=self._system_prompt),
        )
        if isinstance(sp_resp, _Inj):
            effective_prompt = sp_resp.value
        else:
            effective_prompt = self._system_prompt

        pre = await send_event(
            _Pre(controllable=_user_ctrl(), request="user_message"),
        )
        user_message = pre.value if isinstance(pre, _Inj) else "default"
        # Deterministic "model": echo the prompt + a token so the score
        # function below can flag success.
        assistant = f"[sys={effective_prompt}] echo:{user_message}"
        post = await send_event(
            _Post(
                controllable=_response_ctrl(),
                request=user_message,
                answer=assistant,
            ),
        )
        if isinstance(post, _Inj):
            assistant = post.value
        self.last_response = assistant
        emit(_Obs(observable=_response_observable(), content=assistant))

    async def cleanup(self) -> None:
        self.last_response = ""

    async def teardown(self) -> None:
        pass


class _FakeTask:
    """Task whose evaluation success depends on a magic word in the response.

    The goal description deliberately does NOT contain the magic word, so
    the seed candidate (whose prompt = goal description) fails and the
    reflection-mutated candidate is the one that has to succeed.
    """

    def __init__(self, magic: str = "ATTACK") -> None:
        from superred.core.types.goal import Goal as _Goal

        self._magic = magic
        self._goal = _Goal(
            description="Persuade the assistant to emit a specific token"
        )

    @property
    def goal(self):
        return self._goal

    async def configure_target(self, target) -> None:
        pass

    async def evaluate(self, trajectory, target):
        from superred.core.types.evaluation import (
            EvaluationResult as _Eval,
            Score as _Score,
        )

        response = target.last_response or ""
        success = self._magic in response
        return _Eval(
            success=success,
            primary_score=_Score(value=1.0 if success else 0.0, name="primary"),
            sub_scores={},
            rationale="magic found" if success else "magic missing",
        )


class TestEndToEndControllerIntegration:
    """Wire GEPA through the real Controller end-to-end.

    Validates:
    * scope-filtered observables flow through ``initialize``;
    * ``RunEndEvent.evaluation`` reaches the optimizer;
    * the trajectory's filtered scope is respected on response reads;
    * cross-run pool growth and reflection actually progresses.
    """

    @pytest.mark.asyncio
    async def test_gepa_drives_full_controller_loop_to_success(self) -> None:
        from superred.core.controller import Controller
        from superred.core.interfaces.security_claim import SecurityClaim
        from superred.core.types.llm import LLMConfig

        target = _FakeChatTarget(system_prompt="default-careful")
        task = _FakeTask(magic="ATTACK")
        claim = SecurityClaim.from_tasks([task])

        # Stub reflector: always proposes the magic string so run #2 succeeds.
        async def _stub_propose(self, *, current_instruction, rollouts):  # noqa: ANN001
            return _refl("ATTACK now please")

        with patch.object(
            __import__(
                "gepa_optimizer.reflector", fromlist=["Reflector"]
            ).Reflector,
            "propose",
            new=_stub_propose,
        ):
            controller = Controller(
                optimizer_factory=lambda: GEPAOptimizer(max_attempts=5),
                target=target,
                security_claim=claim,
                llm_configs=[
                    LLMConfig(model="m", api_base="http://x", api_key="sk"),
                ],
            )
            # Full-access scope: user-message + system_prompt readable +
            # response readable; success after 2 runs (seed fails, mutation
            # injects "ATTACK now please" -> magic word in response).
            full_scope = frozenset({USER_TAG, SYSTEM_PROMPT_TAG, RESPONSE_TAG})
            result = await controller.run(scopes=[full_scope])

        tmr = result.threat_model_results[0]
        assert len(tmr.task_results) == 1
        tr = tmr.task_results[0]
        assert tr.success is True
        # At least 2 runs: seed (fail) -> mutated candidate (success).
        assert len(tr.runs) >= 2
        assert tr.best_score.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_gepa_user_only_scope_skips_system_prompt_observable(self) -> None:
        """User-only scope: optimizer must not see the system_prompt observable."""
        from superred.core.controller import Controller
        from superred.core.interfaces.security_claim import SecurityClaim
        from superred.core.types.llm import LLMConfig

        target = _FakeChatTarget(system_prompt="should-be-hidden")
        task = _FakeTask(magic="UNREACHABLE")
        claim = SecurityClaim.from_tasks([task])

        captured_observables: list[Any] = []
        original_init = GEPAOptimizer.initialize

        async def _spy_init(
            self, goal, controllables, observables, llm_client
        ):  # noqa: ANN001
            captured_observables.append(list(observables))
            await original_init(self, goal, controllables, observables, llm_client)

        async def _stub_propose(self, *, current_instruction, rollouts):  # noqa: ANN001
            return _refl("more attempts")

        with patch.object(GEPAOptimizer, "initialize", new=_spy_init), patch.object(
            __import__(
                "gepa_optimizer.reflector", fromlist=["Reflector"]
            ).Reflector,
            "propose",
            new=_stub_propose,
        ):
            controller = Controller(
                optimizer_factory=lambda: GEPAOptimizer(max_attempts=2),
                target=target,
                security_claim=claim,
                llm_configs=[
                    LLMConfig(model="m", api_base="http://x", api_key="sk"),
                ],
            )
            user_only = frozenset({USER_TAG})
            await controller.run(scopes=[user_only])

        assert captured_observables, "initialize() must have been called"
        # Out-of-scope: nothing for the optimizer to read.
        assert captured_observables[0] == []
