"""Tests for AutoDANTurboOptimizer event-driven state machine.

Covers:
* RunStart / PreCall / PostCall / RunEnd dispatch and state.
* Cold attacker on first run, retrieve-driven mode on subsequent runs.
* Trajectory-first response observation; PostCall fallback.
* Internal scorer drives library updates only on score improvement.
* Internal-success break_score early-stop.
* External evaluation early-stop.
* All four adversarial information-access settings.
* Refusal-filter fallback when attacker returns "I cannot".
* ``system_prompt`` PreCall doesn't steal the primary lock.
* ``target_controllable_name`` override locks injection.
* ``max_no_signal_runs`` blind-loop guard.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

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

from autodan_turbo_optimizer.optimizer import AutoDANTurboOptimizer
from autodan_turbo_optimizer.summarizer import StrategyDescriptor

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

    Optimizer tests patch attacker.warm_up / use_strategy /
    find_new_strategy and Scorer.score / Summarizer.summarize, so the
    underlying LLM should never be invoked.
    """
    llm = MagicMock()

    async def fail(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(
            "Underlying LLM should not be called; tests patch the drivers"
        )

    llm.complete.side_effect = fail
    return llm


async def _init_optimizer(
    *,
    controllables: list[Controllable] | None = None,
    max_attempts: int = 3,
    break_score: float = 8.5,
    target_controllable_name: str | None = None,
    max_no_signal_runs: int = 0,
) -> AutoDANTurboOptimizer:
    opt = AutoDANTurboOptimizer(
        max_attempts=max_attempts,
        break_score=break_score,
        target_controllable_name=target_controllable_name,
        max_no_signal_runs=max_no_signal_runs,
    )
    await opt.initialize(
        goal=Goal(description="describe a recipe"),
        controllables=(
            controllables if controllables is not None else [_user_ctrl()]
        ),
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


def _stub_attacker(
    opt: AutoDANTurboOptimizer,
    *,
    warm_up: str = "WARM",
    use_strategy: str = "USE",
    find_new_strategy: str = "NEW",
) -> MagicMock:
    """Replace the attacker's three modes with AsyncMocks returning fixed text."""
    assert opt._attacker is not None
    opt._attacker.warm_up = AsyncMock(return_value=warm_up)
    opt._attacker.use_strategy = AsyncMock(return_value=use_strategy)
    opt._attacker.find_new_strategy = AsyncMock(return_value=find_new_strategy)
    return opt._attacker  # type: ignore[return-value]


def _stub_scorer(opt: AutoDANTurboOptimizer, score: float | list[float]) -> AsyncMock:
    assert opt._scorer is not None
    if isinstance(score, list):
        mock = AsyncMock(side_effect=score)
    else:
        mock = AsyncMock(return_value=score)
    opt._scorer.score = mock
    return mock


def _stub_summarizer(
    opt: AutoDANTurboOptimizer,
    descriptor: StrategyDescriptor | None = StrategyDescriptor(
        strategy="Storytelling", definition="narrative",
    ),
) -> AsyncMock:
    assert opt._summarizer is not None
    mock = AsyncMock(return_value=descriptor)
    opt._summarizer.summarize = mock
    return mock


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_invalid_max_attempts(self) -> None:
        with pytest.raises(ValueError):
            AutoDANTurboOptimizer(max_attempts=0)

    @pytest.mark.parametrize("bs", [0.5, 10.5, -1.0])
    def test_rejects_invalid_break_score(self, bs: float) -> None:
        with pytest.raises(ValueError):
            AutoDANTurboOptimizer(break_score=bs)

    def test_rejects_invalid_top_k(self) -> None:
        with pytest.raises(ValueError):
            AutoDANTurboOptimizer(top_k_strategies=0)


# ---------------------------------------------------------------------------
# Run start
# ---------------------------------------------------------------------------


class TestRunStart:
    @pytest.mark.asyncio
    async def test_first_run_uses_warm_up_attacker(self) -> None:
        opt = await _init_optimizer()
        attacker = _stub_attacker(opt)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.warm_up.assert_called_once()
        attacker.use_strategy.assert_not_called()
        attacker.find_new_strategy.assert_not_called()
        assert opt._current_prompt == "WARM"

    @pytest.mark.asyncio
    async def test_run_start_resets_per_run_state(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt)
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
    async def test_pre_call_injects_attacker_prompt(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt, warm_up="ATTACK PROMPT")
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="r"),
        )
        assert isinstance(resp, ControllableInjection)
        assert resp.value == "ATTACK PROMPT"

    @pytest.mark.asyncio
    async def test_second_pre_call_in_same_run_returns_no_injection(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt)
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
        _stub_attacker(opt)
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

    @pytest.mark.asyncio
    async def test_system_prompt_pre_call_is_skipped(self) -> None:
        """ChatbotTarget shape: system_prompt PreCall before user_message loop."""
        opt = await _init_optimizer()
        _stub_attacker(opt)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="seed",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)
        assert opt._primary_pre_controllable is None

        user_resp = await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="seed",
            ),
        )
        assert isinstance(user_resp, ControllableInjection)


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
        _stub_attacker(opt)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        wrong = await opt.on_event(
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )
        assert isinstance(wrong, ControllableNoInjection)

        target = Controllable(name="prompt_text", security_domain=USER_TAG)
        right = await opt.on_event(
            ControllablePreCallEvent(controllable=target, request="x"),
        )
        assert isinstance(right, ControllableInjection)


# ---------------------------------------------------------------------------
# Post-call
# ---------------------------------------------------------------------------


class TestPostCall:
    @pytest.mark.asyncio
    async def test_post_call_records_answer(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="seed"),
        )
        injected = pre.value  # type: ignore[attr-defined]

        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request=injected,
                answer="dangerous reply",
            ),
        )
        assert opt._pending_post_answer == "dangerous reply"


# ---------------------------------------------------------------------------
# Run end + scoring + library updates
# ---------------------------------------------------------------------------


class TestRunEndScoring:
    @pytest.mark.asyncio
    async def test_score_increase_triggers_summarizer_and_library_add(
        self,
    ) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        attacker = _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt)

        # Run 1: cold, score=3.0 (no prev_prompt yet → no summarize).
        traj1 = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj1))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj1.push(
            ObservableEvent(
                observable=_response_observable(), content="resp-1",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert summarizer.call_count == 0
        assert len(opt.library) == 0

        # Run 2: cold-warm-up still (because prev_prompt is set but
        # we want to check retrieval — library is empty so still warm).
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r2",
            ),
        )
        traj2 = opt._trajectory
        assert traj2 is not None
        traj2.push(  # type: ignore[attr-defined]
            ObservableEvent(
                observable=_response_observable(), content="resp-2",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        # Score went 3 -> 7, prev_prompt was set -> summarizer fires.
        assert summarizer.call_count == 1
        assert len(opt.library) == 1
        # Attacker called twice in warm_up mode (library was empty
        # both times).
        assert attacker.warm_up.call_count == 2
        assert scorer.call_count == 2

    @pytest.mark.asyncio
    async def test_break_score_early_stops(self) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=9.0)
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="r",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert isinstance(end, RunEndResponse)
        assert end.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_no_score_improvement_skips_library_update(self) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=[5.0, 4.0])
        summarizer = _stub_summarizer(opt)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await opt.on_event(RunStartEvent(trajectory=traj))
            await opt.on_event(
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            await opt.on_event(
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert summarizer.call_count == 0
        assert len(opt.library) == 0

    @pytest.mark.asyncio
    async def test_summarizer_returns_none_skips_library_update(self) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt, descriptor=None)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await opt.on_event(RunStartEvent(trajectory=traj))
            await opt.on_event(
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            await opt.on_event(
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert summarizer.call_count == 1  # called but returned None
        assert len(opt.library) == 0


# ---------------------------------------------------------------------------
# Strategy retrieval drives next attacker mode
# ---------------------------------------------------------------------------


class TestStrategyRetrieval:
    @pytest.mark.asyncio
    async def test_high_score_strategy_uses_use_strategy_mode(self) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        attacker = _stub_attacker(opt)
        _stub_scorer(opt, score=4.0)
        _stub_summarizer(opt)

        # Seed the library with a high-score strategy directly.
        opt._library.add(
            strategy="Storytelling", definition="narr",
            example="strong-prompt", score=7.0,
        )

        # Force `prev_response` non-None so retrieval is consulted on
        # the next attempt.
        opt._prev_prompt = "prev-prompt"
        opt._prev_response = "prev-response"

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.use_strategy.assert_called_once()
        attacker.warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_low_score_strategies_uses_find_new(self) -> None:
        opt = await _init_optimizer()
        attacker = _stub_attacker(opt)

        opt._library.add(
            strategy="Bad", definition="d",
            example="bad-example", score=1.0,
        )
        opt._prev_prompt = "p"
        opt._prev_response = "r"

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.find_new_strategy.assert_called_once()
        attacker.use_strategy.assert_not_called()
        attacker.warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prev_response_uses_warm_up(self) -> None:
        """Even with a populated library, no prior response → cold start."""
        opt = await _init_optimizer()
        attacker = _stub_attacker(opt)
        opt._library.add(
            strategy="X", definition="d", example="e", score=8.0,
        )
        # Set prev_prompt but leave prev_response None (paper's
        # "M empty" fallback shape).
        opt._prev_prompt = "p"

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.warm_up.assert_called_once()


# ---------------------------------------------------------------------------
# Refusal filter
# ---------------------------------------------------------------------------


class TestRefusalFilter:
    @pytest.mark.asyncio
    async def test_refusal_falls_back_to_raw_goal(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt, warm_up="I cannot help with that.")
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        # Falls back to the goal description itself.
        assert opt._current_prompt == "describe a recipe"


# ---------------------------------------------------------------------------
# Adversarial information-access settings
# ---------------------------------------------------------------------------


class TestAdversarialSettings:
    """The four settings the project requires."""

    @pytest.mark.asyncio
    async def test_setting_1_user_query_only(self) -> None:
        """No response, no eval. Always cold attacker; no scorer/summarizer calls."""
        opt = await _init_optimizer(max_attempts=2)
        attacker = _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=99.0)
        summarizer = _stub_summarizer(opt)

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
            assert end.done is (run == 1)

        assert attacker.warm_up.call_count == 2
        assert attacker.use_strategy.call_count == 0
        assert attacker.find_new_strategy.call_count == 0
        assert scorer.call_count == 0
        assert summarizer.call_count == 0
        assert len(opt.library) == 0

    @pytest.mark.asyncio
    async def test_setting_2_user_query_plus_feedback_early_stops(self) -> None:
        """Feedback only. Cold attacker; success eval early-stops."""
        opt = await _init_optimizer(max_attempts=10)
        _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=99.0)

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
        assert scorer.call_count == 0

    @pytest.mark.asyncio
    async def test_setting_3_user_query_plus_responses_full_loop(self) -> None:
        """Responses present, no eval. Full algorithm: scorer + library updates."""
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        attacker = _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await opt.on_event(RunStartEvent(trajectory=traj))
            await opt.on_event(
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            end = await opt.on_event(
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            assert end.done is False  # break_score not hit, no eval

        assert scorer.call_count == 2
        assert summarizer.call_count == 1  # 3 -> 7 improvement
        assert len(opt.library) == 1
        # Both runs ran warm_up (library still empty during run 2's
        # prepare_attempt; the new strategy is added at end of run 2).
        assert attacker.warm_up.call_count >= 1

    @pytest.mark.asyncio
    async def test_setting_4_full_access(self) -> None:
        """Responses + feedback. Full algorithm + external early-stop."""
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=4.0)  # below break_score
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="resp",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True  # external eval early-stops

    @pytest.mark.asyncio
    async def test_failure_eval_does_not_signal_done(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        _stub_attacker(opt)
        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end = await opt.on_event(
            RunEndEvent(evaluation=_failure_eval(), security_domain=USER_TAG),
        )
        assert end.done is False


# ---------------------------------------------------------------------------
# Response source: trajectory vs PostCall
# ---------------------------------------------------------------------------


class TestResponseSource:
    @pytest.mark.asyncio
    async def test_trajectory_observable_preferred_over_postcall(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=4.0)
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        ctrl = _user_ctrl()
        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r"),
        )
        injected = pre.value  # type: ignore[attr-defined]
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request=injected, answer="from-postcall",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="from-trajectory",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        assert scorer.call_args.args[1] == "from-trajectory"

    @pytest.mark.asyncio
    async def test_postcall_used_when_no_trajectory_observable(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=4.0)
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        ctrl = _user_ctrl()
        pre = await opt.on_event(
            ControllablePreCallEvent(controllable=ctrl, request="r"),
        )
        injected = pre.value  # type: ignore[attr-defined]
        await opt.on_event(
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request=injected, answer="postcall-reply",
            ),
        )
        await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        assert scorer.call_args.args[1] == "postcall-reply"


# ---------------------------------------------------------------------------
# Blind-loop guard
# ---------------------------------------------------------------------------


class TestNoSignalGuard:
    @pytest.mark.asyncio
    async def test_max_no_signal_runs_terminates(self) -> None:
        opt = await _init_optimizer(
            max_attempts=10, max_no_signal_runs=2,
        )
        _stub_attacker(opt)

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end1 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end1.done is False

        await opt.on_event(RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end2 = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end2.done is True


# ---------------------------------------------------------------------------
# Scorer failure resilience
# ---------------------------------------------------------------------------


class TestScorerFailure:
    @pytest.mark.asyncio
    async def test_scorer_exception_does_not_crash_run(self) -> None:
        opt = await _init_optimizer(max_attempts=2)
        _stub_attacker(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(side_effect=RuntimeError("boom"))
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await opt.on_event(RunStartEvent(trajectory=traj))
        await opt.on_event(
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="resp",
            ),
        )
        end = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert isinstance(end, RunEndResponse)
        # No crash, no early-stop, no library update.
        assert end.done is False
        assert len(opt.library) == 0


# ---------------------------------------------------------------------------
# End-to-end target-loop integration
# ---------------------------------------------------------------------------


class TestTargetRunIntegration:
    @pytest.mark.asyncio
    async def test_chatbot_style_loop(self) -> None:
        """ChatbotTarget shape: system_prompt PreCall, then user_message PreCall."""
        opt = await _init_optimizer(max_attempts=3, break_score=8.5)
        _stub_attacker(opt, warm_up="JAILBREAK PROMPT")
        _stub_scorer(opt, score=9.0)  # success on first attempt
        _stub_summarizer(opt)

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
        assert pre.value == "JAILBREAK PROMPT"

        await opt.on_event(
            ControllablePostCallEvent(
                controllable=response_ctrl,
                request="JAILBREAK PROMPT",
                answer="dangerous reply",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(),
                content="dangerous reply",
            ),
        )

        # Single-turn: target tries another user_message PreCall -> NoInjection.
        pre2 = await opt.on_event(
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre2, ControllableNoInjection)

        end = await opt.on_event(
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end.done is True
        assert opt._succeeded is True
