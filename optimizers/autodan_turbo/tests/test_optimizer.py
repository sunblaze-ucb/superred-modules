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

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.exceptions import APIConnectionError

from superred.core.channel import EventEnvelope
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.llm import BudgetExhaustedError, LLMUsage
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

from autodan_turbo_optimizer import retry
from autodan_turbo_optimizer.attacker import AttackerOutput
from autodan_turbo_optimizer.optimizer import AutoDANTurboOptimizer
from autodan_turbo_optimizer.summarizer import StrategyDescriptor
from superred.core.types.observable import ObservableValue

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
    observables: list[ObservableValue] | None = None,
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
        observables=observables if observables is not None else [],
        llm_client=_empty_llm(),
    )
    return opt


def _model_observable(model_id: str) -> ObservableValue:
    obs = Observable(name="model", security_domain=USER_TAG)
    return ObservableValue(observable=obs, content=model_id)


def _system_prompt_observable(content: str) -> ObservableValue:
    obs = Observable(name="system_prompt", security_domain=SYSTEM_PROMPT_TAG)
    return ObservableValue(observable=obs, content=content)


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
    warm_up: str | AttackerOutput = "WARM",
    use_strategy: str | AttackerOutput = "USE",
    find_new_strategy: str | AttackerOutput = "NEW",
) -> MagicMock:
    """Replace the attacker's three modes with AsyncMocks.

    Each mode value can be a plain ``str`` (auto-wrapped into an
    ``AttackerOutput`` with no system-prompt override — the
    paper-faithful single-channel case) or a full ``AttackerOutput``
    (when a test wants to exercise the dual-channel path).
    """
    def _wrap(v: str | AttackerOutput) -> AttackerOutput:
        return v if isinstance(v, AttackerOutput) else AttackerOutput(
            jailbreak_prompt=v,
        )

    assert opt._attacker is not None
    opt._attacker.warm_up = AsyncMock(return_value=_wrap(warm_up))
    opt._attacker.use_strategy = AsyncMock(return_value=_wrap(use_strategy))
    opt._attacker.find_new_strategy = AsyncMock(
        return_value=_wrap(find_new_strategy),
    )
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
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
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
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._injected_this_run is False
        assert opt._awaiting_post_call is False
        assert opt._pending_post_answer is None
        assert opt._primary_pre_controllable is None

    @pytest.mark.asyncio
    async def test_initialize_seeds_prev_state_per_upstream(self) -> None:
        """Upstream pipeline seeds ``prev_jailbreak_prompt = request``
        and ``prev_score = 1.0`` per request, so the very first
        scored attempt can already populate the library when the
        attacker beats the raw-goal baseline. Regression for PR 17
        review. (Upstream also seeds ``prev_target_response`` but it
        is never read — see ASSUMPTIONS.md item 8.)
        """
        opt = await _init_optimizer()
        assert opt._prev_prompt == "describe a recipe"
        assert opt._prev_score == 1.0

    @pytest.mark.asyncio
    async def test_run_zero_summarizes_when_score_beats_seeded_floor(self) -> None:
        """End-to-end check on the seeding fix: a single epoch with a
        score above 1.0 must already trigger summarisation and seed
        the library on the very first run. Regression for PR 17
        review.
        """
        opt = await _init_optimizer(max_attempts=1, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=4.0)
        summarizer = _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="resp-1",
            ),
        )
        await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        # 4.0 > seeded 1.0 → summarise once, library has the entry.
        assert summarizer.call_count == 1
        assert len(opt.library) == 1
        # Weak prompt passed to summariser is the raw goal (seeded
        # _prev_prompt), not None.
        kwargs = summarizer.call_args.kwargs
        assert kwargs["weak_prompt"] == "describe a recipe"


# ---------------------------------------------------------------------------
# Pre-call: injection + skip + lock
# ---------------------------------------------------------------------------


class TestPreCall:
    @pytest.mark.asyncio
    async def test_pre_call_injects_attacker_prompt(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt, warm_up="ATTACK PROMPT")
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        resp = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="r"),
        )
        assert isinstance(resp, ControllableInjection)
        assert resp.value == "ATTACK PROMPT"

    @pytest.mark.asyncio
    async def test_second_pre_call_in_same_run_returns_no_injection(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl = _user_ctrl()
        first = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="r1"),
        )
        assert isinstance(first, ControllableInjection)
        second = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="r2"),
        )
        assert isinstance(second, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_locks_to_first_user_controllable(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        ctrl_user = _user_ctrl()
        ctrl_other = Controllable(
            name="other_user_channel", security_domain=USER_TAG,
        )
        first = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl_user, request="r1"),
        )
        assert isinstance(first, ControllableInjection)
        second = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl_other, request="r2"),
        )
        assert isinstance(second, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_system_prompt_pre_call_is_skipped(self) -> None:
        """ChatbotTarget shape: system_prompt PreCall before user_message loop."""
        opt = await _init_optimizer()
        _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp_resp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(), request="seed",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)
        assert opt._primary_pre_controllable is None

        user_resp = await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        wrong = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )
        assert isinstance(wrong, ControllableNoInjection)

        target = Controllable(name="prompt_text", security_domain=USER_TAG)
        right = await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        ctrl = _user_ctrl()
        pre = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="seed"),
        )
        injected = pre.value  # type: ignore[attr-defined]

        await _dispatch_event(opt,
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
        """Upstream-faithful epoch-0 seeding: prev_score=1.0,
        prev_prompt=raw goal. The first scored attempt that beats 1.0
        already fires the summarizer (raw goal as weak vs attacker
        output as strong).
        """
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt)

        # Run 1: cold attacker, score=3.0. With seeded prev_score=1.0
        # the summarizer fires immediately and the library gets its
        # first entry.
        traj1 = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj1))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj1.push(
            ObservableEvent(
                observable=_response_observable(), content="resp-1",
            ),
        )
        await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert summarizer.call_count == 1
        assert len(opt.library) == 1

        # Run 2: prev_score is now 3.0, score=7.0 > 3.0 so summarizer
        # fires again. The library entry already exists for the same
        # strategy name, so add() merges (still len==1) and appends a
        # second example/score.
        traj2 = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj2))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r2",
            ),
        )
        traj2.push(
            ObservableEvent(
                observable=_response_observable(), content="resp-2",
            ),
        )
        await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )

        assert summarizer.call_count == 2
        assert len(opt.library) == 1
        assert scorer.call_count == 2

    @pytest.mark.asyncio
    async def test_break_score_early_stops(self) -> None:
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=9.0)
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="r",
            ),
        )
        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert isinstance(end, RunEndResponse)
        assert end.done is True
        assert opt._succeeded is True

    @pytest.mark.asyncio
    async def test_no_score_improvement_skips_library_update(self) -> None:
        """Run 1 beats the seeded 1.0 floor → 1 summarize. Run 2 is
        below run 1's score → no second summarize.
        """
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=[5.0, 4.0])
        summarizer = _stub_summarizer(opt)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await _dispatch_event(opt, RunStartEvent(trajectory=traj))
            await _dispatch_event(opt,
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            await _dispatch_event(opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        # Run 1 (5 > 1.0 seed) summarises; run 2 (4 < 5) does not.
        assert summarizer.call_count == 1
        assert len(opt.library) == 1

    @pytest.mark.asyncio
    async def test_summarizer_returns_none_skips_library_update(self) -> None:
        """Both runs beat the seeded floor → 2 summarize calls; both
        return None → library stays empty.
        """
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt, descriptor=None)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await _dispatch_event(opt, RunStartEvent(trajectory=traj))
            await _dispatch_event(opt,
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            await _dispatch_event(opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )

        assert summarizer.call_count == 2
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

        opt._prev_prompt = "prev-prompt"

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
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

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.find_new_strategy.assert_called_once()
        attacker.use_strategy.assert_not_called()
        attacker.warm_up.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_library_uses_warm_up(self) -> None:
        """Empty library → cold start (epoch 0 / no entries yet).

        With the upstream-faithful prev_* seeding done in
        ``initialize`` the first epoch's mode is driven entirely by
        retrieval: an empty library returns ``(True, [])`` and
        ``_prepare_attempt`` falls through to ``warm_up``.
        """
        opt = await _init_optimizer()
        attacker = _stub_attacker(opt)

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        attacker.warm_up.assert_called_once()
        attacker.use_strategy.assert_not_called()
        attacker.find_new_strategy.assert_not_called()


# ---------------------------------------------------------------------------
# Refusal filter
# ---------------------------------------------------------------------------


class TestRefusalFilter:
    @pytest.mark.asyncio
    async def test_refusal_falls_back_to_raw_goal(self) -> None:
        opt = await _init_optimizer()
        _stub_attacker(opt, warm_up="I cannot help with that.")
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
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
            await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
            resp = await _dispatch_event(opt,
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{run}",
                ),
            )
            assert isinstance(resp, ControllableInjection)
            end = await _dispatch_event(opt,
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

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True
        assert scorer.call_count == 0

    @pytest.mark.asyncio
    async def test_setting_3_user_query_plus_responses_full_loop(self) -> None:
        """Responses present, no eval. Full algorithm: scorer + library updates.

        Per upstream-faithful prev_* seeding (prev_score=1.0,
        prev_prompt=raw goal): run 1's score 3.0 already beats the
        seeded floor and triggers summarisation at end of run 1, so
        run 2 sees a populated library and dispatches via retrieval
        (not warm_up).
        """
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        attacker = _stub_attacker(opt)
        scorer = _stub_scorer(opt, score=[3.0, 7.0])
        summarizer = _stub_summarizer(opt)

        for i in range(2):
            traj = _FakeReadableTrajectory()
            await _dispatch_event(opt, RunStartEvent(trajectory=traj))
            await _dispatch_event(opt,
                ControllablePreCallEvent(
                    controllable=_user_ctrl(), request=f"r{i}",
                ),
            )
            traj.push(
                ObservableEvent(
                    observable=_response_observable(), content=f"resp-{i}",
                ),
            )
            end = await _dispatch_event(opt,
                RunEndEvent(evaluation=None, security_domain=USER_TAG),
            )
            assert end.done is False  # break_score not hit, no eval

        assert scorer.call_count == 2
        # Both runs improve over the previous score → 2 summarises.
        assert summarizer.call_count == 2
        # Same strategy name → library merges to 1 entry, 2 examples.
        assert len(opt.library) == 1
        # Run 1 cold-started (empty library); run 2 dispatched via
        # retrieval thanks to the run-1 library seed.
        assert attacker.warm_up.call_count == 1

    @pytest.mark.asyncio
    async def test_setting_4_full_access(self) -> None:
        """Responses + feedback. Full algorithm + external early-stop."""
        opt = await _init_optimizer(max_attempts=10, break_score=8.5)
        _stub_attacker(opt)
        _stub_scorer(opt, score=4.0)  # below break_score
        _stub_summarizer(opt)

        traj = _FakeReadableTrajectory()
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r1",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="resp",
            ),
        )
        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=_success_eval(), security_domain=USER_TAG),
        )
        assert end.done is True  # external eval early-stops

    @pytest.mark.asyncio
    async def test_failure_eval_does_not_signal_done(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end = await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        ctrl = _user_ctrl()
        pre = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="r"),
        )
        injected = pre.value  # type: ignore[attr-defined]
        await _dispatch_event(opt,
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
        await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        ctrl = _user_ctrl()
        pre = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=ctrl, request="r"),
        )
        injected = pre.value  # type: ignore[attr-defined]
        await _dispatch_event(opt,
            ControllablePostCallEvent(
                controllable=_response_ctrl(),
                request=injected, answer="postcall-reply",
            ),
        )
        await _dispatch_event(opt,
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

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end1 = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end1.done is False

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        end2 = await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))
        await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_user_ctrl(), request="r",
            ),
        )
        traj.push(
            ObservableEvent(
                observable=_response_observable(), content="resp",
            ),
        )
        end = await _dispatch_event(opt,
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
        await _dispatch_event(opt, RunStartEvent(trajectory=traj))

        sp_resp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp_resp, ControllableNoInjection)

        user_ctrl = _user_ctrl()
        response_ctrl = _response_ctrl()

        pre = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre, ControllableInjection)
        assert pre.value == "JAILBREAK PROMPT"

        await _dispatch_event(opt,
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
        pre2 = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=user_ctrl, request="seed"),
        )
        assert isinstance(pre2, ControllableNoInjection)

        end = await _dispatch_event(opt,
            RunEndEvent(evaluation=None, security_domain=USER_TAG),
        )
        assert end.done is True
        assert opt._succeeded is True


# ---------------------------------------------------------------------------
# Capability utilization: target_context (static observables) propagation
# ---------------------------------------------------------------------------


class TestTargetContextPropagation:
    """Static observables in scope are captured in ``initialize`` and
    threaded into the attacker's ``target_context`` parameter so the
    attacker can tailor its persuasion. Empty / missing observables
    degrade silently to paper-faithful (no [TARGET CONTEXT] block).
    """

    @pytest.mark.asyncio
    async def test_no_observables_yields_empty_target_context(self) -> None:
        opt = await _init_optimizer(observables=[])
        assert opt._target_context == {}

    @pytest.mark.asyncio
    async def test_observables_captured_into_target_context(self) -> None:
        opt = await _init_optimizer(
            observables=[
                _model_observable("anthropic/claude-3-5-sonnet"),
                _system_prompt_observable("You are CARE-bot."),
            ],
        )
        assert opt._target_context == {
            "model": "anthropic/claude-3-5-sonnet",
            "system_prompt": "You are CARE-bot.",
        }

    @pytest.mark.asyncio
    async def test_empty_string_observables_are_dropped(self) -> None:
        opt = await _init_optimizer(
            observables=[
                _model_observable("   "),
                _system_prompt_observable(""),
            ],
        )
        assert opt._target_context == {}

    @pytest.mark.asyncio
    async def test_target_context_passed_to_warm_up(self) -> None:
        opt = await _init_optimizer(
            observables=[_model_observable("gpt-4o")],
        )
        attacker = _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        attacker.warm_up.assert_called_once()
        call = attacker.warm_up.call_args
        assert call.kwargs["target_context"] == {"model": "gpt-4o"}
        assert call.kwargs["system_prompt_writable"] is False

    @pytest.mark.asyncio
    async def test_target_context_none_when_observables_empty(self) -> None:
        opt = await _init_optimizer(observables=[])
        attacker = _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        # We pass ``None`` (not ``{}``) to keep the attacker's wire
        # protocol unambiguous.
        assert attacker.warm_up.call_args.kwargs["target_context"] is None

    @pytest.mark.asyncio
    async def test_target_context_passed_to_use_strategy(self) -> None:
        opt = await _init_optimizer(
            observables=[_model_observable("claude-3-opus")],
        )
        attacker = _stub_attacker(opt)
        opt._library.add(
            strategy="Storytelling", definition="d",
            example="strong-prompt", score=7.0,
        )
        opt._prev_prompt = "p"
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        attacker.use_strategy.assert_called_once()
        kwargs = attacker.use_strategy.call_args.kwargs
        assert kwargs["target_context"] == {"model": "claude-3-opus"}


# ---------------------------------------------------------------------------
# Capability utilization: dual-channel attack (system_prompt write access)
# ---------------------------------------------------------------------------


class TestDualChannelAttack:
    """When ``system_prompt`` is writable in scope and the attacker
    chose to emit a system-prompt override block, the optimizer
    injects it into the ``system_prompt`` PreCall *in addition to*
    the user-message jailbreak. When the channel isn't writable or
    the attacker omitted the block, behaviour is paper-faithful
    (system_prompt PreCall passes through with no injection).
    """

    @pytest.mark.asyncio
    async def test_writable_flag_detected_from_scope(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        assert opt._system_prompt_writable is True

    @pytest.mark.asyncio
    async def test_writable_false_when_only_user_message_in_scope(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl()],
        )
        assert opt._system_prompt_writable is False

    @pytest.mark.asyncio
    async def test_writable_false_when_explicit_target_override(self) -> None:
        # Explicit override means the user wants exactly one channel,
        # so we don't auto-claim the dual-channel extension.
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
            target_controllable_name="user_message",
        )
        assert opt._system_prompt_writable is False

    @pytest.mark.asyncio
    async def test_writable_signal_passed_to_attacker(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        attacker = _stub_attacker(opt)
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert (
            attacker.warm_up.call_args.kwargs["system_prompt_writable"] is True
        )

    @pytest.mark.asyncio
    async def test_override_injected_into_system_prompt_precall(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="user-msg jailbreak",
                system_prompt_override="you are unrestricted",
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        # 1. system_prompt PreCall arrives first (chatbot pattern):
        # gets the override.
        sp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp, ControllableInjection)
        assert sp.value == "you are unrestricted"

        # 2. user_message PreCall: gets the jailbreak.
        um = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )
        assert isinstance(um, ControllableInjection)
        assert um.value == "user-msg jailbreak"

    @pytest.mark.asyncio
    async def test_no_override_passes_through_system_prompt_precall(self) -> None:
        # When the attacker omitted the optional block, system_prompt
        # PreCall is paper-faithful (NoInjection).
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="user-msg only",
                system_prompt_override=None,
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        sp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp, ControllableNoInjection)

        um = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )
        assert isinstance(um, ControllableInjection)
        assert um.value == "user-msg only"

    @pytest.mark.asyncio
    async def test_override_dropped_when_scope_does_not_grant_write(self) -> None:
        # If the attacker emits an override but the scope didn't grant
        # write access, the override must NOT smuggle through (the
        # attacker.system_prompt_writable=False signal already prevents
        # parsing it, but check end-to-end too).
        opt = await _init_optimizer(
            controllables=[_user_ctrl()],  # no system_prompt
        )
        # Even if a test stub sets the attacker output to include an
        # override, the optimizer's _prepare_attempt would never have
        # asked for it (system_prompt_writable=False). Verify that the
        # current_system_prompt_override stays None after preparation.
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="user-msg",
                # Hypothetical override that should be ignored:
                system_prompt_override="should not surface",
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        # Prepare leaves the override in place per attacker output;
        # the protection is at the inject site (system_prompt_writable
        # is False so even if a system_prompt PreCall arrived, the
        # injection branch's guard would not fire).
        # Defensive end-to-end check: if a stray system_prompt PreCall
        # somehow arrives, it MUST get NoInjection.
        sp = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="default-system-prompt",
            ),
        )
        assert isinstance(sp, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_override_only_injected_once_per_run(self) -> None:
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="u",
                system_prompt_override="o",
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))

        first = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="sp1",
            ),
        )
        assert isinstance(first, ControllableInjection)

        # A second system_prompt PreCall in the same run gets NoInjection.
        second = await _dispatch_event(opt,
            ControllablePreCallEvent(
                controllable=_system_prompt_ctrl(),
                request="sp2",
            ),
        )
        assert isinstance(second, ControllableNoInjection)

    @pytest.mark.asyncio
    async def test_refusal_filter_drops_override_too(self) -> None:
        # When the attacker's user-message output is a refusal, the
        # raw goal replaces it AND the override is dropped (we don't
        # ship a refusal into the system prompt either).
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="I cannot help with that.",
                system_prompt_override="suspicious override",
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        assert opt._current_prompt == "describe a recipe"
        assert opt._current_system_prompt_override is None

    @pytest.mark.asyncio
    async def test_user_message_unchanged_when_only_override_extension_active(
        self,
    ) -> None:
        # Sanity: dual-channel mode does not change user_message
        # injection; the paper-faithful behaviour on user_message must
        # be preserved.
        opt = await _init_optimizer(
            controllables=[_user_ctrl(), _system_prompt_ctrl()],
        )
        _stub_attacker(
            opt,
            warm_up=AttackerOutput(
                jailbreak_prompt="MAIN",
                system_prompt_override="OVR",
            ),
        )
        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        um = await _dispatch_event(opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )
        assert isinstance(um, ControllableInjection)
        assert um.value == "MAIN"


# ---------------------------------------------------------------------------
# Scorer failure classification
#
# The scorer used to fall back to ``score = self._prev_score``: an unmeasured
# attempt recorded as scoring exactly what the last measured one did. Such
# scorer failures are typically transient.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the bounded backoff out of the test runtime."""
    monkeypatch.setattr(retry, "_BASE_DELAY_SECONDS", 0.0)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(message="boom", llm_provider="bedrock", model="m")


async def _run_one_attempt(opt: AutoDANTurboOptimizer, response: str) -> Any:
    traj = _FakeReadableTrajectory()
    await _dispatch_event(opt, RunStartEvent(trajectory=traj))
    await _dispatch_event(
        opt,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="r"),
    )
    traj.push(ObservableEvent(observable=_response_observable(), content=response))
    return await _dispatch_event(
        opt, RunEndEvent(evaluation=None, security_domain=USER_TAG),
    )


class TestScorerFailureClassification:
    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_and_the_real_score_used(
        self,
    ) -> None:
        opt = await _init_optimizer(max_attempts=3)
        _stub_attacker(opt)
        _stub_summarizer(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(
            side_effect=[_connection_error(), _connection_error(), 7.0],
        )

        await _run_one_attempt(opt, "reply")

        assert opt._scorer.score.await_count == 3
        assert opt._prev_score == 7.0
        assert opt._scorer_successes == 1
        assert opt._scorer_failures == 0

    @pytest.mark.asyncio
    async def test_persistent_failure_leaves_the_attempt_unscored(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        _stub_attacker(opt)
        summarizer = _stub_summarizer(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(side_effect=_connection_error())

        await _run_one_attempt(opt, "reply")

        # No fabricated measurement: prev_* still describe the seeded
        # raw-goal baseline, not the prompt no scorer ever saw.
        assert opt._prev_score == 1.0
        assert opt._prev_prompt == "describe a recipe"
        assert summarizer.call_count == 0
        assert opt._scorer_failures == 1

    @pytest.mark.asyncio
    async def test_budget_exhaustion_escapes_and_is_not_retried(self) -> None:
        opt = await _init_optimizer(max_attempts=3)
        _stub_attacker(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(
            side_effect=BudgetExhaustedError("budget exhausted", LLMUsage()),
        )

        with pytest.raises(BudgetExhaustedError):
            await _run_one_attempt(opt, "reply")

        assert opt._scorer.score.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failed_scorer_never_reports_a_break_score_success(
        self,
    ) -> None:
        opt = await _init_optimizer(max_attempts=1, break_score=1.0)
        _stub_attacker(opt)
        _stub_summarizer(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(side_effect=_connection_error())

        result = await _run_one_attempt(opt, "reply")

        assert isinstance(result, RunEndResponse)
        assert result.done is True
        # break_score=1.0 would have been met by the reused prev_score of 1.0.
        assert opt._succeeded is False

    @pytest.mark.asyncio
    async def test_a_task_that_never_scored_is_logged_as_degraded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        opt = await _init_optimizer(max_attempts=1)
        _stub_attacker(opt)
        assert opt._scorer is not None
        opt._scorer.score = AsyncMock(side_effect=_connection_error())

        with caplog.at_level(logging.ERROR):
            await _run_one_attempt(opt, "reply")
            await opt.teardown()

        assert "never produced a score" in caplog.text


class TestBlankPromptIsNeverInjected:
    @pytest.mark.asyncio
    async def test_blank_current_prompt_declines_the_turn(self) -> None:
        # Defence in depth behind the tag-extraction fallback: an empty user
        # message is dropped by litellm's Bedrock Converse transform, and the
        # provider then rejects the whole conversation, failing the task.
        opt = await _init_optimizer()
        _stub_attacker(opt, warm_up="   ")

        await _dispatch_event(opt, RunStartEvent(trajectory=_FakeReadableTrajectory()))
        response = await _dispatch_event(
            opt,
            ControllablePreCallEvent(controllable=_user_ctrl(), request="x"),
        )

        assert isinstance(response, ControllableNoInjection)
