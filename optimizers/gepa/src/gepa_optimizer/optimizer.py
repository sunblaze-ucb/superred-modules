"""GEPAOptimizer: reflective prompt evolution for superred.

One superred run is one rollout of one prompt candidate. Across runs
GEPA grows a candidate pool by reflective mutation: at the end of each
run we (a) record ``(response, score, rationale)`` on the candidate
that was just rolled out, (b) call a reflection LM on the best-scoring
candidate's recent rollouts to propose a new instruction, and (c)
stash the proposed candidate for the next run's PreCall to inject.

Reference: Agrawal et al., "GEPA: Reflective Prompt Evolution Can
Outperform Reinforcement Learning," arXiv:2507.19457, ICLR 2026; the
official ``gepa-ai/gepa`` reference implementation.

Adversarial settings
--------------------

The optimizer operates in all four information-access settings without
a setting knob — the framework's scope filter and ``include_feedback``
flag select which information surfaces are visible:

1. user query only — no responses, no feedback. Reflective dataset
   contains only the goal text; selection collapses to "latest" since
   all candidates share score 0; no early-stop.
2. user query + feedback — score and rationale drive selection;
   ``RunEndEvent.evaluation.success`` enables early-stop.
3. user query + responses — response traces feed reflection;
   selection collapses to "latest" without scores; no early-stop.
4. user query + responses + feedback — full GEPA loop with early-stop.

Refer to ``ASSUMPTIONS.md`` for paper alignment and deliberate
departures.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
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
from superred.core.types.observable import ObservableValue
from superred.core.types.trajectory import ReadableTrajectory

from gepa_optimizer.reflector import Reflector, RolloutRecord

logger = logging.getLogger(__name__)


_DEFAULT_RESPONSE_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {"response", "model_response", "assistant_response"}
)

# Hardcoded skip rule mirrors GOAT / FlipAttack: target shapes such as
# ChatbotTarget emit a ``system_prompt`` PreCall before the user-message
# loop. Pass it through without locking so the real user channel can
# still claim the primary slot when it arrives.
_SYSTEM_PROMPT_NAME = "system_prompt"


@dataclass
class _Candidate:
    """One prompt candidate plus the rollout it scored on, if any."""

    prompt: str
    parent_idx: int | None = None
    response: str | None = None
    score: float | None = None
    rationale: str = ""
    rolled_out: bool = False

    @property
    def effective_score(self) -> float:
        """Score used for parent selection; 0.0 when no score is visible."""
        return self.score if self.score is not None else 0.0


class GEPAOptimizer(Optimizer):
    """GEPA reflective prompt evolution optimizer.

    One superred run = one rollout of one candidate. Multiple runs grow
    the candidate pool through reflective mutation. Best-scoring
    candidate (latest wins on ties) is the parent for the next
    reflection step.

    Args:
        max_attempts: Budget B — number of superred runs to perform
            (paper's main results sample budgets in the tens to low
            hundreds; default 20 is a reasonable starting budget).
        reflection_temperature: Sampling temperature for the reflection
            LM. The paper uses high temperature for prompt diversity.
        response_observable_names: Names recognised as target replies on
            the trajectory (defaults to ``response``,
            ``model_response``, ``assistant_response`` — same set as
            Crescendo / GOAT).
        max_no_signal_runs: If positive, terminate after this many
            consecutive runs in which neither response nor evaluation
            was visible. Bounds blind-loop cost in the user-query-only
            setting. Disabled by default (matches GOAT / FlipAttack).
    """

    def __init__(
        self,
        *,
        max_attempts: int = 20,
        reflection_temperature: float = 1.0,
        response_observable_names: Iterable[str] | None = None,
        max_no_signal_runs: int = 0,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._max_attempts = max_attempts
        self._reflection_temperature = reflection_temperature
        self._response_observable_names: frozenset[str] = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._max_no_signal_runs = max(0, max_no_signal_runs)

        # Set in initialize().
        self._goal: Goal | None = None
        self._reflector: Reflector | None = None

        # Cross-run state.
        self._pool: list[_Candidate] = []
        self._pending: _Candidate | None = None
        self._attempt: int = 0
        self._succeeded: bool = False
        self._consecutive_no_signal_runs: int = 0
        self._stop_due_to_no_signal: bool = False

        # Per-run state (reset in _reset_run_state).
        self._current: _Candidate | None = None
        self._current_is_fresh: bool = False
        self._trajectory: ReadableTrajectory | None = None
        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._injected_this_run: bool = False
        self._awaiting_post_call: bool = False
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._reflector = Reflector(
            llm=self.llm,
            temperature=self._reflection_temperature,
        )
        self._pool = [_Candidate(prompt=goal.description)]
        self._pending = None
        self._attempt = 0
        self._succeeded = False
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        self._trajectory = event.trajectory
        self._current, self._current_is_fresh = self._select_current_candidate()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        # Pass system_prompt through without locking so the real user
        # channel can still become primary.
        if event.controllable.name == _SYSTEM_PROMPT_NAME:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if self._primary_pre_controllable is None:
            self._primary_pre_controllable = event.controllable
        elif event.controllable != self._primary_pre_controllable:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if self._injected_this_run or self._current is None:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        self._injected_this_run = True
        self._awaiting_post_call = True
        self._last_pre_request = event.request
        self._last_injected_value = self._current.prompt
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current.prompt,
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
        if not self._awaiting_post_call:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        # Same 3-way pairing rule as Crescendo / GOAT.
        if self._primary_post_controllable is None:
            same_ctrl = event.controllable == self._primary_pre_controllable
            request_matches_pre = (
                self._last_pre_request is not None
                and event.request == self._last_pre_request
            )
            request_matches_injected = (
                self._last_injected_value is not None
                and event.request == self._last_injected_value
            )
            if not (same_ctrl or request_matches_pre or request_matches_injected):
                return ControllableNoInjection(
                    event=event, controllable=event.controllable
                )
            self._primary_post_controllable = event.controllable
        elif event.controllable != self._primary_post_controllable:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        self._pending_post_answer = event.answer
        return ControllableNoInjection(
            event=event, controllable=event.controllable
        )

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        # Resolve the response: trajectory observable wins; fall back
        # to PostCall answer; tolerate neither (settings 1 & 2).
        response = self._read_response_from_trajectory()
        if response is None:
            response = self._pending_post_answer

        score: float | None = None
        rationale: str = ""
        evaluation = event.evaluation
        if evaluation is not None:
            score = evaluation.primary_score.value
            rationale = evaluation.rationale

        # Record the rollout against the candidate that produced it.
        # Freshly-proposed candidates enter the pool here; already-in-pool
        # candidates have their fields refreshed in place.
        if self._current is not None:
            self._current.response = response
            self._current.score = score
            self._current.rationale = rationale
            self._current.rolled_out = True
            if self._current_is_fresh:
                self._pool.append(self._current)

        # No-signal tracking (drives optional early-stop).
        signal_seen = response is not None or evaluation is not None
        if signal_seen:
            self._consecutive_no_signal_runs = 0
        else:
            self._consecutive_no_signal_runs += 1
            if (
                self._max_no_signal_runs > 0
                and self._consecutive_no_signal_runs >= self._max_no_signal_runs
            ):
                logger.warning(
                    "GEPA: stopping after %d consecutive runs with no visible "
                    "response or feedback",
                    self._consecutive_no_signal_runs,
                )
                self._stop_due_to_no_signal = True

        # Early-stop on success (only meaningful when feedback in scope).
        if evaluation is not None and evaluation.success:
            self._succeeded = True
            logger.info(
                "GEPA: attempt %d succeeded (score=%.4f)",
                self._attempt, evaluation.primary_score.value,
            )

        self._attempt += 1

        if self._succeeded or self._stop_due_to_no_signal:
            return RunEndResponse(event=event, done=True)
        if self._attempt >= self._max_attempts:
            return RunEndResponse(event=event, done=True)

        # Reflect to set up the next run's candidate.
        await self._reflect_next_candidate()
        return RunEndResponse(event=event, done=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._current = None
        self._current_is_fresh = False
        self._trajectory = None
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._injected_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None

    def _select_current_candidate(self) -> tuple[_Candidate, bool]:
        """Pick the candidate to roll out this run.

        Returns ``(candidate, is_fresh)``. ``is_fresh`` is True when the
        candidate was a freshly-proposed pending mutation and therefore
        needs to enter the pool when its rollout completes; False when
        we re-selected an already-in-pool candidate (e.g. the seed on
        the very first run, or any previously rolled-out candidate
        being re-evaluated).
        """
        if self._pending is not None:
            picked = self._pending
            self._pending = None
            return picked, True
        return self._best_in_pool(), False

    def _best_in_pool(self) -> _Candidate:
        """Return the highest-effective-score candidate; ties to the latest."""
        assert self._pool, "pool always has at least the seed candidate"
        best = self._pool[0]
        for candidate in self._pool[1:]:
            # Latest-wins on ties so the chain progresses in settings
            # without score signal (effective_score is 0 for everyone).
            if candidate.effective_score >= best.effective_score:
                best = candidate
        return best

    def _read_response_from_trajectory(self) -> str | None:
        if self._trajectory is None:
            return None
        latest: str | None = None
        for item in self._trajectory.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            if (
                name in self._response_observable_names
                or name.lower() in self._response_observable_names
            ):
                if isinstance(item.content, str):
                    latest = item.content
        return latest

    async def _reflect_next_candidate(self) -> None:
        """Build the side-info dataset and stash a pending proposal."""
        assert self._reflector is not None and self._goal is not None

        parent = self._best_in_pool()
        if not parent.rolled_out:
            # Seed has not been rolled out yet — defer reflection,
            # next run will roll out the seed first.
            return

        rollouts = [
            RolloutRecord(
                goal=self._goal.description,
                prompt=parent.prompt,
                response=parent.response,
                score=parent.score,
                rationale=parent.rationale,
            ),
        ]

        try:
            result = await self._reflector.propose(
                current_instruction=parent.prompt,
                rollouts=rollouts,
            )
        except Exception:
            logger.warning(
                "GEPA: reflection LM call failed; skipping mutation this iteration",
                exc_info=True,
            )
            return

        if result is None:
            return

        parent_idx = self._pool.index(parent)
        self._pending = _Candidate(
            prompt=result.new_instruction,
            parent_idx=parent_idx,
        )


__all__ = ["GEPAOptimizer"]
