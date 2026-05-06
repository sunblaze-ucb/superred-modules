"""BijectionOptimizer: bijection-learning jailbreak for superred.

One superred run = one bijection trial. Across runs the optimizer
samples a fresh random bijection at the configured complexity
(paper's "best-of-N" formulation, §1.3 and §3.3 of Huang et al., ICLR
2025). No reflection loop, no LLM-driven attacker — bijection
construction is deterministic given the random seed; the prompt is a
direct port of the upstream
``haizelabs/bijection-learning/run_bijection_attacks.py``.

Adversarial settings
--------------------

The optimizer operates in all four information-access settings without
a setting knob — the framework's scope filter and ``include_feedback``
flag select which information surfaces are visible:

1. user query only — keep sampling fresh bijections until ``max_attempts``
   is exhausted; never declare success (no judge signal).
2. user query + feedback — sample fresh bijections; early-stop on
   ``RunEndEvent.evaluation.success`` (paper's best-of-N stopping rule).
3. user query + responses — same as setting 1: bijections are sampled,
   not adapted, so target responses don't drive the optimizer.
4. user query + responses + feedback — same as setting 2.

The optimizer is intentionally response-blind on the attack-construction
path because the bijection is randomized, not adapted; this matches
the paper's best-of-N formulation exactly.

Refer to ``ASSUMPTIONS.md`` for paper alignment and deliberate
departures.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable

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

from bijection_optimizer.bijection import Bijection, generate_bijection
from bijection_optimizer.prompts import (
    render_attack_prompt,
    render_teaching_intro,
)

logger = logging.getLogger(__name__)


_DEFAULT_RESPONSE_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {"response", "model_response", "assistant_response"}
)

_SYSTEM_PROMPT_NAME = "system_prompt"


class BijectionOptimizer(Optimizer):
    """Bijection learning best-of-N jailbreak optimizer.

    Args:
        bijection_type: ``"letter"`` (alphabet permutation) or
            ``"digit"`` (each non-fixed letter → unique
            ``num_digits``-digit number). Paper's main results use
            ``digit`` for stronger models and ``letter`` for weaker
            ones (Table 1).
        fixed_size: Number of letters that map to themselves. The
            paper's dispersion is ``26 - fixed_size``. Default ``10``
            matches the Sonnet-optimal ``digit`` setting (dispersion
            16) from Table 1.
        num_digits: Encoding length for ``digit`` codomain. Paper
            default 2.
        digit_delimiter: String inserted before each substituted
            numeric token. Paper default two spaces.
        num_teaching_shots: How many English ↔ encoded teaching
            pairs to include in the packed prompt. Paper default 10.
        max_attempts: Attack budget; how many fresh random bijections
            to try before giving up. Paper budgets range 6–47;
            default 6 matches the run script's default.
        response_observable_names: Names recognised as target replies
            on the trajectory (defaults to the same set as Crescendo /
            GEPA / GOAT).
        target_controllable_name: Optional override; when set, lock
            injection onto exactly this named controllable. Default
            ``None`` keeps the paper's threat model — inject into the
            user-message channel; if the controllable scope also grants
            ``system_prompt`` write access, split the prompt across the
            two channels (intro → system_prompt, shots+query →
            user_message).
        max_no_signal_runs: If positive, terminate after this many
            consecutive runs in which neither response nor evaluation
            was visible. Disabled by default (matches GOAT / GEPA).
        seed: Optional ``int`` seed for the bijection RNG so a run is
            reproducible.
    """

    def __init__(
        self,
        *,
        bijection_type: str = "digit",
        fixed_size: int = 10,
        num_digits: int = 2,
        digit_delimiter: str = "  ",
        num_teaching_shots: int = 10,
        max_attempts: int = 6,
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        max_no_signal_runs: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if num_teaching_shots < 0:
            raise ValueError("num_teaching_shots must be >= 0")
        if bijection_type not in {"letter", "digit"}:
            raise ValueError(
                f"bijection_type must be 'letter' or 'digit', got {bijection_type!r}"
            )
        if not 0 <= fixed_size <= 26:
            raise ValueError("fixed_size must be in [0, 26]")

        self._bijection_type = bijection_type
        self._fixed_size = fixed_size
        self._num_digits = num_digits
        self._digit_delimiter = digit_delimiter
        self._num_teaching_shots = num_teaching_shots
        self._max_attempts = max_attempts
        self._response_observable_names: frozenset[str] = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._target_controllable_name = target_controllable_name
        self._max_no_signal_runs = max(0, max_no_signal_runs)
        self._rng = random.Random(seed)

        # Set in initialize().
        self._goal: Goal | None = None
        self._system_prompt_in_scope: bool = False

        # Cross-run state.
        self._attempt: int = 0
        self._succeeded: bool = False
        self._consecutive_no_signal_runs: int = 0
        self._stop_due_to_no_signal: bool = False

        # Per-run state (reset in _reset_run_state).
        self._current_bijection: Bijection | None = None
        self._current_user_message: str = ""
        self._current_system_prompt: str = ""
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
        # Detect whether the caller's scope grants write access to the
        # system_prompt channel — drives the two-channel split.
        self._system_prompt_in_scope = any(
            c.name == _SYSTEM_PROMPT_NAME for c in controllables
        )
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
            return self._handle_run_end(event)
        return EventResponse(event=event)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        self._trajectory = event.trajectory
        self._prepare_attempt()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._target_controllable_name is not None:
            # Explicit-target mode: lock onto exactly this name.
            if event.controllable.name != self._target_controllable_name:
                return ControllableNoInjection(
                    event=event, controllable=event.controllable
                )
        else:
            # Default mode. Two channels supported, mirroring
            # FlipAttack: when the controller's scope grants both
            # system_prompt and user_message, split the prompt; when
            # only user_message is in scope, fold the intro into it.
            if event.controllable.name == _SYSTEM_PROMPT_NAME:
                if not self._system_prompt_in_scope:
                    # Defensive: scope filter didn't grant it but the
                    # event still surfaced. Pass through.
                    return ControllableNoInjection(
                        event=event, controllable=event.controllable
                    )
                return ControllableInjection(
                    event=event,
                    controllable=event.controllable,
                    value=self._current_system_prompt,
                )

        if self._primary_pre_controllable is None:
            self._primary_pre_controllable = event.controllable
        elif event.controllable != self._primary_pre_controllable:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if self._injected_this_run:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        self._injected_this_run = True
        self._awaiting_post_call = True
        self._last_pre_request = event.request
        self._last_injected_value = self._current_user_message
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current_user_message,
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
        if not self._awaiting_post_call:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        # Same 3-way pairing rule used by Crescendo / GEPA / GOAT.
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

    def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        # Trajectory-first response read; fall back to PostCall answer.
        # We don't act on the response (best-of-N is response-blind),
        # but we still track its presence for the no-signal early-stop.
        response_seen = self._read_response_from_trajectory() is not None
        if not response_seen:
            response_seen = self._pending_post_answer is not None

        evaluation = event.evaluation

        signal_seen = response_seen or evaluation is not None
        if signal_seen:
            self._consecutive_no_signal_runs = 0
        else:
            self._consecutive_no_signal_runs += 1
            if (
                self._max_no_signal_runs > 0
                and self._consecutive_no_signal_runs >= self._max_no_signal_runs
            ):
                logger.warning(
                    "Bijection: stopping after %d consecutive runs with no "
                    "visible response or feedback",
                    self._consecutive_no_signal_runs,
                )
                self._stop_due_to_no_signal = True

        if evaluation is not None and evaluation.success:
            self._succeeded = True
            logger.info(
                "Bijection: attempt %d succeeded (score=%.4f)",
                self._attempt, evaluation.primary_score.value,
            )

        self._attempt += 1

        if self._succeeded or self._stop_due_to_no_signal:
            return RunEndResponse(event=event, done=True)
        if self._attempt >= self._max_attempts:
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._current_bijection = None
        self._current_user_message = ""
        self._current_system_prompt = ""
        self._trajectory = None
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._injected_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None

    def _prepare_attempt(self) -> None:
        """Sample a fresh bijection and render the per-run prompts."""
        assert self._goal is not None
        bijection = generate_bijection(
            codomain=self._bijection_type,
            fixed_size=self._fixed_size,
            num_digits=self._num_digits,
            delimiter=self._digit_delimiter,
            rng=self._rng,
        )
        self._current_bijection = bijection

        if self._target_controllable_name is None and self._system_prompt_in_scope:
            # Two-channel split (FlipAttack pattern): teaching intro
            # plus mapping → system_prompt; teaching shots + encoded
            # query → user_message.
            self._current_system_prompt = render_teaching_intro(bijection)
            self._current_user_message = render_attack_prompt(
                bijection=bijection,
                goal=self._goal.description,
                num_teaching_shots=self._num_teaching_shots,
                include_intro=False,
            )
        else:
            self._current_system_prompt = ""
            self._current_user_message = render_attack_prompt(
                bijection=bijection,
                goal=self._goal.description,
                num_teaching_shots=self._num_teaching_shots,
                include_intro=True,
            )

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


__all__ = ["BijectionOptimizer"]
