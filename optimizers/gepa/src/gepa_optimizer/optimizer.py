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

Threat-model fidelity
---------------------

The optimizer adapts to the *capability* surface the controller's
scope grants:

* All in-scope static observables (e.g. ``system_prompt``, ``model``)
  are surfaced as a ``target_observables`` dict in the reflective
  dataset so the reflection LM sees whatever capability the threat
  model actually grants — not just one hardcoded surface.
* When the controller's scope grants write access to ``system_prompt``
  the optimizer auto-claims it and attacks that channel by default
  (the higher-leverage attack surface). Otherwise the user-message
  channel is attacked. The explicit ``target_controllable_name``
  constructor knob always wins over auto-claim.

Failed reflection
-----------------

Reflection is the only thing that makes one run differ from the last,
so a reflection that produces nothing is never absorbed: the cost cap
is re-raised, a failing call is retried and then surfaced as
``ReflectionUnavailable``, and an LM that keeps proposing nothing
parseable ends the search. Re-sending the identical prompt to the
target would spend target budget for no search progress and record the
result as an ordinary failed attack. See ``ASSUMPTIONS.md``, "When
Reflection Does Not Produce a Mutation".

Refer to ``ASSUMPTIONS.md`` for paper alignment and deliberate
departures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

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
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from gepa_optimizer.reflector import ReflectionResult, Reflector, RolloutRecord

logger = logging.getLogger(__name__)


_DEFAULT_RESPONSE_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {"response", "model_response", "assistant_response"}
)

# Hardcoded system-prompt name (matches the chatbot target's naming).
# Used for two things: (1) the default-mode skip rule on the
# ``system_prompt`` ControllablePreCallEvent so the user-message
# channel can still claim the primary slot when ``system_prompt`` is
# read-only, and (2) the auto-claim default — when the controller's
# scope grants ``system_prompt`` as a *writable* controllable and the
# caller didn't pin ``target_controllable_name`` explicitly, the
# optimizer prefers it over ``user_message`` because the system
# prompt is the higher-leverage attack surface.
_SYSTEM_PROMPT_NAME = "system_prompt"

# Per-candidate rollout history depth. Matches the GEPA paper's default
# minibatch size of 3, which is what the reflection LM expects to see
# in the side-info dataset.
_ROLLOUT_HISTORY_SIZE = 3

# Base of the exponential backoff between reflection retries, in seconds.
_REFLECTION_BACKOFF_S = 1.0


class ReflectionUnavailable(RuntimeError):
    """Every attempt to reach the reflection LM failed.

    Raised out of ``on_event`` so the controller ends the task with
    ``stop_reason="error"`` and the formatted traceback on
    ``TaskResult.error``. A dead reflection LM means GEPA never searched;
    recording that as an ordinary score-0 attacker failure would make an
    infrastructure outage indistinguishable from a target that held.
    """


def _backoff_delay(attempt: int) -> float:
    """Full-jitter exponential backoff for retry number ``attempt`` (1-based).

    Jittered rather than fixed because a whole matrix cell retries against
    the same provider at the same moment; synchronised retries turn one
    rate-limit into a standing wave.
    """
    return random.uniform(0.0, _REFLECTION_BACKOFF_S * 2 ** (attempt - 1))


@dataclass
class _Candidate:
    """One prompt candidate plus the rollout it scored on, if any.

    ``rollouts`` is a bounded ring buffer of recent rollouts that
    feeds both the reflection LM (recent traces as side-info) and
    parent selection (mean score across the buffer, so a single
    lucky/unlucky trial doesn't dominate over a steadier candidate).
    ``score``/``response``/``rationale`` mirror the *latest* rollout
    for inspection / debugging only.
    """

    prompt: str
    parent_idx: int | None = None
    response: str | None = None
    score: float | None = None
    rationale: str = ""
    rolled_out: bool = False
    rollouts: deque[RolloutRecord] = field(
        default_factory=lambda: deque(maxlen=_ROLLOUT_HISTORY_SIZE)
    )

    @property
    def effective_score(self) -> float:
        """Mean score across the recent-rollouts buffer.

        Returns 0.0 when no scored rollout is available — keeps
        latest-wins tie-breaking working in the no-feedback settings
        where every candidate sits at 0.0.
        """
        scored = [r.score for r in self.rollouts if r.score is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)


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
        response_observable_names: Names recognised as target replies on
            the trajectory (defaults to ``response``,
            ``model_response``, ``assistant_response`` — same set as
            Crescendo / GOAT).
        target_controllable_name: When set, the optimizer locks
            injection onto exactly the controllable with this name and
            ignores all others. Default ``None`` enables auto-claim:
            attack ``system_prompt`` when the controller's scope
            includes it as a writable controllable (higher-leverage
            attack surface); otherwise attack ``user_message`` and
            skip the ``system_prompt`` PreCall (read-only system
            prompt).
        max_no_signal_runs: If positive, terminate after this many
            consecutive runs in which neither response nor evaluation
            was visible. Bounds blind-loop cost in the user-query-only
            setting. Disabled by default (matches GOAT / FlipAttack).
        max_consecutive_no_mutation: Terminate after this many
            consecutive reflections that returned no parseable proposal.
            A re-roll of the same parent is worth something (it refreshes
            the parent's rollout buffer, so the next reflection sees
            different side-info), but once the buffer has turned over the
            reflection LM is being handed input it has already declined;
            further runs spend target budget for zero search progress.
            Default 3 = ``_ROLLOUT_HISTORY_SIZE``. Set to 0 to disable.
        reflection_retries: Extra attempts for a reflection LM call that
            raises (default 2, so 3 attempts) with jittered exponential
            backoff. Transient provider errors are the common case and
            must not cost a task.
        reflection_retry_deadline: Seconds after the first attempt beyond
            which no further retry is started (default 120). Guards the
            task time cap: a provider timeout can itself burn 600 s, and
            three of those would trip the controller's timeout and
            discard the task.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 20,
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        max_no_signal_runs: int = 0,
        max_consecutive_no_mutation: int = _ROLLOUT_HISTORY_SIZE,
        reflection_retries: int = 2,
        reflection_retry_deadline: float = 120.0,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._max_attempts = max_attempts
        self._max_consecutive_no_mutation = max(0, max_consecutive_no_mutation)
        self._reflection_retries = max(0, reflection_retries)
        self._reflection_retry_deadline = max(0.0, reflection_retry_deadline)
        self._response_observable_names: frozenset[str] = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._max_no_signal_runs = max(0, max_no_signal_runs)

        # User-supplied override; resolved (with auto-claim) inside
        # ``initialize`` into ``_target_controllable_name``.
        self._target_controllable_name_override = target_controllable_name

        # Set in initialize().
        self._goal: Goal | None = None
        self._reflector: Reflector | None = None
        self._target_observables: dict[str, str] | None = None
        # Resolved attack channel: override if set, else auto-claim
        # ``system_prompt`` when writable, else ``None`` (default
        # user-message attack with system_prompt skipped).
        self._target_controllable_name: str | None = None

        # Cross-run state.
        self._pool: list[_Candidate] = []
        self._pending: _Candidate | None = None
        self._attempt: int = 0
        self._succeeded: bool = False
        self._consecutive_no_signal_runs: int = 0
        self._stop_due_to_no_signal: bool = False
        self._consecutive_no_mutation: int = 0
        # A reflection failure detected at RunEnd, re-raised at the next
        # RunStart — see ``_handle_run_start``.
        self._pending_failure: BaseException | None = None

        # Per-run state (reset in _reset_run_state).
        self._current: _Candidate | None = None
        self._current_is_fresh: bool = False
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
        # No sampling temperature is sent to the reflection LM: the
        # paper pins a high one for prompt diversity, but reasoning
        # models reject the parameter outright, which would turn every
        # reflection into a hard failure on those models.
        self._reflector = Reflector(llm=self.llm)
        self._target_observables = self._extract_static_observables(observables)
        self._target_controllable_name = self._resolve_target_controllable_name(
            controllables,
        )
        self._pool = [_Candidate(prompt=goal.description)]
        self._pending = None
        self._attempt = 0
        self._succeeded = False
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False
        self._consecutive_no_mutation = 0
        self._pending_failure = None
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
        if self._pending_failure is not None:
            # Reflection died at the end of the previous run. Raise here
            # rather than there: the controller sends RunStartEvent before
            # it calls the target, so the task ends without paying for one
            # more target call, and the previous run's completed
            # evaluation is already recorded instead of being replaced by
            # the controller's synthetic zero-score error result.
            failure = self._pending_failure
            self._pending_failure = None
            raise failure
        self._reset_run_state()
        self._current, self._current_is_fresh = self._select_current_candidate()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._target_controllable_name is not None:
            # Explicit-target mode: lock onto exactly this name; skip
            # everything else (including the otherwise-skipped
            # ``system_prompt`` channel if the user picked something
            # else).
            if event.controllable.name != self._target_controllable_name:
                return ControllableNoInjection(
                    event=event, controllable=event.controllable
                )
        else:
            # Default mode: skip the system-prompt PreCall without
            # locking, then lock onto the first remaining controllable.
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
        # candidates have their fields refreshed in place and the rollout
        # appended to their bounded history (for reflection only).
        if self._current is not None:
            assert self._goal is not None
            self._current.response = response
            self._current.score = score
            self._current.rationale = rationale
            self._current.rolled_out = True
            self._current.rollouts.append(
                RolloutRecord(
                    goal=self._goal.description,
                    prompt=self._current.prompt,
                    response=response,
                    score=score,
                    rationale=rationale,
                    target_observables=self._target_observables,
                )
            )
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
        if not await self._reflect_next_candidate():
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._current = None
        self._current_is_fresh = False
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
        if self.current_trajectory is None:
            return None
        latest: str | None = None
        for item in self.current_trajectory.drain():
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

    @staticmethod
    def _extract_static_observables(
        observables: list[ObservableValue],
    ) -> dict[str, str] | None:
        """Return a name → content dict of in-scope static observables.

        ``observables`` is already scope-filtered by the controller, so
        every entry the optimizer sees here is one the threat model
        explicitly granted read access to. Each value is surfaced as a
        ``target_observables`` field on every ``RolloutRecord`` so the
        reflection LM sees whatever capability the controller actually
        granted (system prompt, model identity, …) rather than just
        one hardcoded surface.

        Non-string values and empty / whitespace strings are dropped
        (matches the ``format_reflective_dataset`` field-skip rule).
        """
        out: dict[str, str] = {}
        for value in observables:
            content = value.content
            if isinstance(content, str) and content.strip():
                out[value.observable.name] = content
        return out or None

    def _resolve_target_controllable_name(
        self, controllables: list[Controllable],
    ) -> str | None:
        """Resolve ``target_controllable_name`` from override + scope.

        Resolution order:
        1. Explicit constructor override always wins.
        2. Auto-claim ``system_prompt`` when the controller's scope
           grants it as a writable controllable — the higher-leverage
           attack surface, and matches the paper's "single-component
           optimisation" framing more naturally than user-message.
        3. Otherwise leave ``None`` so the default user-message attack
           path runs (and ``system_prompt`` PreCalls are skipped).
        """
        if self._target_controllable_name_override is not None:
            return self._target_controllable_name_override
        for ctrl in controllables:
            if ctrl.name == _SYSTEM_PROMPT_NAME:
                logger.info(
                    "GEPA: auto-claiming write access to %r as the attack "
                    "channel (higher-leverage than user_message)",
                    _SYSTEM_PROMPT_NAME,
                )
                return _SYSTEM_PROMPT_NAME
        return None

    async def _reflect_next_candidate(self) -> bool:
        """Build the side-info dataset and stash a pending proposal.

        Returns whether the search may continue. ``False`` stops the task:
        without a fresh candidate the next run would re-send the identical
        prompt to the target at full cost, which looks like a legitimate
        multi-run search but makes no progress at all.

        A reflection call that keeps failing does *not* return ``False`` —
        it records the exception for ``_handle_run_start`` to raise, so the
        task is reported as an error rather than as a completed search.
        """
        assert self._reflector is not None and self._goal is not None

        parent = self._best_in_pool()
        if not parent.rolled_out:
            # Seed has not been rolled out yet — defer reflection,
            # next run will roll out the seed first.
            return True

        # Replay every recent rollout we have for the parent so the
        # reflection LM sees as much signal as we've already paid for.
        rollouts = list(parent.rollouts)

        try:
            result = await self._propose_with_retry(
                current_instruction=parent.prompt,
                rollouts=rollouts,
            )
        except (BudgetExhaustedError, ReflectionUnavailable) as exc:
            # Both end the task, and both must be visible in the recorded
            # stop_reason: the controller maps BudgetExhaustedError to
            # "budget_exhausted" and anything else to "error".
            logger.warning(
                "GEPA: reflection ended the task at attempt %d (%s)",
                self._attempt,
                type(exc).__name__,
                exc_info=True,
            )
            self._pending_failure = exc
            return True

        if result is None:
            # The reflection LM answered but proposed nothing parseable —
            # a legitimate attacker-model outcome (typically a refusal to
            # improve the attack), not an infrastructure failure. Re-roll
            # the parent so its rollout buffer turns over and the next
            # reflection sees different side-info, but bound it.
            self._consecutive_no_mutation += 1
            if (
                self._max_consecutive_no_mutation > 0
                and self._consecutive_no_mutation >= self._max_consecutive_no_mutation
            ):
                logger.warning(
                    "GEPA: reflection LM proposed no parseable mutation %d times "
                    "in a row after attempt %d; stopping instead of re-sending "
                    "the identical prompt for the remaining %d attempts",
                    self._consecutive_no_mutation,
                    self._attempt,
                    self._max_attempts - self._attempt,
                )
                return False
            return True

        self._consecutive_no_mutation = 0
        parent_idx = self._pool.index(parent)
        self._pending = _Candidate(
            prompt=result.new_instruction,
            parent_idx=parent_idx,
        )
        return True

    async def _propose_with_retry(
        self,
        *,
        current_instruction: str,
        rollouts: list[RolloutRecord],
    ) -> ReflectionResult | None:
        """Call the reflection LM, retrying a failing call a bounded number of times.

        Returns the proposal, or ``None`` when the LM answered but its
        output held no parseable instruction.

        Raises:
            BudgetExhaustedError: The attacker's cost cap is spent. Never
                retried (retrying is a cap escape) and never converted:
                the controller turns it into
                ``stop_reason="budget_exhausted"``.
            ReflectionUnavailable: Every attempt raised.
        """
        assert self._reflector is not None

        deadline = time.monotonic() + self._reflection_retry_deadline
        max_attempts = self._reflection_retries + 1
        last_error: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await self._reflector.propose(
                    current_instruction=current_instruction,
                    rollouts=rollouts,
                )
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "GEPA: reflection LM call failed (attempt %d/%d)",
                    attempt,
                    max_attempts,
                    exc_info=True,
                )
                if attempt == max_attempts:
                    break
                delay = _backoff_delay(attempt)
                if time.monotonic() + delay >= deadline:
                    # A slow failure (a provider timeout is minutes, not
                    # seconds) has already spent the retry window; sleeping
                    # on would risk the controller's task timeout, which
                    # discards the task outright.
                    logger.warning(
                        "GEPA: reflection retry deadline (%.1fs) reached after "
                        "attempt %d; not retrying further",
                        self._reflection_retry_deadline,
                        attempt,
                    )
                    break
                await asyncio.sleep(delay)

        assert last_error is not None
        raise ReflectionUnavailable(
            f"GEPA reflection LM unreachable: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


__all__ = ["GEPAOptimizer", "ReflectionUnavailable"]
