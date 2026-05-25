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
* Writable surfaces are auto-claimed by runtime priority: agentic
  read/tool-return PostCall surfaces first, then ``system_prompt``,
  then user-prompt-style channels. The explicit
  ``target_controllable_name`` constructor knob always wins.

Refer to ``ASSUMPTIONS.md`` for paper alignment and deliberate
departures.
"""

from __future__ import annotations

import logging
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
from superred.core.types.observable import ObservableValue

from gepa_optimizer.reflector import Reflector, RolloutRecord

logger = logging.getLogger(__name__)


_DEFAULT_RESPONSE_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {"response", "model_response", "assistant_response"}
)
_AGENTIC_OBSERVABLE_NAME_HINTS: tuple[str, ...] = (
    "agent_trace_message",
    "agent_trace_tool_response",
    "agent_trace_tool_call",
    "last_response",
    "output",
    "assistant",
    "response",
    "reply",
)
_AGENTIC_READ_PREFIXES: tuple[str, ...] = ("read__", "tool_call:")
_USER_PROMPT_NAMES: frozenset[str] = frozenset(
    {"user_prompt", "user_message", "query", "prompt"}
)
_TOOL_CATALOG_NAMES: frozenset[str] = frozenset(
    {
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_rewrite_doc",
        "tool_catalog_unregister",
    }
)
_MAX_AGENT_OBSERVATIONS = 5
_MAX_AGENT_OBSERVATION_CHARS = 2000

# Hardcoded system-prompt name (matches the chatbot target's naming).
# Used for two things: (1) the default-mode skip rule on the
# ``system_prompt`` ControllablePreCallEvent so the user-message
# channel can still claim the primary slot when neither an agentic
# read/tool surface nor writable ``system_prompt`` is selected, and
# (2) prompt-surface resolution.
_SYSTEM_PROMPT_NAME = "system_prompt"

# Per-candidate rollout history depth. Matches the GEPA paper's default
# minibatch size of 3, which is what the reflection LM expects to see
# in the side-info dataset.
_ROLLOUT_HISTORY_SIZE = 3


@dataclass(frozen=True)
class _InjectionSurface:
    """Resolved attack surface for the current target scope."""

    kind: str
    names: frozenset[str]

    @property
    def is_post_call(self) -> bool:
        return self.kind == "agentic_read"

    @property
    def is_pre_call(self) -> bool:
        return self.kind in {"system_prompt", "user_prompt", "explicit_pre"}

    @property
    def primary_name(self) -> str | None:
        return next(iter(self.names), None)

    def matches(self, controllable: Controllable) -> bool:
        name = controllable.name
        if self.kind == "agentic_read":
            return name in self.names or _is_agentic_read_name(name)
        if self.kind == "user_prompt":
            return name in self.names or _is_user_prompt_name(name)
        return name in self.names

    def to_record(self, actual_name: str | None = None) -> dict[str, str]:
        name = actual_name or self.primary_name or self.kind
        return {"type": self.kind, "name": name}


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
        reflection_temperature: Sampling temperature for the reflection
            LM. The paper uses high temperature for prompt diversity.
        response_observable_names: Names recognised as target replies on
            the trajectory (defaults to ``response``,
            ``model_response``, ``assistant_response`` — same set as
            Crescendo / GOAT).
        target_controllable_name: When set, the optimizer locks
            injection onto exactly the controllable with this name and
            ignores all others. Default ``None`` enables auto-claim:
            attack agentic read/tool-return PostCall surfaces first
            when present, otherwise attack ``system_prompt`` when
            writable, otherwise attack a user-prompt-style PreCall.
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
        target_controllable_name: str | None = None,
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
        self._injection_surface: _InjectionSurface | None = None

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
        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._injected_this_run: bool = False
        self._awaiting_post_call: bool = False
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None
        self._injected_surface_name: str | None = None
        self._injected_surface_type: str | None = None
        self._legitimate_tool_return: str | None = None
        self._agent_observations: list[str] = []

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
        self._target_observables = self._extract_static_observables(observables)
        self._injection_surface = self._resolve_injection_surface(controllables)
        self._target_controllable_name = self._resolve_target_controllable_name()
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
        self._current, self._current_is_fresh = self._select_current_candidate()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        surface = self._injection_surface
        if surface is not None and surface.is_post_call:
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if surface is not None and surface.is_pre_call:
            if not surface.matches(event.controllable):
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
        if surface is not None:
            self._injected_surface_type = surface.kind
            self._injected_surface_name = event.controllable.name
        else:
            self._injected_surface_type = "user_prompt"
            self._injected_surface_name = event.controllable.name
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current.prompt,
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        surface = self._injection_surface
        if (
            surface is not None
            and surface.is_post_call
            and surface.matches(event.controllable)
        ):
            if self._injected_this_run or self._current is None:
                return ControllableNoInjection(
                    event=event, controllable=event.controllable
                )
            self._injected_this_run = True
            self._injected_surface_type = surface.kind
            self._injected_surface_name = event.controllable.name
            self._legitimate_tool_return = event.answer
            self._last_injected_value = self._render_post_call_payload(
                event.answer,
            )
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=self._last_injected_value,
            )

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
                    injection_surface=self._rollout_surface_record(),
                    tool_return=self._legitimate_tool_return,
                    agent_observations=list(self._agent_observations) or None,
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
        await self._reflect_next_candidate()
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
        self._injected_surface_name = None
        self._injected_surface_type = None
        self._legitimate_tool_return = None
        self._agent_observations = []

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

    def _render_post_call_payload(self, legitimate_content: str) -> str:
        """Render an indirect prompt-injection payload for a read/tool return."""
        assert self._current is not None
        candidate = self._current.prompt
        if not legitimate_content.strip():
            return candidate
        return (
            f"{legitimate_content.rstrip()}\n\n"
            "[GEPA INDIRECT PROMPT INJECTION]\n"
            f"{candidate}"
        )

    def _rollout_surface_record(self) -> dict[str, str] | None:
        if self._injected_surface_type is not None:
            name = self._injected_surface_name or self._injected_surface_type
            return {"type": self._injected_surface_type, "name": name}
        if self._injection_surface is not None:
            return self._injection_surface.to_record()
        return None

    def _add_agent_observation(self, text: str) -> None:
        if len(text) > _MAX_AGENT_OBSERVATION_CHARS:
            text = f"{text[: _MAX_AGENT_OBSERVATION_CHARS].rstrip()}..."
        self._agent_observations.append(text)
        if len(self._agent_observations) > _MAX_AGENT_OBSERVATIONS:
            self._agent_observations = self._agent_observations[
                -_MAX_AGENT_OBSERVATIONS:
            ]

    def _read_response_from_trajectory(self) -> str | None:
        if self.current_trajectory is None:
            return None
        latest: str | None = None
        for item in self.current_trajectory.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            content = _stringify_content(item.content)
            if _is_agentic_observation_name(name) and content.strip():
                self._add_agent_observation(f"{name}: {content.strip()}")
            if (
                name in self._response_observable_names
                or name.lower() in self._response_observable_names
                or _is_response_like_observable_name(name)
            ):
                if content:
                    latest = content
        return latest

    def _resolve_injection_surface(
        self, controllables: list[Controllable],
    ) -> _InjectionSurface | None:
        """Resolve the highest-leverage writable surface in scope.

        Explicit ``target_controllable_name`` wins. Otherwise prefer
        agentic read/tool-return PostCall surfaces, then writable
        ``system_prompt``, then user-prompt-style PreCall surfaces.
        Falling back to ``None`` preserves the legacy first-non-system
        PreCall lock for chatbot targets with unusual controllable names.
        """
        names = [ctrl.name for ctrl in controllables]
        override = self._target_controllable_name_override
        if override is not None:
            if _is_agentic_read_name(override):
                return _InjectionSurface("agentic_read", frozenset({override}))
            return _InjectionSurface("explicit_pre", frozenset({override}))

        read_names = frozenset(name for name in names if _is_agentic_read_name(name))
        if read_names:
            logger.info(
                "GEPA: auto-claiming agentic read/tool return surface(s): %s",
                ", ".join(sorted(read_names)),
            )
            return _InjectionSurface("agentic_read", read_names)

        if _SYSTEM_PROMPT_NAME in names:
            logger.info(
                "GEPA: auto-claiming write access to %r as the attack "
                "channel (higher-leverage than user_message)",
                _SYSTEM_PROMPT_NAME,
            )
            return _InjectionSurface(
                "system_prompt", frozenset({_SYSTEM_PROMPT_NAME})
            )

        user_names = frozenset(name for name in names if _is_user_prompt_name(name))
        if user_names:
            return _InjectionSurface("user_prompt", user_names)
        return None

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

    def _resolve_target_controllable_name(self) -> str | None:
        """Resolve ``target_controllable_name`` from override + scope.

        Kept for backward-compatible tests / inspection; the actual
        event routing uses ``_injection_surface``. For agentic read
        surfaces and default user-prompt surfaces this remains ``None``.
        """
        if self._target_controllable_name_override is not None:
            return self._target_controllable_name_override
        if (
            self._injection_surface is not None
            and self._injection_surface.kind == "system_prompt"
        ):
            return _SYSTEM_PROMPT_NAME
        return None

    async def _reflect_next_candidate(self) -> None:
        """Build the side-info dataset and stash a pending proposal."""
        assert self._reflector is not None and self._goal is not None

        parent = self._best_in_pool()
        if not parent.rolled_out:
            # Seed has not been rolled out yet — defer reflection,
            # next run will roll out the seed first.
            return

        # Replay every recent rollout we have for the parent so the
        # reflection LM sees as much signal as we've already paid for.
        rollouts = list(parent.rollouts)

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


def _is_agentic_read_name(name: str) -> bool:
    return name.startswith(_AGENTIC_READ_PREFIXES)


def _is_user_prompt_name(name: str) -> bool:
    normalized = name.lower()
    return normalized in _USER_PROMPT_NAMES or "user" in normalized


def _is_tool_catalog_name(name: str) -> bool:
    return name in _TOOL_CATALOG_NAMES


def _is_agentic_observation_name(name: str) -> bool:
    normalized = name.lower()
    return any(hint in normalized for hint in _AGENTIC_OBSERVABLE_NAME_HINTS)


def _is_response_like_observable_name(name: str) -> bool:
    normalized = name.lower()
    return any(
        hint in normalized
        for hint in ("response", "assistant", "reply", "output", "last_response")
    )


def _stringify_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


__all__ = ["GEPAOptimizer"]
