"""GEPA-Agentic: reflective prompt evolution for SuperRed agent targets."""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

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

from gepa_agentic_optimizer.reflector import Reflector, RolloutRecord

logger = logging.getLogger(__name__)

_DEFAULT_RESPONSE_OBSERVABLE_NAMES: frozenset[str] = frozenset(
    {"response", "model_response", "assistant_response"}
)
_SYSTEM_PROMPT_NAME = "system_prompt"
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
_TOOL_CATALOG_REWRITE_DOC = "tool_catalog_rewrite_doc"
_CONTENT_PREFIXES: tuple[str, ...] = (
    "read__",
    "read:",
    "tool_call:",
    "tool_call__",
    "tool:",
    "skill_call:",
    "skill_call__",
    "skill:",
)
_CATALOG_TOOL_HINTS: tuple[str, ...] = (
    "browser",
    "content",
    "document",
    "fetch",
    "html",
    "knowledge",
    "memory",
    "page",
    "read",
    "retrieve",
    "search",
    "tool",
    "web",
)
_CONTENT_SURFACE_HINTS: tuple[str, ...] = (
    "answer carries",
    "content",
    "document",
    "external",
    "html",
    "knowledge",
    "memory",
    "observation",
    "page",
    "per-call event",
    "per-read injection",
    "rag",
    "read",
    "retrieval",
    "result",
    "return",
    "tool response",
    "web",
)
_HTML_CONTENT_HINTS: tuple[str, ...] = (
    "<html",
    "<body",
    "<input",
    "<form",
    "aria-label",
    "href=",
)
_AGENT_OBSERVABLE_HINTS: tuple[str, ...] = (
    "agent_trace_message",
    "agent_trace_tool_response",
    "agent_trace_tool_call",
    "last_response",
    "output",
    "assistant",
    "response",
    "reply",
)
_ROLLOUT_HISTORY_SIZE = 3
_MAX_AGENT_OBSERVATIONS = 8
_MAX_AGENT_OBSERVATION_CHARS = 2000
_DEFAULT_MAX_POOL_SIZE = 8


@dataclass
class _SurfaceStats:
    attempts: int = 0
    misses: int = 0
    best_score: float = 0.0


@dataclass(frozen=True)
class _SurfaceClassification:
    surface_type: str
    confidence: float
    reason: str

    @property
    def is_content(self) -> bool:
        return self.surface_type == "content_postcall"


@dataclass
class _Candidate:
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
        scored = [r.score for r in self.rollouts if r.score is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)


class GEPAAgenticOptimizer(Optimizer):
    """GEPA variant for agent targets with runtime surface selection.

    The chatbot GEPA implementation remains in ``optimizers/gepa``. This
    optimizer keeps the same reflective candidate loop but targets agentic
    content surfaces, especially AgentDojo-style ``read__...`` and
    inspect-agent-style ``tool:<name>`` PostCall events. Surface names are
    hints, not the contract; scoped metadata and runtime content are also used.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 20,
        reflection_temperature: float = 1.0,
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        content_controllable_names: Iterable[str] | None = None,
        max_no_signal_runs: int = 0,
        max_content_injections_per_run: int = 3,
        max_pool_size: int = _DEFAULT_MAX_POOL_SIZE,
        use_tool_catalog_rewrite_when_available: bool = True,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_content_injections_per_run < 1:
            raise ValueError("max_content_injections_per_run must be at least 1")
        if max_pool_size < 1:
            raise ValueError("max_pool_size must be at least 1")
        self._max_attempts = max_attempts
        self._reflection_temperature = reflection_temperature
        self._max_content_injections_per_run = max_content_injections_per_run
        self._max_pool_size = max_pool_size
        self._use_tool_catalog_rewrite_when_available = (
            use_tool_catalog_rewrite_when_available
        )
        self._response_names_explicit = response_observable_names is not None
        self._response_observable_names: frozenset[str] = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        explicit_content = tuple(content_controllable_names or ())
        self._explicit_content_names = {name for name in explicit_content} | {
            name.lower() for name in explicit_content
        }
        self._target_controllable_name = target_controllable_name
        self._max_no_signal_runs = max(0, max_no_signal_runs)

        self._goal: Goal | None = None
        self._reflector: Reflector | None = None
        self._target_observables: dict[str, str] | None = None
        self._controllables: dict[str, Controllable] = {}
        self._content_surface_names: list[str] = []
        self._tool_catalog: list[dict[str, Any]] = []
        self._surface_stats: dict[str, _SurfaceStats] = {}

        self._pool: list[_Candidate] = []
        self._pending: _Candidate | None = None
        self._attempt = 0
        self._succeeded = False
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False

        self._current: _Candidate | None = None
        self._current_is_fresh = False
        self._planned_content_surface: str | None = None
        self._planned_content_surfaces: list[str] = []
        self._injected_this_run = False
        self._content_injection_count = 0
        self._catalog_rewrite_injected = False
        self._selected_surface: dict[str, str] | None = None
        self._injected_surfaces: list[dict[str, str]] = []
        self._selection_reason = ""
        self._observed_surfaces: list[dict[str, str]] = []
        self._observed_surface_names: set[str] = set()
        self._legitimate_tool_return: str | None = None
        self._pending_post_answer: str | None = None
        self._agent_observations: list[str] = []
        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None

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
        self._controllables = {ctrl.name: ctrl for ctrl in controllables}
        self._content_surface_names = self._discover_content_surfaces(controllables)
        self._tool_catalog = self._extract_tool_catalog(observables)
        self._surface_stats = {
            name: self._surface_stats.get(name, _SurfaceStats())
            for name in self._content_surface_names
        }
        self._pool = [_Candidate(prompt=goal.description)]
        self._pending = None
        self._attempt = 0
        self._succeeded = False
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

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

    def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        self._current, self._current_is_fresh = self._select_current_candidate()
        self._planned_content_surfaces = self._choose_content_surfaces()
        self._planned_content_surface = (
            self._planned_content_surfaces[0] if self._planned_content_surfaces else None
        )
        if self._planned_content_surface is not None:
            self._selection_reason = (
                "planned compatible content PostCall surfaces: "
                + ", ".join(repr(name) for name in self._planned_content_surfaces)
            )
        elif self._content_surface_names:
            self._selection_reason = "content surfaces known but none selected"
        else:
            self._selection_reason = (
                "no agentic content surface classified; prompt fallback allowed"
            )
        return EventResponse(event=event)

    def _handle_pre_call(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        self._record_observed_surface(event.controllable, "pre")
        if self._current is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        name = event.controllable.name
        if (
            self._target_controllable_name is None
            and name == _TOOL_CATALOG_REWRITE_DOC
        ):
            return self._maybe_inject_tool_catalog_rewrite(event)

        if self._target_controllable_name is not None:
            if name != self._target_controllable_name:
                return ControllableNoInjection(
                    event=event, controllable=event.controllable
                )
        elif self._planned_content_surface is not None:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        elif name in _TOOL_CATALOG_NAMES:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        elif name == _SYSTEM_PROMPT_NAME or self._is_user_prompt(name):
            pass
        else:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._primary_pre_controllable is None:
            self._primary_pre_controllable = event.controllable
        elif event.controllable != self._primary_pre_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._last_pre_request = event.request
        self._last_injected_value = self._current.prompt
        surface_type = "system_prompt" if name == _SYSTEM_PROMPT_NAME else "user_prompt"
        surface = {
            "name": name,
            "type": surface_type,
            "event_kind": "pre",
            "mode": "explicit" if self._target_controllable_name else "fallback",
        }
        self._record_injection_surface(surface)
        if not self._selection_reason:
            self._selection_reason = "prompt fallback selected"
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current.prompt,
        )

    def _handle_post_call(
        self,
        event: ControllablePostCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        classification = self._record_observed_surface(
            event.controllable, "post", content=event.answer
        )
        if self._current is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        name = event.controllable.name
        should_inject = False
        mode = "runtime_compatible"
        if self._target_controllable_name is not None:
            should_inject = name == self._target_controllable_name
            mode = "explicit"
        elif self._content_injection_count >= self._max_content_injections_per_run:
            should_inject = False
        elif name in self._planned_content_surfaces:
            should_inject = True
            mode = "planned"
        elif classification.is_content:
            should_inject = True
            mode = "runtime_discovery"
            self._selection_reason = (
                f"runtime-discovered compatible content PostCall surface {name!r}"
            )

        if not should_inject:
            self._maybe_pair_post_call_response(event)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._injected_this_run = True
        self._content_injection_count += 1
        self._legitimate_tool_return = event.answer
        injected_value = self._render_content_payload(event.answer)
        self._last_injected_value = injected_value
        surface = {
            "name": name,
            "type": classification.surface_type,
            "event_kind": "post",
            "mode": mode,
            "classification_reason": classification.reason,
        }
        self._record_injection_surface(surface)
        self._surface_stats.setdefault(name, _SurfaceStats()).attempts += 1
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=injected_value,
        )

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        response = self._read_response_from_trajectory()
        if response is None:
            response = self._pending_post_answer

        score: float | None = None
        rationale = ""
        evaluation = event.evaluation
        if evaluation is not None:
            score = evaluation.primary_score.value
            rationale = evaluation.rationale

        signal_seen = (
            response is not None
            or evaluation is not None
            or bool(self._agent_observations)
        )
        if not self._injected_this_run and not signal_seen:
            self._update_surface_stats(score)
            self._consecutive_no_signal_runs += 1
            if (
                self._max_no_signal_runs > 0
                and self._consecutive_no_signal_runs >= self._max_no_signal_runs
            ):
                self._stop_due_to_no_signal = True
            return RunEndResponse(event=event, done=self._stop_due_to_no_signal)

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
                    selected_surface=self._selected_surface_with_all_injections(),
                    observed_surfaces=list(self._observed_surfaces) or None,
                    tool_return=self._legitimate_tool_return,
                    agent_observations=list(self._agent_observations) or None,
                    selection_reason=self._selection_reason,
                )
            )
            if self._current_is_fresh and self._should_accept_fresh_candidate(
                self._current, evaluation
            ):
                self._pool.append(self._current)
                self._prune_pool()

        self._update_surface_stats(score)
        if signal_seen:
            self._consecutive_no_signal_runs = 0
        else:
            self._consecutive_no_signal_runs += 1
            if (
                self._max_no_signal_runs > 0
                and self._consecutive_no_signal_runs >= self._max_no_signal_runs
            ):
                self._stop_due_to_no_signal = True

        if evaluation is not None and evaluation.success:
            self._succeeded = True

        self._attempt += 1
        if (
            self._succeeded
            or self._stop_due_to_no_signal
            or self._attempt >= self._max_attempts
        ):
            return RunEndResponse(event=event, done=True)

        await self._reflect_next_candidate()
        return RunEndResponse(event=event, done=False)

    def _reset_run_state(self) -> None:
        self._current = None
        self._current_is_fresh = False
        self._planned_content_surface = None
        self._planned_content_surfaces = []
        self._injected_this_run = False
        self._content_injection_count = 0
        self._catalog_rewrite_injected = False
        self._selected_surface = None
        self._injected_surfaces = []
        self._selection_reason = ""
        self._observed_surfaces = []
        self._observed_surface_names = set()
        self._legitimate_tool_return = None
        self._pending_post_answer = None
        self._agent_observations = []
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._last_pre_request = None
        self._last_injected_value = None

    def _select_current_candidate(self) -> tuple[_Candidate, bool]:
        if self._pending is not None:
            picked = self._pending
            self._pending = None
            return picked, True
        return self._best_in_pool(), False

    def _best_in_pool(self) -> _Candidate:
        assert self._pool
        best = self._pool[0]
        for candidate in self._pool[1:]:
            if candidate.effective_score >= best.effective_score:
                best = candidate
        return best

    def _should_accept_fresh_candidate(
        self,
        candidate: _Candidate,
        evaluation: Any,
    ) -> bool:
        """Gate pool growth like GEPA's strict-improvement acceptance.

        Upstream GEPA accepts a reflected proposal only when minibatch score
        improves. SuperRed drives one rollout at a time, so the local analogue is
        strict improvement over the parent candidate's effective score. No-score
        threat models may still execute a pending candidate once, but do not grow
        the persistent candidate pool without an objective signal.
        """
        if evaluation is not None and evaluation.success:
            return True
        if candidate.score is None or candidate.parent_idx is None:
            return False
        if not 0 <= candidate.parent_idx < len(self._pool):
            return False
        parent_score = self._pool[candidate.parent_idx].effective_score
        return candidate.effective_score > parent_score

    def _prune_pool(self) -> None:
        if len(self._pool) <= self._max_pool_size:
            return
        seed = self._pool[0]
        rest = sorted(
            self._pool[1:],
            key=lambda candidate: candidate.effective_score,
            reverse=True,
        )
        self._pool = [seed, *rest[: self._max_pool_size - 1]]

    def _choose_content_surfaces(self) -> list[str]:
        if self._target_controllable_name is not None:
            ctrl = self._controllables.get(self._target_controllable_name)
            if ctrl is not None and self._classify_surface(ctrl, "post").is_content:
                return [self._target_controllable_name]
            return []
        if not self._content_surface_names:
            return []

        def key(name: str) -> tuple[int, int, float, str]:
            stats = self._surface_stats.setdefault(name, _SurfaceStats())
            return (stats.attempts, stats.misses, -stats.best_score, name)

        return sorted(self._content_surface_names, key=key)

    def _render_content_payload(self, legitimate_content: str) -> str:
        assert self._current is not None
        if not legitimate_content.strip():
            return self._current.prompt
        return f"{legitimate_content.rstrip()}\n\n{self._current.prompt}"

    def _record_injection_surface(self, surface: dict[str, str]) -> None:
        self._injected_this_run = True
        if self._selected_surface is None:
            self._selected_surface = dict(surface)
        self._injected_surfaces.append(dict(surface))

    def _selected_surface_with_all_injections(self) -> dict[str, Any] | None:
        if self._selected_surface is None:
            return None
        out: dict[str, Any] = dict(self._selected_surface)
        if len(self._injected_surfaces) > 1:
            out["all_injected_surfaces"] = [dict(item) for item in self._injected_surfaces]
        return out

    def _record_observed_surface(
        self,
        controllable: Controllable,
        event_kind: str,
        content: str = "",
    ) -> _SurfaceClassification:
        classification = self._classify_surface(controllable, event_kind, content=content)
        key = f"{event_kind}:{controllable.name}"
        if key not in self._observed_surface_names:
            self._observed_surface_names.add(key)
            self._observed_surfaces.append(
                {
                    "name": controllable.name,
                    "event_kind": event_kind,
                    "type": classification.surface_type,
                    "confidence": f"{classification.confidence:.2f}",
                    "reason": classification.reason,
                }
            )
        if event_kind == "post" and classification.is_content:
            if controllable.name not in self._surface_stats:
                self._surface_stats[controllable.name] = _SurfaceStats()
            if controllable.name not in self._content_surface_names:
                self._content_surface_names.append(controllable.name)
                self._content_surface_names.sort()
        return classification

    def _classify_surface(
        self,
        controllable: Controllable,
        event_kind: str,
        content: str = "",
    ) -> _SurfaceClassification:
        name = controllable.name
        normalized = name.lower()
        if name == _SYSTEM_PROMPT_NAME:
            return _SurfaceClassification("system_prompt", 1.0, "canonical name")
        if name in _TOOL_CATALOG_NAMES:
            return _SurfaceClassification("tool_catalog", 1.0, "canonical catalog name")
        if self._is_user_prompt(name):
            return _SurfaceClassification("user_prompt", 0.9, "user prompt name")
        if event_kind == "post":
            if normalized in self._explicit_content_names or name in (
                self._explicit_content_names
            ):
                return _SurfaceClassification(
                    "content_postcall", 1.0, "explicit content controllable"
                )
            if any(name.startswith(prefix) for prefix in _CONTENT_PREFIXES):
                return _SurfaceClassification(
                    "content_postcall",
                    1.0,
                    "agent content/tool-return prefix",
                )
            haystack = (
                f"{controllable.name} {controllable.description} "
                f"{controllable.value_type}".lower()
            )
            if any(hint in haystack for hint in _CONTENT_SURFACE_HINTS):
                return _SurfaceClassification(
                    "content_postcall", 0.75, "content-like controllable metadata"
                )
            if _looks_like_environment_content(content):
                return _SurfaceClassification(
                    "content_postcall", 0.7, "runtime content looks like HTML/environment data"
                )
        return _SurfaceClassification("unknown", 0.0, "no content/prompt signal")

    def _discover_content_surfaces(
        self,
        controllables: list[Controllable],
    ) -> list[str]:
        out = [
            ctrl.name
            for ctrl in controllables
            if self._classify_surface(ctrl, "post").is_content
        ]
        return sorted(dict.fromkeys(out))

    def _maybe_inject_tool_catalog_rewrite(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        if (
            not self._use_tool_catalog_rewrite_when_available
            or self._catalog_rewrite_injected
            or self._current is None
        ):
            return ControllableNoInjection(event=event, controllable=event.controllable)
        tool = self._select_catalog_tool()
        if tool is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._catalog_rewrite_injected = True
        payload = {
            "name": tool["name"],
            "description": self._catalog_rewrite_description(tool),
        }
        surface = {
            "name": event.controllable.name,
            "type": "tool_catalog",
            "event_kind": "pre",
            "mode": "catalog_rewrite",
            "classification_reason": "visible tool catalog identifies content-like tool",
        }
        self._record_injection_surface(surface)
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=json.dumps(payload, ensure_ascii=False),
        )

    def _select_catalog_tool(self) -> dict[str, str] | None:
        for entry in self._tool_catalog:
            haystack = f"{entry.get('name', '')} {entry.get('description', '')}".lower()
            if any(hint in haystack for hint in _CATALOG_TOOL_HINTS):
                return entry
        return self._tool_catalog[0] if self._tool_catalog else None

    def _catalog_rewrite_description(self, tool: dict[str, str]) -> str:
        assert self._current is not None
        base = tool.get("description", "").strip()
        if not base:
            base = f"Use {tool['name']} to retrieve task-relevant external context."
        return f"{base}\n\n{self._current.prompt}"

    def _maybe_pair_post_call_response(self, event: ControllablePostCallEvent) -> None:
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
                return
            self._primary_post_controllable = event.controllable
        elif event.controllable != self._primary_post_controllable:
            return
        self._pending_post_answer = event.answer

    def _update_surface_stats(self, score: float | None) -> None:
        injected_content_names = {
            surface["name"]
            for surface in self._injected_surfaces
            if surface.get("type") == "content_postcall"
        }
        for name in self._planned_content_surfaces:
            if name not in injected_content_names:
                self._surface_stats.setdefault(name, _SurfaceStats()).misses += 1
        for name in injected_content_names:
            stats = self._surface_stats.setdefault(name, _SurfaceStats())
            if score is not None:
                stats.best_score = max(stats.best_score, score)

    def _read_response_from_trajectory(self) -> str | None:
        if self.current_trajectory is None:
            return None
        latest: str | None = None
        for item in self.current_trajectory.drain():
            if isinstance(item, ObservableEvent):
                name = item.observable.name
                content = self._stringify_content(item.content)
                if _is_agent_observable(name) and content.strip():
                    self._add_agent_observation(f"{name}: {content.strip()}")
                normalized = name.lower()
                name_matches_response = (
                    name in self._response_observable_names
                    or normalized in self._response_observable_names
                )
                if not self._response_names_explicit:
                    name_matches_response = name_matches_response or (
                        _is_response_like_observable(name, content)
                    )
                if name_matches_response and content:
                    latest = content
                continue
            if isinstance(item, ControllablePreCallEvent):
                self._add_agent_observation(
                    f"controllable_pre:{item.controllable.name}: "
                    f"{self._stringify_content(item.request)}"
                )
                continue
            if isinstance(item, ControllablePostCallEvent):
                content = self._stringify_content(item.answer)
                self._add_agent_observation(
                    f"controllable_post:{item.controllable.name}: {content}"
                )
                if content and not self._response_names_explicit:
                    latest = content
        return latest

    def _add_agent_observation(self, text: str) -> None:
        if len(text) > _MAX_AGENT_OBSERVATION_CHARS:
            text = f"{text[: _MAX_AGENT_OBSERVATION_CHARS].rstrip()}..."
        self._agent_observations.append(text)
        if len(self._agent_observations) > _MAX_AGENT_OBSERVATIONS:
            self._agent_observations = self._agent_observations[
                -_MAX_AGENT_OBSERVATIONS:
            ]

    @staticmethod
    def _extract_static_observables(
        observables: list[ObservableValue],
    ) -> dict[str, str] | None:
        out: dict[str, str] = {}
        for value in observables:
            content = _stringify_value(value.content)
            if content.strip():
                out[value.observable.name] = content
        return out or None

    @staticmethod
    def _extract_tool_catalog(
        observables: list[ObservableValue],
    ) -> list[dict[str, str]]:
        for value in observables:
            if "tool" not in value.observable.name.lower():
                continue
            content = value.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    continue
            entries = list(_iter_tool_entries(content))
            if entries:
                return entries
        return []

    @staticmethod
    def _is_user_prompt(name: str) -> bool:
        normalized = name.lower()
        return normalized in _USER_PROMPT_NAMES or "user" in normalized

    @staticmethod
    def _stringify_content(content: Any) -> str:
        return _stringify_value(content)

    async def _reflect_next_candidate(self) -> None:
        assert self._reflector is not None
        parent = self._best_in_pool()
        if not parent.rolled_out:
            return
        try:
            result = await self._reflector.propose(
                current_instruction=parent.prompt,
                rollouts=list(parent.rollouts),
            )
        except Exception:
            logger.warning(
                "GEPA-Agentic: reflection LM call failed; skipping mutation",
                exc_info=True,
            )
            return
        if result is None:
            return
        self._pending = _Candidate(
            prompt=result.new_instruction,
            parent_idx=self._pool.index(parent),
        )


def _is_agent_observable(name: str) -> bool:
    normalized = name.lower()
    return any(hint in normalized for hint in _AGENT_OBSERVABLE_HINTS)


def _is_response_like_observable(name: str, content: str = "") -> bool:
    normalized = name.lower()
    if any(
        hint in normalized
        for hint in ("response", "assistant", "reply", "output", "last_response")
    ):
        return True
    return "agent_trace_message" in normalized and "assistant" in content.lower()


def _looks_like_environment_content(content: str) -> bool:
    normalized = content.lower()
    return any(hint in normalized for hint in _HTML_CONTENT_HINTS)


def _iter_tool_entries(content: Any) -> Iterable[dict[str, str]]:
    if isinstance(content, list):
        for item in content:
            if isinstance(item, Mapping):
                name = item.get("name")
                if isinstance(name, str) and name.strip():
                    yield {
                        "name": name.strip(),
                        "description": _stringify_value(item.get("description")).strip(),
                    }
    elif isinstance(content, Mapping):
        for name, description in content.items():
            if isinstance(name, str) and name.strip():
                yield {
                    "name": name.strip(),
                    "description": _stringify_value(description).strip(),
                }


def _stringify_value(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(content)


__all__ = ["GEPAAgenticOptimizer"]
