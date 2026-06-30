"""The MUZZLE optimizer: an adaptive agentic indirect-prompt-injection attack.

This is the only framework-facing class. It collapses MUZZLE's AutoGen multi-agent graph
(arXiv:2602.09222, gsiros/muzzle SHA ed611c0) into a single :class:`Optimizer` running a
three-phase state machine over the controller's multi-run loop, with one fixed ``self.llm``
playing every red-team role through the byte-identical per-role prompts.

Phases (advanced only at ``RunEndEvent``; one injection per run via the ``_selected`` latch):

* **OBSERVE** (upstream BENIGN run): inject nothing; reconstruct the victim transcript,
  Summarize it into a playbook, Graft the playbook into ranked vessels (in-scope content
  surfaces), Prompt the playbook+goal into a one-sentence instruction. No vessel -> done
  (passthrough baseline).
* **PROBE** (upstream PROBE + skip_element): plant ``[INSTRUCTION_PLACEHOLDER]`` at the
  current vessel. If it did not surface (the vessel never fired), advance to the next vessel
  WITHOUT scoring; once the vessels are exhausted, stop. If it surfaced, localize the PAIR
  transcript template and the victim system prompt, then move to ATTACK.
* **ATTACK** (upstream PAIR + ACTUAL + Judge + reflection): run the offline PAIR loop
  (``self.llm`` surrogate) to craft the payload, inject it in one real run. The SecurityClaim
  (``RunEndEvent.evaluation.success``) is authoritative -> done on success. Otherwise the
  Judge attributes the failure: ``instruction`` (payload delivered but ignored) -> refine the
  payload on the same vessel; ``UI``/``unclear`` (payload not delivered) -> next vessel.

Budget: every ``self.llm`` call may raise ``BudgetExhaustedError``; it is allowed to
propagate so the controller records ``budget_exhausted``. See ASSUMPTIONS.md.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from muzzle_optimizer.constants import PLACEHOLDER_ALIAS, PLACEHOLDER_INSTR
from muzzle_optimizer.grafter import Grafter
from muzzle_optimizer.judge import Judge, contains_injected_content
from muzzle_optimizer.pair.bridge import PairBridge
from muzzle_optimizer.prompter import Prompter
from muzzle_optimizer.summarizer import Summarizer
from muzzle_optimizer.templates import render_injection
from muzzle_optimizer.transcript import (
    PlaceholderNotFoundError,
    build_transcript,
    extract_placeholder_template,
    find_victim_system_prompt,
    recover_action_url,
    recover_user_goal,
)
from muzzle_optimizer.vessels import (
    SYSTEM_PROMPT_NAME,
    Vessel,
    build_vessels,
    is_content_surface,
    is_user_prompt_surface,
)

logger = logging.getLogger(__name__)

_OBSERVE = "OBSERVE"
_PROBE = "PROBE"
_ATTACK = "ATTACK"

_PLACEHOLDER_TOKENS = (PLACEHOLDER_INSTR, PLACEHOLDER_ALIAS)
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset(
    {"response", "model_response", "assistant_response", "agent_response"}
)
_ATTRIBUTIONS = frozenset({"UI", "instruction", "unclear"})


class MuzzleOptimizer(Optimizer):
    """Adaptive agentic indirect-prompt-injection optimizer (MUZZLE, arXiv:2602.09222).

    Zero required constructor arguments so an ``OptimizerFactory`` can build a fresh instance
    per task; all scenario specifics are inferred in :meth:`initialize`.

    Args:
        max_attempts: Maximum scored ATTACK runs before stopping (the controller's
            ``max_runs_per_task`` is the hard backstop, covering OBSERVE + PROBE runs too).
        pair_n_streams: PAIR parallel streams (budget-aware soft cap; upstream is 5).
        pair_n_iterations: PAIR refinement iterations (budget-aware soft cap; upstream is 30).
        pair_keep_last_n: PAIR attacker history window.
        pair_judge_method: PAIR judge method (``"gpt"`` mirrors MUZZLE's GPT judge; ``"gcg"``
            is the cheaper refusal-dictionary judge).
        response_observable_names: Observable names whose content is the victim's reply.
        target_controllable_name: Pin the vessel to a single controllable name (ablations).
        extra_content_names: Extra controllable names to force-classify as content surfaces
            (for targets with opaque naming).
        use_injection_template: Wrap the ATTACK payload in the byte-identical upstream
            injection template (off by default; the faithful main path plants the bare PAIR
            payload, see ASSUMPTIONS.md deviation 2).
        max_tries: JSON-retry budget for each brain helper (upstream default 5).
    """

    def __init__(
        self,
        *,
        max_attempts: int = 8,
        pair_n_streams: int = 3,
        pair_n_iterations: int = 4,
        pair_keep_last_n: int = 4,
        pair_judge_method: str = "gpt",
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        extra_content_names: Iterable[str] | None = None,
        use_injection_template: bool = False,
        max_tries: int = 5,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_tries < 1:
            raise ValueError("max_tries must be at least 1")
        self._max_attempts = max_attempts
        self._pair_n_streams = pair_n_streams
        self._pair_n_iterations = pair_n_iterations
        self._pair_keep_last_n = pair_keep_last_n
        self._pair_judge_method = pair_judge_method
        self._response_names = (
            frozenset(response_observable_names)
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._target_controllable_name = target_controllable_name
        self._extra_content_names = frozenset(extra_content_names or ())
        self._use_injection_template = use_injection_template
        self._max_tries = max_tries

        # Brain helpers + PAIR bridge (built in initialize, after self.llm is set).
        self._goal: Goal | None = None
        self._controllables: list[Controllable] = []
        self._observables: list[ObservableValue] = []
        self._summarizer: Summarizer | None = None
        self._grafter: Grafter | None = None
        self._prompter: Prompter | None = None
        self._judge: Judge | None = None
        self._pair: PairBridge | None = None

        # Cross-run state.
        self._phase = _OBSERVE
        self._vessels: list[Vessel] = []
        self._vessel_idx = 0
        self._instruction = ""
        self._victim_system_prompt: str | None = None
        self._user_goal: str | None = None
        self._action_url: str | None = None
        self._pair_template: str | None = None
        self._current_inject_value: str | None = None
        self._reflection: dict[str, Any] | None = None
        self._attempt_index = 0
        self._succeeded = False
        self._exhausted = False
        self._can_inject = False

        # Per-run state.
        self._injected = False
        self._selected = False

    # ------------------------------------------------------------------ lifecycle

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._controllables = list(controllables)
        self._observables = list(observables)
        self._summarizer = Summarizer(self.llm, max_tries=self._max_tries)
        self._grafter = Grafter(self.llm, max_tries=self._max_tries)
        self._prompter = Prompter(self.llm, max_tries=self._max_tries)
        self._judge = Judge(self.llm, max_tries=self._max_tries)
        self._pair = PairBridge(
            self.llm,
            n_streams=self._pair_n_streams,
            n_iterations=self._pair_n_iterations,
            keep_last_n=self._pair_keep_last_n,
            judge_method=self._pair_judge_method,
        )
        self._can_inject = any(
            is_content_surface(ctrl, extra_names=self._extra_content_names)
            or is_user_prompt_surface(ctrl)
            or ctrl.name == SYSTEM_PROMPT_NAME
            for ctrl in controllables
        )

        self._phase = _OBSERVE
        self._vessels = []
        self._vessel_idx = 0
        self._instruction = ""
        self._victim_system_prompt = None
        self._user_goal = None
        self._action_url = None
        self._pair_template = None
        self._current_inject_value = None
        self._reflection = None
        self._attempt_index = 0
        self._succeeded = False
        self._exhausted = False
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

    def _reset_run_state(self) -> None:
        self._injected = False
        self._selected = False

    # ------------------------------------------------------------------ dispatch

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_inject(event, is_post=False, base=event.request)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_inject(event, is_post=True, base=event.answer)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        if self._is_done():
            return EventResponse(event=event)
        # ATTACK runs craft the payload offline (self.llm) just before the real run.
        if self._phase == _ATTACK and self._current_inject_value is None:
            await self._run_pair()
        return EventResponse(event=event)

    def _handle_inject(
        self,
        event: ControllablePreCallEvent | ControllablePostCallEvent,
        *,
        is_post: bool,
        base: str,
    ) -> ControllableInjection | ControllableNoInjection:
        if self._phase == _OBSERVE or self._selected or self._current_inject_value is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        vessel = self._current_vessel()
        if vessel is None or not self._event_matches_vessel(event.controllable, vessel, is_post):
            return ControllableNoInjection(event=event, controllable=event.controllable)
        self._selected = True
        self._injected = True
        value = self._compose(base, self._current_inject_value)
        return ControllableInjection(event=event, controllable=event.controllable, value=value)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        if self._phase == _OBSERVE:
            return await self._end_observe(event)
        if self._phase == _PROBE:
            return await self._end_probe(event)
        return await self._end_attack(event)

    # ------------------------------------------------------------------ phases

    async def _end_observe(self, event: RunEndEvent) -> RunEndResponse:
        assert self._summarizer is not None
        assert self._grafter is not None
        assert self._prompter is not None
        assert self._goal is not None
        items = self._snapshot()
        transcript = build_transcript(items, response_observable_names=self._response_names)
        playbook = await self._summarizer.summarize(transcript)
        graft = await self._grafter.graft(playbook)
        candidates = graft.get("candidates", []) if isinstance(graft, dict) else []
        if not isinstance(candidates, list):
            candidates = []
        vessels = build_vessels(
            self._controllables,
            grafter_candidates=candidates,
            extra_content_names=self._extra_content_names,
        )
        if self._target_controllable_name is not None:
            vessels = [v for v in vessels if v.controllable.name == self._target_controllable_name]
        self._vessels = vessels

        instruction = await self._prompter.make_instruction(playbook, self._goal.description)
        self._instruction = instruction.strip() or self._goal.description
        self._victim_system_prompt = find_victim_system_prompt(items, self._observables)
        self._user_goal = recover_user_goal(items, self._observables)
        self._action_url = recover_action_url(items, self._observables)

        if not self._vessels:
            reason = (
                "no in-scope content or prompt surface"
                if not self._can_inject
                else "no surface survived grafter ranking"
            )
            logger.info("MUZZLE: %s; passthrough baseline.", reason)
            self._exhausted = True
            return RunEndResponse(event=event, done=True)

        self._phase = _PROBE
        self._vessel_idx = 0
        self._current_inject_value = PLACEHOLDER_INSTR
        return RunEndResponse(event=event, done=False)

    async def _end_probe(self, event: RunEndEvent) -> RunEndResponse:
        items = self._snapshot()
        transcript = build_transcript(items, response_observable_names=self._response_names)
        surfaced = self._injected and await contains_injected_content(
            transcript, {"injection": PLACEHOLDER_INSTR}
        )
        if not surfaced:
            return self._next_vessel_or_done(event)

        # Surfaced: localize the PAIR template + victim system prompt, then attack.
        try:
            step_text, sys_prompt = extract_placeholder_template(transcript, _PLACEHOLDER_TOKENS)
            self._pair_template = step_text
            if sys_prompt and not self._victim_system_prompt:
                self._victim_system_prompt = sys_prompt
        except PlaceholderNotFoundError:
            self._pair_template = None
        self._phase = _ATTACK
        self._current_inject_value = None  # triggers PAIR at the next RunStart
        self._reflection = None
        return RunEndResponse(event=event, done=False)

    async def _end_attack(self, event: RunEndEvent) -> RunEndResponse:
        assert self._judge is not None
        framework_success = event.evaluation.success if event.evaluation is not None else None
        if framework_success:
            self._succeeded = True
            return RunEndResponse(event=event, done=True)

        self._attempt_index += 1
        items = self._snapshot()
        transcript = build_transcript(items, response_observable_names=self._response_names)
        payload = self._current_inject_value or self._instruction
        payload_present = await contains_injected_content(transcript, {"injection": payload})
        vessel = self._current_vessel()
        evaluation = await self._judge.evaluate(
            attack_method=self._attack_method(vessel),
            malicious_instruction={"injection": payload},
            transcript=transcript,
            payload_present=payload_present,
            framework_success=framework_success,
        )
        attribution = self._attribution(evaluation)

        if self._is_done():
            return RunEndResponse(event=event, done=True)

        if attribution == "instruction":
            # Payload delivered but ineffective: refine on the same vessel via PAIR.
            self._reflection = {
                "most_recent_working_payload": payload,
                "recommendations": evaluation.get("recommendations"),
                "reflection_details": evaluation,
            }
            self._current_inject_value = None
            return RunEndResponse(event=event, done=False)

        # "UI" / "unclear": payload did not land here -> try the next vessel.
        return self._next_vessel_or_done(event)

    # ------------------------------------------------------------------ helpers

    def _next_vessel_or_done(self, event: RunEndEvent) -> RunEndResponse:
        self._advance_vessel()
        if self._vessel_idx >= len(self._vessels):
            self._exhausted = True
            return RunEndResponse(event=event, done=True)
        self._phase = _PROBE
        self._current_inject_value = PLACEHOLDER_INSTR
        return RunEndResponse(event=event, done=False)

    def _advance_vessel(self) -> None:
        self._vessel_idx += 1
        self._pair_template = None
        self._reflection = None
        self._current_inject_value = None

    async def _run_pair(self) -> None:
        assert self._pair is not None
        result = await self._pair.craft(
            instruction=self._instruction,
            transcript_template=self._pair_template,
            victim_system_prompt=self._victim_system_prompt,
            reflection=self._reflection,
        )
        payload = result.best_prompt or self._instruction
        self._current_inject_value = self._shape_payload(payload)

    def _shape_payload(self, payload: str) -> str:
        if not self._use_injection_template:
            return payload
        return render_injection(
            instruction=payload, action_url=self._action_url, user_goal=self._user_goal
        )

    def _compose(self, base: str, payload: str) -> str:
        base = base or ""
        if base.strip():
            return f"{base}\n\n{payload}"
        return payload

    def _current_vessel(self) -> Vessel | None:
        if 0 <= self._vessel_idx < len(self._vessels):
            return self._vessels[self._vessel_idx]
        return None

    @staticmethod
    def _event_matches_vessel(controllable: Controllable, vessel: Vessel, is_post: bool) -> bool:
        # Content surfaces are PostCall, prompt fallbacks are PreCall. Match the controllable
        # by VALUE equality (targets rebuild value-equal controllables across runs, and this
        # disambiguates same-named surfaces, e.g. ASB's opi_tool_observation, by domain).
        if vessel.kind == "content":
            return is_post and controllable == vessel.controllable
        return (not is_post) and controllable == vessel.controllable

    @staticmethod
    def _attack_method(vessel: Vessel | None) -> dict[str, Any]:
        if vessel is None:
            return {"element": "", "why": "", "how": "", "confidence": 0.0}
        return {
            "element": vessel.element or vessel.controllable.name,
            "why": vessel.why,
            "how": vessel.how,
            "confidence": vessel.confidence,
        }

    @staticmethod
    def _attribution(evaluation: dict[str, Any]) -> str:
        component = evaluation.get("component_attribution")
        if isinstance(component, dict):
            primary = component.get("primary")
            if isinstance(primary, str) and primary in _ATTRIBUTIONS:
                return primary
        return "unclear"

    def _snapshot(self) -> list[object]:
        trajectory = self.current_trajectory
        if trajectory is None:
            return []
        return list(trajectory.snapshot())

    def _is_done(self) -> bool:
        return self._succeeded or self._exhausted or self._attempt_index >= self._max_attempts


__all__ = ["MuzzleOptimizer"]
