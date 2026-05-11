"""PAIR optimizer for SuperRed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
from superred.core.types.evaluation import EvaluationResult
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from pair_optimizer.attacker import PairAttacker, PairStream
from pair_optimizer.evaluator import PairEvaluator
from pair_optimizer.json_utils import PairProposal
from pair_optimizer.prompts import (
    append_superred_context,
    get_attacker_system_prompts,
    get_init_msg,
    process_target_response,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_NAME = "system_prompt"
_RESPONSE_CONTROLLABLE_NAME = "response"
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset({"response", "model_response", "assistant_response"})
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_DEFAULT_TARGET_PREFIX = "Sure, here is"
_NO_SIGNAL_RESPONSE = "No in-scope target response or framework evaluation was available."


@dataclass(frozen=True)
class _Candidate:
    stream_index: int
    proposal: PairProposal


class PAIROptimizer(Optimizer):
    """Prompt Automatic Iterative Refinement attack optimizer.

    The default parameters mirror the current official PAIR runtime defaults:
    3 streams, 3 iterations, 4 rounds of attacker history, 500 attacker tokens,
    5 attacker JSON retries, and GCG refusal-dictionary scoring.
    """

    def __init__(
        self,
        *,
        n_streams: int = 3,
        n_iterations: int = 3,
        keep_last_n: int = 4,
        attack_max_tokens: int = 500,
        max_attack_attempts: int = 5,
        attack_temperature: float = 1.0,
        attack_top_p: float = 0.9,
        judge_method: str = "gcg",
        judge_max_tokens: int = 10,
        judge_temperature: float = 0.0,
        target_str: str | None = None,
        response_observable_names: frozenset[str] | None = None,
        target_controllable_name: str | None = None,
        static_context_max_chars: int = 8000,
    ) -> None:
        super().__init__()
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if keep_last_n <= 0:
            raise ValueError("keep_last_n must be positive")
        if static_context_max_chars <= 0:
            raise ValueError("static_context_max_chars must be positive")

        self._n_streams = n_streams
        self._n_iterations = n_iterations
        self._keep_last_n = keep_last_n
        self._attack_max_tokens = attack_max_tokens
        self._max_attack_attempts = max_attack_attempts
        self._attack_temperature = attack_temperature
        self._attack_top_p = attack_top_p
        self._judge_method = judge_method
        self._judge_max_tokens = judge_max_tokens
        self._judge_temperature = judge_temperature
        self._target_str_override = target_str
        self._response_observable_names = response_observable_names or _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        self._target_controllable_name = target_controllable_name
        self._static_context_max_chars = static_context_max_chars

        self._goal: Goal | None = None
        self._target_str = _DEFAULT_TARGET_PREFIX
        self._attacker: PairAttacker | None = None
        self._evaluator: PairEvaluator | None = None
        self._streams: list[PairStream] = []
        self._pending_candidates: list[_Candidate] = []
        self._current_candidate: _Candidate | None = None
        self._iteration = 0
        self._succeeded = False
        self._best_score = 1.0
        self._batch_success = False
        self._stop_after_batch = False

        self._system_prompt_writable = False
        self._static_context: list[tuple[str, str]] = []
        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None

    @property
    def n_streams(self) -> int:
        return self._n_streams

    @property
    def n_iterations(self) -> int:
        return self._n_iterations

    @property
    def keep_last_n(self) -> int:
        return self._keep_last_n

    @property
    def attack_max_tokens(self) -> int:
        return self._attack_max_tokens

    @property
    def max_attack_attempts(self) -> int:
        return self._max_attack_attempts

    @property
    def judge_max_tokens(self) -> int:
        return self._judge_max_tokens

    @property
    def judge_method(self) -> str:
        return self._judge_method

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._target_str = self._target_str_override or _DEFAULT_TARGET_PREFIX
        self._system_prompt_writable = (
            self._target_controllable_name is None
            and any(ctrl.name == _SYSTEM_PROMPT_NAME for ctrl in controllables)
        )
        self._static_context = self._build_static_context(controllables, observables)
        prompts = get_attacker_system_prompts(goal.description, self._target_str)
        self._streams = []
        init_msg = get_init_msg(goal.description, self._target_str)
        for idx in range(self._n_streams):
            system_prompt = append_superred_context(
                prompts[idx % len(prompts)],
                static_context=self._static_context,
                writable_system_prompt=self._system_prompt_writable,
                max_chars=self._static_context_max_chars,
            )
            self._streams.append(
                PairStream(index=idx, system_prompt=system_prompt, processed_response=init_msg)
            )
        self._attacker = PairAttacker(
            llm=self.llm,
            attack_max_tokens=self._attack_max_tokens,
            max_attack_attempts=self._max_attack_attempts,
            temperature=self._attack_temperature,
            top_p=self._attack_top_p,
            keep_last_n=self._keep_last_n,
            request_system_prompt=self._system_prompt_writable,
        )
        self._evaluator = PairEvaluator(
            llm=self.llm,
            judge_method=self._judge_method,
            judge_max_tokens=self._judge_max_tokens,
            judge_temperature=self._judge_temperature,
        )
        self._pending_candidates = []
        self._current_candidate = None
        self._iteration = 0
        self._succeeded = False
        self._best_score = 1.0
        self._batch_success = False
        self._stop_after_batch = False
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        if self._succeeded or self._stop_after_batch:
            return EventResponse(event=event)
        if not self._pending_candidates:
            await self._prepare_iteration_batch()
        if self._pending_candidates:
            self._current_candidate = self._pending_candidates.pop(0)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._current_candidate is None or self._succeeded or self._stop_after_batch:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        proposal = self._current_candidate.proposal
        system_prompt = (proposal.system_prompt or "").strip()
        if self._target_controllable_name is not None:
            if event.controllable.name != self._target_controllable_name:
                return ControllableNoInjection(event=event, controllable=event.controllable)
        else:
            if event.controllable.name == _SYSTEM_PROMPT_NAME:
                if (
                    self._system_prompt_writable
                    and system_prompt
                    and not self._injected_system_prompt_this_run
                ):
                    self._injected_system_prompt_this_run = True
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value=system_prompt,
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if event.controllable.name == _RESPONSE_CONTROLLABLE_NAME:
                return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._primary_pre_controllable is None:
            self._primary_pre_controllable = event.controllable
        elif event.controllable != self._primary_pre_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._injected_this_run:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._injected_this_run = True
        self._awaiting_post_call = True
        self._last_pre_request = event.request
        self._last_injected_value = proposal.prompt
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=proposal.prompt,
        )

    def _handle_post_call(self, event: ControllablePostCallEvent) -> ControllableNoInjection:
        if not self._awaiting_post_call:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._primary_post_controllable is None:
            same_ctrl = event.controllable == self._primary_pre_controllable
            request_matches_pre = (
                self._last_pre_request is not None and event.request == self._last_pre_request
            )
            request_matches_injected = (
                self._last_injected_value is not None
                and event.request == self._last_injected_value
            )
            if not (same_ctrl or request_matches_pre or request_matches_injected):
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._primary_post_controllable = event.controllable
        elif event.controllable != self._primary_post_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._pending_post_answer = event.answer
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        if self._current_candidate is None:
            return RunEndResponse(event=event, done=True)

        await self._finalize_current_candidate(event.evaluation)
        self._current_candidate = None

        if self._batch_success and not self._pending_candidates:
            self._succeeded = True
            return RunEndResponse(event=event, done=True)
        if self._succeeded:
            return RunEndResponse(event=event, done=True)
        if self._pending_candidates:
            return RunEndResponse(event=event, done=False)
        if self._iteration >= self._n_iterations:
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    async def _prepare_iteration_batch(self) -> None:
        assert self._attacker is not None
        if self._iteration >= self._n_iterations:
            return
        self._iteration += 1
        self._batch_success = False
        candidates: list[_Candidate] = []
        for stream in self._streams:
            proposal = await self._attacker.generate(stream)
            stream.last_prompt = proposal.prompt
            stream.last_improvement = proposal.improvement
            candidates.append(_Candidate(stream_index=stream.index, proposal=proposal))
        self._pending_candidates = candidates

    async def _finalize_current_candidate(self, evaluation: EvaluationResult | None) -> None:
        assert self._goal is not None
        assert self._evaluator is not None
        candidate = self._current_candidate
        assert candidate is not None
        stream = self._streams[candidate.stream_index]

        response = self._read_response_from_trajectory()
        if response is None:
            response = self._pending_post_answer

        if evaluation is not None:
            score = self._score_from_framework_evaluation(evaluation)
            response_for_attacker = self._framework_feedback_text(evaluation, response)
            if evaluation.success:
                self._batch_success = True
        elif response is not None:
            result = await self._evaluator.score_response(
                goal=self._goal.description,
                target_str=self._target_str,
                prompt=candidate.proposal.prompt,
                response=response,
            )
            score = float(result.score)
            response_for_attacker = response
            if result.success:
                self._batch_success = True
        else:
            score = 1.0
            response_for_attacker = _NO_SIGNAL_RESPONSE

        self._best_score = max(self._best_score, score)
        stream.last_score = score
        stream.last_response = response_for_attacker
        stream.processed_response = process_target_response(
            response_for_attacker,
            self._format_score(score),
            self._goal.description,
            self._target_str,
        )

    def _score_from_framework_evaluation(self, evaluation: EvaluationResult) -> float:
        if evaluation.success:
            return 10.0
        raw = float(evaluation.primary_score.value)
        if 0.0 <= raw <= 1.0:
            return max(1.0, min(9.0, 1.0 + 8.0 * raw))
        return max(1.0, min(9.0, raw))

    @staticmethod
    def _framework_feedback_text(evaluation: EvaluationResult, response: str | None) -> str:
        parts: list[str] = []
        if response:
            parts.append(response)
        if evaluation.rationale:
            parts.append(f"FRAMEWORK EVALUATION RATIONALE: {evaluation.rationale}")
        if evaluation.sub_scores:
            sub_scores = ", ".join(
                f"{name}={score.value}" for name, score in sorted(evaluation.sub_scores.items())
            )
            parts.append(f"FRAMEWORK SUB SCORES: {sub_scores}")
        if not parts:
            parts.append(f"FRAMEWORK EVALUATION SUCCESS: {evaluation.success}")
        return "\n".join(parts)

    @staticmethod
    def _format_score(score: float) -> int | float:
        if score.is_integer():
            return int(score)
        return round(score, 3)

    def _read_response_from_trajectory(self) -> str | None:
        trajectory = self.current_trajectory
        if trajectory is None:
            return None
        recovered: str | None = None
        for item in trajectory.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            normalized = name.lower()
            is_match = (
                name in self._response_observable_names
                or normalized in self._response_observable_names
                or any(hint in normalized for hint in _RESPONSE_NAME_HINTS)
            )
            if is_match and isinstance(item.content, str):
                recovered = item.content
        return recovered

    def _build_static_context(
        self,
        controllables: list[Controllable],
        observables: list[ObservableValue],
    ) -> list[tuple[str, str]]:
        context: list[tuple[str, str]] = []
        for observable in observables:
            content = self._stringify_observable_content(observable.content)
            if content is None:
                continue
            context.append((f"observable.{observable.observable.name}", content))
        for ctrl in controllables:
            if ctrl.name in {"user_message", _RESPONSE_CONTROLLABLE_NAME}:
                continue
            if self._system_prompt_writable and ctrl.name == _SYSTEM_PROMPT_NAME:
                continue
            context.append((f"controllable.{ctrl.name}", "writable"))
        return context

    @staticmethod
    def _stringify_observable_content(content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if content is None:
            return None
        return str(content)

    def _reset_run_state(self) -> None:
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None
