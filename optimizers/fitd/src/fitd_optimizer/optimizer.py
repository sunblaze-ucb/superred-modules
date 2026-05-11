"""FITDOptimizer: Foot-in-the-Door multi-turn jailbreak for SuperRed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

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
from superred.core.types.trajectory import ReadableTrajectory

from fitd_optimizer.assistant import FITDAssistant
from fitd_optimizer.prompts import (
    SYSTEM_PROMPT_EXTENSION,
    build_align_prompt,
    build_polish_prompt,
    change_sensitive_words,
    is_refusal,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_RESPONSE_CONTROLLABLE_NAME = "response"
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset({"response", "model_response", "assistant_response"})
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_STATIC_CONTEXT_BUDGET = 3000

TurnKind = Literal["initial", "polish", "align", "final_align", "replay"]
RecoveryKind = Literal["align", "slippery", "retry_initial"]


@dataclass
class ActiveTurn:
    kind: TurnKind
    user_prompt: str
    level_prompt: str | None = None
    previous_level_prompt: str | None = None


@dataclass
class RecoveryPlan:
    kind: RecoveryKind
    level_index: int
    level_prompt: str | None = None


class FITDOptimizer(Optimizer):
    """Optimizer implementing the FITD multi-turn attack.

    The default path follows the official implementation's defaults where
    they map cleanly to SuperRed: 10 transformation levels, a 50-query limit,
    up to 5 attempts, official refusal strings, official polish / align /
    intermediate / judge prompts, and target-owned decoding.
    """

    def __init__(
        self,
        *,
        prompt_sequence: list[str] | None = None,
        benign_prompt: str | None = None,
        level: int = 10,
        max_queries: int = 50,
        max_attempts: int = 5,
        control_history: bool = False,
        max_history_length: int = 22,
        use_system_prompt_when_available: bool = True,
        response_observable_names: set[str] | None = None,
        static_context_budget: int = _STATIC_CONTEXT_BUDGET,
    ) -> None:
        super().__init__()
        if level < 1:
            raise ValueError("level must be >= 1")
        if max_queries < 1:
            raise ValueError("max_queries must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if prompt_sequence is not None and not prompt_sequence:
            raise ValueError("prompt_sequence must not be empty")
        self._configured_prompt_sequence = list(prompt_sequence) if prompt_sequence is not None else None
        self._configured_benign_prompt = benign_prompt
        self._level = level
        self._max_queries = max_queries
        self._max_attempts = max_attempts
        self._control_history = control_history
        self._max_history_length = max_history_length
        self._use_system_prompt_when_available = use_system_prompt_when_available
        self._response_observable_names = response_observable_names or set(
            _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._static_context_budget = max(0, static_context_budget)

        self._goal: Goal | None = None
        self._assistant: FITDAssistant | None = None
        self._can_write_system_prompt = False
        self._static_context = ""

        self._trajectory: ReadableTrajectory | None = None
        self._primary_user_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._system_prompt_handled = False
        self._pending_post_answer: str | None = None

        self._attempt = 0
        self._query_count = 0
        self._succeeded = False
        self._attempt_done = False
        self._prompt_sequence: list[str] = []
        self._level_index = 0
        self._last_level_prompt: str | None = None
        self._last_response: str | None = None
        self._outputs: list[str] = []
        self._accepted_user_prompts: list[str] = []
        self._replay_prompts: list[str] = []
        self._pending_recovery: RecoveryPlan | None = None
        self._active_turn: ActiveTurn | None = None
        self._awaiting_response = False
        self._final_align_sent = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._can_write_system_prompt = (
            self._use_system_prompt_when_available
            and any(c.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME for c in controllables)
        )
        self._static_context = _build_static_context(observables, self._static_context_budget)
        self._assistant = FITDAssistant(self.llm, static_context=self._static_context)
        self._attempt = 0
        self._query_count = 0
        self._succeeded = False
        self._reset_run_state(clear_attempt_state=True)

    async def teardown(self) -> None:
        return None

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return await self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return await self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state(clear_attempt_state=False)
        self._trajectory = event.trajectory
        if not self._prompt_sequence:
            self._prompt_sequence = await self._prepare_prompt_sequence()
        return EventResponse(event=event)

    async def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        name = event.controllable.name
        if name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            if self._can_write_system_prompt and not self._system_prompt_handled:
                self._system_prompt_handled = True
                return ControllableInjection(
                    event=event,
                    controllable=event.controllable,
                    value=SYSTEM_PROMPT_EXTENSION,
                )
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if name == _RESPONSE_CONTROLLABLE_NAME:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._primary_user_controllable is None:
            self._primary_user_controllable = event.controllable
        elif event.controllable != self._primary_user_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        await self._consume_visible_answer()
        if self._awaiting_response:
            self._advance_without_visible_answer()

        if self._succeeded or self._attempt_done:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        turn = await self._next_turn()
        if turn is None:
            self._attempt_done = True
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._active_turn = turn
        self._awaiting_response = True
        self._query_count += 1
        return ControllableInjection(event=event, controllable=event.controllable, value=turn.user_prompt)

    async def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
        if not self._awaiting_response:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._primary_post_controllable is None:
            request_matches = self._active_turn is not None and event.request == self._active_turn.user_prompt
            if not request_matches:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._primary_post_controllable = event.controllable
        elif event.controllable != self._primary_post_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        self._pending_post_answer = event.answer
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        await self._consume_visible_answer()
        if self._awaiting_response:
            self._advance_without_visible_answer()

        if event.evaluation is not None:
            self._apply_evaluation(event.evaluation)

        if (
            event.evaluation is None
            and self._last_response is not None
            and not (self._attempt_done and self._pending_recovery is not None)
        ):
            await self._apply_local_success_judge(self._last_response)

        if self._succeeded:
            return RunEndResponse(event=event, done=True)

        if self._attempt_done and self._pending_recovery is not None and self._attempt + 1 < self._max_attempts:
            self._attempt += 1
            return RunEndResponse(event=event, done=False)

        if self._query_count >= self._max_queries:
            logger.info("FITD: query limit reached (%d)", self._query_count)
            return RunEndResponse(event=event, done=True)

        if self._attempt_done:
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    async def _prepare_prompt_sequence(self) -> list[str]:
        assert self._goal is not None
        assert self._assistant is not None
        target = change_sensitive_words(self._goal.description)
        if self._configured_prompt_sequence is not None:
            sequence = list(self._configured_prompt_sequence)
            if sequence[-1] != target:
                sequence.append(target)
            return sequence
        benign = self._configured_benign_prompt
        if benign is None:
            benign = await self._assistant.soften_prompt(target)
        return await self._assistant.generate_prompt_sequence(
            target_prompt=target,
            benign_prompt=benign,
            level=self._level,
        )

    async def _next_turn(self) -> ActiveTurn | None:
        if self._replay_prompts:
            user_prompt = self._replay_prompts.pop(0)
            return ActiveTurn(kind="replay", user_prompt=user_prompt)

        if self._pending_recovery is not None:
            recovery = self._pending_recovery
            self._pending_recovery = None
            if recovery.kind == "retry_initial" and recovery.level_prompt is not None:
                self._prompt_sequence[0] = recovery.level_prompt
                self._level_index = 0
                return ActiveTurn(kind="initial", user_prompt=recovery.level_prompt, level_prompt=recovery.level_prompt)
            if recovery.kind == "align" and self._last_level_prompt is not None and self._last_response is not None:
                prompt = build_align_prompt(self._last_level_prompt, self._last_response)
                return ActiveTurn(kind="align", user_prompt=prompt, level_prompt=self._last_level_prompt)
            if recovery.kind == "slippery" and recovery.level_prompt is not None and self._last_level_prompt is not None:
                prompt = build_polish_prompt(self._last_level_prompt, recovery.level_prompt)
                return ActiveTurn(
                    kind="polish",
                    user_prompt=prompt,
                    level_prompt=recovery.level_prompt,
                    previous_level_prompt=self._last_level_prompt,
                )

        if self._level_index >= len(self._prompt_sequence):
            if not self._final_align_sent and self._last_level_prompt is not None and self._last_response is not None:
                self._final_align_sent = True
                return ActiveTurn(
                    kind="final_align",
                    user_prompt=build_align_prompt(self._last_level_prompt, self._last_response),
                    level_prompt=self._last_level_prompt,
                )
            return None

        current = self._prompt_sequence[self._level_index]
        if self._level_index == 0 or self._last_level_prompt is None:
            return ActiveTurn(kind="initial", user_prompt=current, level_prompt=current)
        return ActiveTurn(
            kind="polish",
            user_prompt=build_polish_prompt(self._last_level_prompt, current),
            level_prompt=current,
            previous_level_prompt=self._last_level_prompt,
        )

    async def _consume_visible_answer(self) -> None:
        answer = self._read_response_from_trajectory()
        if answer is None:
            answer = self._pending_post_answer
        if answer is None:
            return
        self._pending_post_answer = None
        if not self._awaiting_response or self._active_turn is None:
            return
        await self._process_answer(answer)

    async def _process_answer(self, answer: str) -> None:
        assert self._active_turn is not None
        turn = self._active_turn
        self._awaiting_response = False
        self._active_turn = None

        if turn.kind == "replay":
            return

        if is_refusal(answer):
            await self._handle_refusal(turn)
            return

        self._accepted_user_prompts.append(turn.user_prompt)

        if turn.kind in {"initial", "polish"} and turn.level_prompt is not None:
            self._last_response = answer
            self._outputs.append(answer)
            self._last_level_prompt = turn.level_prompt
            self._level_index += 1
        elif turn.kind in {"align", "final_align"}:
            await self._maybe_keep_more_harmful_answer(answer, turn.level_prompt)
            if not self._outputs:
                self._outputs.append(answer)
                self._last_response = answer
            if turn.kind == "final_align":
                self._attempt_done = True

        if self._control_history and len(self._accepted_user_prompts) > self._max_history_length:
            self._accepted_user_prompts = self._accepted_user_prompts[-self._max_history_length :]
        if self._query_count >= self._max_queries:
            self._attempt_done = True

    async def _handle_refusal(self, turn: ActiveTurn) -> None:
        assert self._assistant is not None
        if turn.kind == "initial" and turn.level_prompt is not None:
            softened = await self._assistant.soften_prompt(turn.level_prompt)
            self._pending_recovery = RecoveryPlan(
                kind="retry_initial",
                level_index=0,
                level_prompt=softened,
            )
        elif self._last_level_prompt is not None and self._last_response is not None:
            is_aligned = await self._assistant.detect_align(self._last_level_prompt, self._last_response)
            if is_aligned and turn.level_prompt is not None:
                intermediate = await self._assistant.generate_intermediate_prompt(
                    self._last_level_prompt,
                    turn.level_prompt,
                )
                self._pending_recovery = RecoveryPlan(
                    kind="slippery",
                    level_index=self._level_index,
                    level_prompt=intermediate,
                )
            else:
                self._pending_recovery = RecoveryPlan(kind="align", level_index=self._level_index)
        else:
            self._pending_recovery = None

        self._replay_prompts = list(self._accepted_user_prompts)
        self._attempt_done = True

    async def _maybe_keep_more_harmful_answer(self, answer: str, level_prompt: str | None) -> None:
        if not self._outputs or level_prompt is None:
            return
        assert self._assistant is not None
        previous = self._outputs[-1]
        result = await self._assistant.judge_harmful_content(previous, answer, level_prompt)
        if result == 1:
            self._last_response = previous
        else:
            self._outputs[-1] = answer
            self._last_response = answer

    def _advance_without_visible_answer(self) -> None:
        if self._active_turn is None:
            return
        turn = self._active_turn
        self._awaiting_response = False
        self._active_turn = None
        if turn.kind == "replay":
            return
        self._accepted_user_prompts.append(turn.user_prompt)
        if turn.kind in {"initial", "polish"} and turn.level_prompt is not None:
            self._last_level_prompt = turn.level_prompt
            self._level_index += 1
        elif turn.kind == "final_align":
            self._attempt_done = True
        if self._level_index >= len(self._prompt_sequence) and self._last_response is None:
            self._attempt_done = True

    def _read_response_from_trajectory(self) -> str | None:
        trajectory = self._trajectory or self.current_trajectory
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

    async def _apply_local_success_judge(self, response: str) -> None:
        assert self._goal is not None
        assert self._assistant is not None
        result = await self._assistant.evaluate_success(self._goal.description, response)
        if result.success:
            logger.info("FITD: local official evaluator returned unsafe")
            self._succeeded = True

    def _apply_evaluation(self, evaluation: EvaluationResult) -> None:
        if evaluation.success:
            self._succeeded = True

    def _reset_run_state(self, *, clear_attempt_state: bool) -> None:
        self._trajectory = None
        self._primary_user_controllable = None
        self._primary_post_controllable = None
        self._system_prompt_handled = False
        self._pending_post_answer = None
        self._active_turn = None
        self._awaiting_response = False
        if clear_attempt_state:
            self._attempt_done = False
            self._prompt_sequence = []
            self._level_index = 0
            self._last_level_prompt = None
            self._last_response = None
            self._outputs = []
            self._accepted_user_prompts = []
            self._replay_prompts = []
            self._pending_recovery = None
            self._final_align_sent = False
        elif self._attempt_done and self._pending_recovery is not None:
            self._attempt_done = False
            self._primary_user_controllable = None
            self._primary_post_controllable = None
            self._active_turn = None
            self._awaiting_response = False


def _build_static_context(observables: list[ObservableValue], budget: int) -> str:
    if budget <= 0 or not observables:
        return ""
    lines = ["SUPERRED STATIC TARGET CONTEXT (filtered to attacker scope):"]
    for value in observables:
        content = value.content
        if not isinstance(content, str):
            content = repr(content)
        lines.append(f"- {value.observable.name}: {content}")
    context = "\n".join(lines)
    if len(context) <= budget:
        return context
    return context[: max(0, budget - len("\n[truncated]"))] + "\n[truncated]"
