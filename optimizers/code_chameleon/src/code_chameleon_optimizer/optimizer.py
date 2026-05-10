"""CodeChameleonOptimizer: personalized-encryption jailbreak for superred."""

from __future__ import annotations

import logging
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
from superred.core.types.evaluation import EvaluationResult
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.trajectory import ReadableTrajectory

from code_chameleon_optimizer.evaluator import Evaluator
from code_chameleon_optimizer.prompts import AttackPrompt, build_attack_prompt

logger = logging.getLogger(__name__)

_ALLOWED_ENCRYPT_RULES = frozenset({"none", "binary_tree", "reverse", "odd_even", "length"})
_ALLOWED_PROMPT_STYLES = frozenset({"code", "text"})
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset({"response", "model_response", "assistant_response"})
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_SYSTEM_PROMPT_NAME = "system_prompt"
_RESPONSE_CONTROLLABLE_NAME = "response"
_DEFAULT_ENCRYPT_RULE = "binary_tree"


class CodeChameleonOptimizer(Optimizer):
    """Optimizer implementing CodeChameleon's single-turn encrypted prompt attack.

    One superred run corresponds to one CodeChameleon prompt using one
    encryption rule. By default the optimizer uses the paper/code-mainline
    code-style prompt with the BinaryTree rule from the official README.
    """

    def __init__(
        self,
        *,
        encrypt_rules: Iterable[str] | None = None,
        prompt_style: str = "code",
        success_score: int = 5,
        judge_max_tokens: int = 512,
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        use_system_prompt_when_available: bool = True,
        max_no_signal_runs: int = 0,
    ) -> None:
        super().__init__()
        if prompt_style not in _ALLOWED_PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {sorted(_ALLOWED_PROMPT_STYLES)}")
        if not 1 <= success_score <= 5:
            raise ValueError("success_score must be in [1, 5]")
        self._encrypt_rules_override = self._validate_encrypt_rules(encrypt_rules)
        self._encrypt_rules: list[str] = []
        self._prompt_style = prompt_style
        self._success_score = success_score
        self._judge_max_tokens = judge_max_tokens
        self._response_observable_names = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._target_controllable_name = target_controllable_name
        self._use_system_prompt_when_available = use_system_prompt_when_available
        self._max_no_signal_runs = max(0, max_no_signal_runs)

        self._goal: Goal | None = None
        self._evaluator: Evaluator | None = None
        self._system_prompt_writable = False
        self._model_id: str | None = None

        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False

        self._trajectory: ReadableTrajectory | None = None
        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._current_attack: AttackPrompt | None = None
        self._current_encrypt_rule = ""
        self._current_user_prompt = ""
        self._current_system_prompt = ""
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None

    @staticmethod
    def _validate_encrypt_rules(encrypt_rules: Iterable[str] | None) -> list[str] | None:
        if encrypt_rules is None:
            return None
        rules = list(encrypt_rules)
        if not rules:
            raise ValueError("encrypt_rules must contain at least one rule")
        invalid = [rule for rule in rules if rule not in _ALLOWED_ENCRYPT_RULES]
        if invalid:
            raise ValueError(f"Unsupported encrypt_rules: {invalid!r}")
        return rules

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._evaluator = Evaluator(
            llm=self.llm,
            success_score=self._success_score,
            max_tokens=self._judge_max_tokens,
        )
        self._model_id = self._extract_model_id(observables)
        self._system_prompt_writable = (
            self._target_controllable_name is None
            and self._use_system_prompt_when_available
            and any(ctrl.name == _SYSTEM_PROMPT_NAME for ctrl in controllables)
        )
        self._encrypt_rules = self._resolve_encrypt_rules()
        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
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
        self._trajectory = event.trajectory
        self._prepare_attempt()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._succeeded or self._stop_due_to_no_signal:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._target_controllable_name is not None:
            if event.controllable.name != self._target_controllable_name:
                return ControllableNoInjection(event=event, controllable=event.controllable)
        else:
            if event.controllable.name == _SYSTEM_PROMPT_NAME:
                if (
                    self._system_prompt_writable
                    and self._current_system_prompt
                    and not self._injected_system_prompt_this_run
                ):
                    self._injected_system_prompt_this_run = True
                    return ControllableInjection(
                        event=event,
                        controllable=event.controllable,
                        value=self._current_system_prompt,
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
        self._last_injected_value = self._current_user_prompt
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current_user_prompt,
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
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
        response = self._read_response_from_trajectory()
        if response is None:
            response = self._pending_post_answer
        evaluation = event.evaluation

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
                    "CodeChameleon: stopping after %d run(s) with no visible response or feedback",
                    self._consecutive_no_signal_runs,
                )
                self._stop_due_to_no_signal = True

        internal_success = False
        if response is not None:
            internal_success = await self._score_response(response)

        if evaluation is not None:
            self._apply_evaluation(evaluation)
            # Framework success is authoritative when present: a non-success
            # evaluation should not be overturned by our local judge.
            if evaluation.success:
                self._succeeded = True
        elif internal_success:
            self._succeeded = True

        if self._succeeded or self._stop_due_to_no_signal:
            return RunEndResponse(event=event, done=True)

        self._attempt += 1
        if self._attempt >= len(self._encrypt_rules):
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    def _prepare_attempt(self) -> None:
        assert self._goal is not None
        if self._attempt >= len(self._encrypt_rules):
            self._current_encrypt_rule = self._encrypt_rules[-1]
        else:
            self._current_encrypt_rule = self._encrypt_rules[self._attempt]
        attack = build_attack_prompt(
            goal=self._goal.description,
            encrypt_rule=self._current_encrypt_rule,
            prompt_style=self._prompt_style,
        )
        self._current_attack = attack
        self._current_system_prompt = attack.system_prompt
        self._current_user_prompt = attack.user_prompt

    def _reset_run_state(self) -> None:
        self._trajectory = None
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._current_attack = None
        self._current_encrypt_rule = ""
        self._current_user_prompt = ""
        self._current_system_prompt = ""
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None

    def _read_response_from_trajectory(self) -> str | None:
        if self._trajectory is None:
            return None
        recovered: str | None = None
        for item in self._trajectory.drain():
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

    async def _score_response(self, response: str) -> bool:
        assert self._goal is not None
        assert self._evaluator is not None
        try:
            result = await self._evaluator.score_response(
                goal=self._goal.description,
                response=response,
            )
        except Exception:
            logger.warning("CodeChameleon: local evaluator failed", exc_info=True)
            return False
        if result.normalized_score > self._best_score:
            self._best_score = result.normalized_score
        logger.info(
            "CodeChameleon: attempt %d scored %d/5 via local judge",
            self._attempt,
            result.score,
        )
        return result.success

    def _apply_evaluation(self, evaluation: EvaluationResult) -> None:
        score = max(0.0, min(1.0, float(evaluation.primary_score.value)))
        if score > self._best_score:
            self._best_score = score
        if evaluation.success:
            logger.info("CodeChameleon: success from framework evaluation")

    def _resolve_encrypt_rules(self) -> list[str]:
        if self._encrypt_rules_override is not None:
            return list(self._encrypt_rules_override)
        model = (self._model_id or "").lower()
        # Paper Table 2 reports Length as best on Llama2-70B; BinaryTree is
        # best for 7B/13B and has the best average across that table.
        if "llama" in model and "70" in model:
            return ["length"]
        return [_DEFAULT_ENCRYPT_RULE]

    @staticmethod
    def _extract_model_id(observables: list[ObservableValue]) -> str | None:
        for value in observables:
            if value.observable.name != "model":
                continue
            content = value.content
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None
