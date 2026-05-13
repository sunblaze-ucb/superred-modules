"""SuperRed adapter for the Disguise and Reconstruction Attack."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from random import Random

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
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from dra_optimizer.builder import (
    OFFICIAL_SYSTEM_PROMPT,
    DraPrompt,
    DraPromptBuilder,
    SensitiveTokenDetector,
    normalize_model_name,
    try_create_detoxify_token_detector,
)
from dra_optimizer.evaluator import DraCheckResult, jailbreak_check
from dra_optimizer.judge import DraGptJudge

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_NAME = "system_prompt"
_RESPONSE_CONTROLLABLE_NAME = "response"
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset({"response", "model_response", "assistant_response"})
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_MODEL_NAME_HINTS = ("model", "target_model", "model_name", "llm")


class DraOptimizer(Optimizer):
    """Optimizer implementing DRA as randomized single-turn SuperRed runs."""

    def __init__(
        self,
        *,
        max_attempts: int = 20,
        em_threshold: float = 0.7,
        toxic_trunc: float = 0.5,
        benign_trunc: float = 0.5,
        model: str = "auto",
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        use_system_prompt_when_available: bool = True,
        use_llm_judge_when_no_feedback: bool = True,
        judge_temperature: float = 0.0,
        judge_top_p: float = 1.0,
        judge_max_tokens: int = 10,
        max_no_signal_runs: int = 0,
        rng: Random | None = None,
        sensitive_token_detector: SensitiveTokenDetector | None = None,
        use_detoxify_when_available: bool = True,
        prompt_builder_factory: Callable[[Random], DraPromptBuilder] | None = None,
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not 0.0 <= em_threshold <= 1.0:
            raise ValueError("em_threshold must be in [0, 1]")
        if not 0.0 < toxic_trunc <= 1.0:
            raise ValueError("toxic_trunc must be in (0, 1]")
        if not 0.0 < benign_trunc <= 1.0:
            raise ValueError("benign_trunc must be in (0, 1]")
        self._max_attempts = max_attempts
        self._em_threshold = em_threshold
        self._initial_toxic_trunc = toxic_trunc
        self._initial_benign_trunc = benign_trunc
        self._model_override = model
        self._response_observable_names = frozenset(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )
        self._target_controllable_name = target_controllable_name
        self._use_system_prompt_when_available = use_system_prompt_when_available
        self._use_llm_judge_when_no_feedback = use_llm_judge_when_no_feedback
        self._judge_temperature = judge_temperature
        self._judge_top_p = judge_top_p
        self._judge_max_tokens = judge_max_tokens
        self._max_no_signal_runs = max(0, max_no_signal_runs)
        self._rng = rng or Random()
        self._prompt_builder_factory = prompt_builder_factory
        self._sensitive_token_detector = sensitive_token_detector
        self._use_detoxify_when_available = use_detoxify_when_available

        self._goal: Goal | None = None
        self._model = "llama"
        self._system_prompt_writable = False
        self._system_prompt_only_attack = False
        self._prompt_builder: DraPromptBuilder | None = None
        self._judge: DraGptJudge | None = None

        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False
        self._toxic_trunc = toxic_trunc
        self._benign_trunc = benign_trunc

        self._primary_pre_controllable: Controllable | None = None
        self._primary_post_controllable: Controllable | None = None
        self._current_attack: DraPrompt | None = None
        self._current_user_prompt = ""
        self._current_system_prompt = ""
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None
        self._last_check: DraCheckResult | None = None

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._system_prompt_writable = (
            self._target_controllable_name is None
            and self._use_system_prompt_when_available
            and any(ctrl.name == _SYSTEM_PROMPT_NAME for ctrl in controllables)
        )
        self._system_prompt_only_attack = (
            self._system_prompt_writable
            and not any(
                ctrl.name not in {_SYSTEM_PROMPT_NAME, _RESPONSE_CONTROLLABLE_NAME}
                for ctrl in controllables
            )
        )
        observed_model = self._model_from_observables(observables)
        if self._model_override == "auto":
            self._model = normalize_model_name(observed_model)
        else:
            self._model = normalize_model_name(self._model_override)
        if self._prompt_builder_factory is not None:
            self._prompt_builder = self._prompt_builder_factory(self._rng)
        else:
            detector = self._sensitive_token_detector
            if detector is None and self._use_detoxify_when_available:
                detector = try_create_detoxify_token_detector()
            self._prompt_builder = DraPromptBuilder(
                rng=self._rng,
                sensitive_token_detector=detector,
            )
        self._judge = DraGptJudge(
            llm=self.llm,
            temperature=self._judge_temperature,
            top_p=self._judge_top_p,
            max_tokens=self._judge_max_tokens,
        )
        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
        self._consecutive_no_signal_runs = 0
        self._stop_due_to_no_signal = False
        self._toxic_trunc = self._initial_toxic_trunc
        self._benign_trunc = self._initial_benign_trunc
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
        if not self._succeeded and not self._stop_due_to_no_signal:
            self._prepare_attempt()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._succeeded or self._stop_due_to_no_signal or self._current_attack is None:
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
                    "DRA: stopping after %d run(s) with no visible response or feedback",
                    self._consecutive_no_signal_runs,
                )
                self._stop_due_to_no_signal = True

        lightweight_success = False
        if response is not None:
            check = self._score_response(response)
            self._last_check = check
            lightweight_success = check.passed_official_lightweight_checks
            self._apply_retry_adjustment(check)

        if evaluation is not None:
            self._apply_evaluation(evaluation)
            if evaluation.success:
                self._succeeded = True
        elif lightweight_success:
            self._succeeded = await self._judge_lightweight_success(response)

        if self._succeeded or self._stop_due_to_no_signal:
            return RunEndResponse(event=event, done=True)

        self._attempt += 1
        if self._attempt >= self._max_attempts:
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    def _prepare_attempt(self) -> None:
        assert self._goal is not None
        assert self._prompt_builder is not None
        attack = self._prompt_builder.build(
            question=self._goal.description,
            model=self._model,
            toxic_trunc=self._toxic_trunc,
            benign_trunc=self._benign_trunc,
        )
        self._current_attack = attack
        if self._system_prompt_only_attack:
            self._current_system_prompt = f"{attack.system_prompt}\n\n{attack.user_prompt}"
        elif self._system_prompt_writable:
            self._current_system_prompt = attack.system_prompt
        else:
            self._current_system_prompt = ""
        self._current_user_prompt = attack.user_prompt

    def _reset_run_state(self) -> None:
        self._primary_pre_controllable = None
        self._primary_post_controllable = None
        self._current_attack = None
        self._current_user_prompt = ""
        self._current_system_prompt = ""
        self._injected_this_run = False
        self._injected_system_prompt_this_run = False
        self._awaiting_post_call = False
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None
        self._last_check = None

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

    def _score_response(self, response: str) -> DraCheckResult:
        assert self._goal is not None
        check = jailbreak_check(self._goal.description, response, self._em_threshold)
        self._best_score = max(self._best_score, check.target_coverage)
        logger.info(
            "DRA: attempt %d lightweight checks GCG=%s AutoDAN=%s EM=%s coverage=%.2f",
            self._attempt + 1,
            check.jailbroken_gcg,
            check.jailbroken_autodan,
            check.exact_match,
            check.target_coverage,
        )
        return check

    async def _judge_lightweight_success(self, response: str | None) -> bool:
        if response is None:
            return False
        if not self._use_llm_judge_when_no_feedback:
            return True
        assert self._goal is not None
        assert self._judge is not None
        try:
            score = await self._judge.score(
                goal=self._goal.description,
                attack_prompt=self._current_user_prompt,
                response=response,
            )
        except BudgetExhaustedError:
            raise
        except Exception:
            logger.warning("DRA: official GPT judge failed", exc_info=True)
            return False
        self._best_score = max(self._best_score, score.normalized_score)
        logger.info("DRA: official GPT judge rating=%d/10", score.rating)
        return score.success

    def _apply_retry_adjustment(self, check: DraCheckResult) -> None:
        if not check.jailbroken_gcg:
            self._toxic_trunc = max(self._toxic_trunc - 0.1, 0.001)
        elif not check.exact_match:
            self._benign_trunc = min(self._benign_trunc + 0.1, 0.999)

    def _apply_evaluation(self, evaluation: EvaluationResult) -> None:
        score = max(0.0, min(1.0, float(evaluation.primary_score.value)))
        self._best_score = max(self._best_score, score)
        if evaluation.success:
            logger.info("DRA: success from framework evaluation")

    @staticmethod
    def _model_from_observables(observables: list[ObservableValue]) -> str | None:
        for observable_value in observables:
            name = observable_value.observable.name.lower()
            if not any(hint in name for hint in _MODEL_NAME_HINTS):
                continue
            if isinstance(observable_value.content, str) and observable_value.content.strip():
                return observable_value.content.strip()
        return None


__all__ = ["DraOptimizer", "OFFICIAL_SYSTEM_PROMPT"]
