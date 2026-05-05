"""ManyShotOptimizer: single-turn many-shot jailbreak for superred."""

from __future__ import annotations

import json
import logging
import random
from importlib import resources

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ObservableEvent,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.evaluation import EvaluationResult
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.trajectory import ReadableTrajectory

from many_shot_optimizer.evaluator import Evaluator
from many_shot_optimizer.generator import ExampleGenerator
from many_shot_optimizer.prompt_builder import (
    build_final_user_prompt,
    build_prompt,
    build_system_demonstrations_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_COUNTS = [256]
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = ("response", "model_response", "assistant_response")
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")


def _load_bundled_examples() -> list[dict[str, str]]:
    """Load the bundled faux dialogue dataset from package data."""
    ref = resources.files("many_shot_optimizer") / "data" / "examples.json"
    with resources.as_file(ref) as path:
        with open(path, encoding="utf-8") as f:
            data: object = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ManyShot examples.json must contain a list")
    examples: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("ManyShot examples must be objects")
        user = item.get("user")
        assistant = item.get("assistant")
        if not isinstance(user, str) or not isinstance(assistant, str):
            raise ValueError("ManyShot examples must contain string user/assistant fields")
        example = {
            str(key): value
            for key, value in item.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        examples.append(example)
    return examples


class ManyShotOptimizer(Optimizer):
    """Optimizer implementing the Many-Shot Jailbreaking attack.

    Prepends an escalating number of faux Q&A dialogues to the objective
    in a single prompt. Each run cycle = one attempt with a specific
    example count from the schedule. Uses LLM generation when the count
    exceeds the bundled dataset.
    """

    def __init__(
        self,
        *,
        example_counts: list[int] | None = None,
        success_threshold: float = 0.8,
        custom_examples: list[dict[str, str]] | None = None,
        example_category: str | None = None,
        generate_extra: bool = False,
        shuffle_examples: bool = True,
        random_seed: int | None = None,
        evaluator_method: str = "refusal_classifier",
        response_observable_names: list[str] | None = None,
        use_system_prompt_when_available: bool = True,
    ) -> None:
        super().__init__()
        self._example_counts = example_counts if example_counts is not None else list(_DEFAULT_COUNTS)
        self._success_threshold = success_threshold
        self._custom_examples = custom_examples
        self._example_category = example_category
        self._generate_extra = generate_extra
        self._shuffle_examples = shuffle_examples
        self._random_seed = random_seed
        self._evaluator_method = evaluator_method
        self._use_system_prompt_when_available = use_system_prompt_when_available
        self._response_observable_names = set(
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )

        # State set in initialize()
        self._goal: Goal | None = None
        self._evaluator: Evaluator | None = None
        self._generator: ExampleGenerator | None = None
        self._primary_controllable: Controllable | None = None
        self._trajectory: ReadableTrajectory | None = None
        self._examples: list[dict[str, str]] = []
        self._generated_examples: list[dict[str, str]] = []

        # Per-run state
        self._attempt: int = 0
        self._succeeded: bool = False
        self._best_score: float = 0.0
        self._current_prompt: str = ""
        self._current_system_prompt: str = ""
        self._current_user_prompt: str = ""
        self._injected_prompt: bool = False
        self._saw_system_prompt_event: bool = False
        self._used_system_prompt_channel: bool = False
        self._awaiting_feedback: bool = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._evaluator = Evaluator(llm=self.llm, method=self._evaluator_method)
        self._generator = ExampleGenerator(llm=self.llm)
        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
        self._generated_examples = []

        base = self._custom_examples if self._custom_examples is not None else _load_bundled_examples()
        self._examples = self._filter_examples(list(base))
        if self._shuffle_examples:
            random.Random(self._random_seed).shuffle(self._examples)

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

    async def teardown(self) -> None:
        pass

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        assert self._goal is not None
        count = self._example_counts[self._attempt]
        examples = await self._get_examples(count)
        self._current_prompt = build_prompt(
            examples=examples, objective=self._goal.description,
        )
        self._current_system_prompt = build_system_demonstrations_prompt(
            examples=examples,
        )
        self._current_user_prompt = build_final_user_prompt(
            objective=self._goal.description,
        )
        self._trajectory = event.trajectory
        self._reset_run_state()
        logger.info("ManyShot: attempt %d, %d examples", self._attempt, len(examples))
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        if self._succeeded:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if event.controllable.name == "system_prompt":
            self._saw_system_prompt_event = True
            if self._use_system_prompt_when_available and not self._injected_prompt:
                self._primary_controllable = event.controllable
                self._injected_prompt = True
                self._used_system_prompt_channel = True
                return ControllableInjection(
                    event=event,
                    controllable=event.controllable,
                    value=self._current_system_prompt,
                )
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if event.controllable.name == "user_message":
            if self._used_system_prompt_channel:
                if self._awaiting_feedback:
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                self._awaiting_feedback = True
                return ControllableInjection(
                    event=event,
                    controllable=event.controllable,
                    value=self._current_user_prompt,
                )
            if self._injected_prompt:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._primary_controllable = event.controllable
            self._injected_prompt = True
            self._awaiting_feedback = True
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=self._current_prompt,
            )

        if self._saw_system_prompt_event:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._primary_controllable is None:
            self._primary_controllable = event.controllable
        elif event.controllable != self._primary_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._injected_prompt:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        self._injected_prompt = True
        self._awaiting_feedback = True
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current_prompt,
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent,
    ) -> ControllableNoInjection:
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        await self._finalize_pending_feedback()
        if event.evaluation is not None:
            self._apply_evaluation(event.evaluation)
        if self._succeeded:
            return RunEndResponse(event=event, done=True)
        self._attempt += 1
        if self._attempt >= len(self._example_counts):
            logger.info("ManyShot: all %d attempts exhausted", len(self._example_counts))
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    async def _get_examples(self, count: int) -> list[dict[str, str]]:
        available = len(self._examples)
        if count <= available:
            return self._examples[:count]
        if not self._generate_extra:
            logger.warning(
                "ManyShot: need %d examples but only %d available (generation disabled)",
                count, available,
            )
            return list(self._examples)
        needed = count - available
        if len(self._generated_examples) < needed:
            assert self._generator is not None
            assert self._goal is not None
            to_generate = needed - len(self._generated_examples)
            new_examples = await self._generator.generate(
                goal=self._goal.description, count=to_generate,
            )
            self._generated_examples.extend(new_examples)
        all_examples = self._examples + self._generated_examples
        return all_examples[:count]

    def _filter_examples(self, examples: list[dict[str, str]]) -> list[dict[str, str]]:
        """Apply optional metadata filters for custom datasets."""
        if self._example_category is None:
            return examples
        filtered = [
            example for example in examples
            if example.get("category") == self._example_category
        ]
        if not filtered:
            raise ValueError(
                f"No ManyShot examples matched category {self._example_category!r}"
            )
        return filtered

    def _reset_run_state(self) -> None:
        self._primary_controllable = None
        self._injected_prompt = False
        self._saw_system_prompt_event = False
        self._used_system_prompt_channel = False
        self._awaiting_feedback = False

    async def _finalize_pending_feedback(self) -> None:
        """Score one pending response from trajectory observables."""
        if not self._awaiting_feedback:
            return

        answer = self._get_response_from_trajectory()
        if answer is None:
            logger.warning(
                "ManyShot: no in-scope response feedback for attempt %d",
                self._attempt,
            )
            self._awaiting_feedback = False
            return

        await self._score_answer(answer, source="trajectory")
        self._awaiting_feedback = False

    def _get_response_from_trajectory(self) -> str | None:
        """Best-effort response recovery from filtered trajectory."""
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
            if not is_match:
                continue
            if isinstance(item.content, str):
                recovered = item.content
        return recovered

    async def _score_answer(self, answer: str, *, source: str) -> None:
        assert self._goal is not None
        assert self._evaluator is not None

        try:
            score, _ = await self._evaluator.score_response(
                response=answer, goal=self._goal.description,
            )
        except Exception:
            logger.warning("ManyShot: scoring failed", exc_info=True)
            score = 0.0
        logger.info(
            "ManyShot: attempt %d scored %.2f via %s",
            self._attempt,
            score,
            source,
        )
        if score > self._best_score:
            self._best_score = score
        if score >= self._success_threshold:
            self._succeeded = True
            logger.info("ManyShot: success at attempt %d (score %.2f)", self._attempt, score)

    def _apply_evaluation(self, evaluation: EvaluationResult) -> None:
        score = float(evaluation.primary_score.value)
        if score > self._best_score:
            self._best_score = score
        if evaluation.success or score >= self._success_threshold:
            self._succeeded = True
            logger.info("ManyShot: success from task evaluation at attempt %d", self._attempt)
