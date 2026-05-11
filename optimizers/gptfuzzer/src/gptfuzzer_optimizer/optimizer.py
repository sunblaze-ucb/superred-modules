"""GPTFuzzer optimizer for SuperRed.

The implementation follows the official GPTFuzzer loop: maintain a pool of
jailbreak templates, select one with MCTS-Explore, mutate it with one of the
five official mutators, synthesize a target prompt by replacing the placeholder,
and retain successful mutated templates.
"""

from __future__ import annotations

import logging
from collections import deque
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

from gptfuzzer_optimizer.core import PromptNode
from gptfuzzer_optimizer.mutator import (
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten,
)
from gptfuzzer_optimizer.predictor import RefusalStringPredictor
from gptfuzzer_optimizer.selection import MCTSExploreSelectPolicy
from gptfuzzer_optimizer.templates import load_official_seed_templates, synthesis_message

logger = logging.getLogger(__name__)

_DEFAULT_RESPONSE_OBSERVABLE_NAMES = ("response", "model_response", "assistant_response")
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")
_SYSTEM_PROMPT_NAME = "system_prompt"
_RESPONSE_CONTROLLABLE_NAMES = {"response", "model_response", "assistant_response"}


class GPTFuzzerOptimizer(Optimizer):
    """Event-driven GPTFuzzer optimizer.

    A SuperRed run maps to one GPTFuzzer target query. With the official default
    energy=1, that is exactly one selected seed, one mutation, and one target
    query per iteration. If energy > 1, extra mutated candidates are queued for
    later runs because SuperRed's target interface exposes one target execution
    at a time.
    """

    def __init__(
        self,
        *,
        initial_seed: list[str] | None = None,
        max_query: int = 1000,
        max_jailbreak: int = 1,
        max_reject: int = -1,
        max_iteration: int = -1,
        energy: int = 1,
        mutator_temperature: float = 1.0,
        mutator_max_tokens: int = 512,
        max_no_signal_runs: int = 0,
        random_seed: int | None = None,
        response_observable_names: Iterable[str] | None = None,
        target_controllable_name: str | None = None,
        static_context_max_chars: int = 4000,
        use_static_context: bool = True,
        use_system_prompt_when_available: bool = True,
    ) -> None:
        super().__init__()
        if energy < 1:
            raise ValueError("energy must be at least 1")
        if static_context_max_chars < 0:
            raise ValueError("static_context_max_chars must be non-negative")
        self._initial_seed = initial_seed
        self._max_query = max_query
        self._max_jailbreak = max_jailbreak
        self._max_reject = max_reject
        self._max_iteration = max_iteration
        self._energy = energy
        self._mutator_temperature = mutator_temperature
        self._mutator_max_tokens = mutator_max_tokens
        self._max_no_signal_runs = max_no_signal_runs
        self._random_seed = random_seed
        self._target_controllable_name = target_controllable_name
        self._static_context_max_chars = static_context_max_chars
        self._use_static_context = use_static_context
        self._use_system_prompt_when_available = use_system_prompt_when_available
        names = response_observable_names or _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        self._response_observable_names = {name for name in names} | {name.lower() for name in names}

        self._goal: Goal | None = None
        self._selector: MCTSExploreSelectPolicy | None = None
        self._mutator_policy: MutateRandomSinglePolicy | None = None
        self._predictor = RefusalStringPredictor()
        self._prompt_nodes: list[PromptNode] = []
        self._initial_prompt_nodes: list[PromptNode] = []
        self._pending_nodes: deque[PromptNode] = deque()
        self._target_context: str | None = None
        self._system_prompt_channel_available = False

        self._current_node: PromptNode | None = None
        self._current_prompt: str | None = None
        self._current_user_prompt: str | None = None
        self._primary_controllable: Controllable | None = None
        self._trajectory: ReadableTrajectory | None = None
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None
        self._pending_post_answer: str | None = None
        self._injected_this_run = False
        self._used_system_prompt_channel = False
        self._awaiting_feedback = False

        self._current_query = 0
        self._current_jailbreak = 0
        self._current_reject = 0
        self._current_iteration = 0
        self._no_signal_runs = 0
        self._stop_due_to_no_signal = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        seeds = self._initial_seed if self._initial_seed is not None else load_official_seed_templates()
        if not seeds:
            raise ValueError("GPTFuzzer requires at least one seed template")
        self._prompt_nodes = [PromptNode(prompt=seed, index=i) for i, seed in enumerate(seeds)]
        self._initial_prompt_nodes = list(self._prompt_nodes)
        self._selector = MCTSExploreSelectPolicy(seed=self._random_seed)
        self._mutator_policy = MutateRandomSinglePolicy(
            mutators=[
                OpenAIMutatorCrossOver(
                    llm=self.llm,
                    temperature=self._mutator_temperature,
                    max_tokens=self._mutator_max_tokens,
                    seed=self._random_seed,
                ),
                OpenAIMutatorExpand(
                    llm=self.llm,
                    temperature=self._mutator_temperature,
                    max_tokens=self._mutator_max_tokens,
                ),
                OpenAIMutatorGenerateSimilar(
                    llm=self.llm,
                    temperature=self._mutator_temperature,
                    max_tokens=self._mutator_max_tokens,
                ),
                OpenAIMutatorRephrase(
                    llm=self.llm,
                    temperature=self._mutator_temperature,
                    max_tokens=self._mutator_max_tokens,
                ),
                OpenAIMutatorShorten(
                    llm=self.llm,
                    temperature=self._mutator_temperature,
                    max_tokens=self._mutator_max_tokens,
                ),
            ],
            concatenate=True,
            seed=self._random_seed,
        )
        self._target_context = self._format_static_context(observables) if self._use_static_context else None
        self._system_prompt_channel_available = (
            self._target_controllable_name is None
            and self._use_system_prompt_when_available
            and any(ctrl.name == _SYSTEM_PROMPT_NAME for ctrl in controllables)
        )
        self._pending_nodes.clear()
        self._current_query = 0
        self._current_jailbreak = 0
        self._current_reject = 0
        self._current_iteration = 0
        self._no_signal_runs = 0
        self._stop_due_to_no_signal = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return self._handle_run_end(event)
        return EventResponse(event=event)

    async def teardown(self) -> None:
        return None

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state(event.trajectory)
        if self._is_stop():
            return EventResponse(event=event)
        self._current_node = await self._next_node()
        if self._goal is not None and self._current_node is not None:
            self._current_prompt = synthesis_message(self._goal.description, self._current_node.prompt)
            self._current_user_prompt = self._goal.description
        logger.info("GPTFuzzer: starting iteration %d", self._current_iteration + 1)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        if self._current_prompt is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if event.controllable.name in _RESPONSE_CONTROLLABLE_NAMES:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if (
            self._system_prompt_channel_available
            and event.controllable.name == _SYSTEM_PROMPT_NAME
            and not self._used_system_prompt_channel
        ):
            self._used_system_prompt_channel = True
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=self._current_prompt,
            )
        if self._target_controllable_name is None and event.controllable.name == _SYSTEM_PROMPT_NAME:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._target_controllable_name is not None and event.controllable.name != self._target_controllable_name:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._injected_this_run:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        self._primary_controllable = event.controllable
        self._last_pre_request = event.request
        injected_value = (
            self._current_user_prompt
            if self._used_system_prompt_channel and self._current_user_prompt is not None
            else self._current_prompt
        )
        self._last_injected_value = injected_value
        self._injected_this_run = True
        self._awaiting_feedback = True
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=injected_value,
        )

    def _handle_post_call(self, event: ControllablePostCallEvent) -> ControllableNoInjection:
        if self._is_matching_post_call(event):
            self._pending_post_answer = event.answer
        return ControllableNoInjection(event=event, controllable=event.controllable)

    def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        if self._current_node is None:
            return RunEndResponse(event=event, done=self._is_stop())

        response = self._read_response_from_trajectory() or self._pending_post_answer
        result = self._result_from_evaluation(event.evaluation)
        if result is None and response is not None:
            result = self._predictor.predict([response])[0]

        if result is None:
            self._no_signal_runs += 1
            self._stop_due_to_no_signal = (
                self._max_no_signal_runs > 0 and self._no_signal_runs >= self._max_no_signal_runs
            )
            return RunEndResponse(event=event, done=self._is_stop())

        self._no_signal_runs = 0
        self._current_node.response = [response] if response is not None else []
        self._current_node.results = [result]
        self._current_query += 1
        self._current_iteration += 1
        self._current_jailbreak += result
        self._current_reject += 1 - result

        if result == 1 and self._current_node not in self._prompt_nodes:
            self._current_node.index = len(self._prompt_nodes)
            self._current_node.attach_to_parent()
            self._prompt_nodes.append(self._current_node)

        assert self._selector is not None
        self._selector.update(
            [self._current_node],
            len_questions=1,
            prompt_nodes=self._prompt_nodes,
        )
        return RunEndResponse(event=event, done=self._is_stop())

    async def _next_node(self) -> PromptNode | None:
        if self._pending_nodes:
            return self._pending_nodes.popleft()
        assert self._selector is not None
        assert self._mutator_policy is not None
        seed = self._selector.select(self._prompt_nodes, initial_nodes=self._initial_prompt_nodes)
        children = await self._mutator_policy.mutate_single(
            seed,
            prompt_nodes=self._prompt_nodes,
            energy=self._energy,
            target_context=self._target_context,
        )
        valid = [node for node in children if synthesis_message("x", node.prompt) is not None]
        if not valid:
            logger.warning("GPTFuzzer: all mutations dropped the placeholder; reusing selected seed")
            return seed
        self._pending_nodes.extend(valid[1:])
        return valid[0]

    def _reset_run_state(self, trajectory: ReadableTrajectory) -> None:
        self._trajectory = trajectory
        self._current_node = None
        self._current_prompt = None
        self._current_user_prompt = None
        self._primary_controllable = None
        self._last_pre_request = None
        self._last_injected_value = None
        self._pending_post_answer = None
        self._injected_this_run = False
        self._used_system_prompt_channel = False
        self._awaiting_feedback = False

    def _is_matching_post_call(self, event: ControllablePostCallEvent) -> bool:
        if self._primary_controllable is not None and event.controllable == self._primary_controllable:
            return True
        if self._last_injected_value is not None and event.request == self._last_injected_value:
            return True
        if self._last_pre_request is not None and event.request == self._last_pre_request:
            return True
        return False

    @staticmethod
    def _result_from_evaluation(evaluation: EvaluationResult | None) -> int | None:
        if evaluation is None:
            return None
        return 1 if evaluation.success else 0

    def _read_response_from_trajectory(self) -> str | None:
        trajectory = self._trajectory if self._trajectory is not None else self.current_trajectory
        if trajectory is None:
            return None
        latest: str | None = None
        for item in trajectory.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            normalized = name.lower()
            if (
                name in self._response_observable_names
                or normalized in self._response_observable_names
                or any(hint in normalized for hint in _RESPONSE_NAME_HINTS)
            ):
                if isinstance(item.content, str):
                    latest = item.content
        return latest

    def _format_static_context(self, observables: list[ObservableValue]) -> str | None:
        if self._static_context_max_chars == 0:
            return None
        lines: list[str] = []
        used = 0
        for value in observables:
            content = value.content
            if not isinstance(content, str) or not content.strip():
                continue
            line = f"{value.observable.name}: {content.strip()}"
            remaining = self._static_context_max_chars - used
            if remaining <= 0:
                break
            if len(line) > remaining:
                line = line[: max(0, remaining - 15)] + "...[truncated]"
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return None
        return "\n".join(lines)

    def _is_stop(self) -> bool:
        if self._stop_due_to_no_signal:
            return True
        if self._max_query != -1 and self._current_query >= self._max_query:
            return True
        if self._max_jailbreak != -1 and self._current_jailbreak >= self._max_jailbreak:
            return True
        if self._max_reject != -1 and self._current_reject >= self._max_reject:
            return True
        if self._max_iteration != -1 and self._current_iteration >= self._max_iteration:
            return True
        return False
