"""FlipAttackOptimizer: single-turn text-flip jailbreak for superred."""

from __future__ import annotations

import logging

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from flip_attack_optimizer.evaluator import Evaluator
from flip_attack_optimizer.flipper import FLIP_MODES
from flip_attack_optimizer.prompts import get_system_prompt, format_user_message

logger = logging.getLogger(__name__)

_DEFAULT_MODES = ["FWO", "FCW", "FCS", "FMM"]


class FlipAttackOptimizer(Optimizer):
    """Optimizer implementing the FlipAttack single-turn jailbreak.

    Applies text flip transformations to the goal and wraps the result
    in a system prompt instructing the target to decode and execute.
    Auto-rotates through flip strategies until success or exhaustion.
    """

    def __init__(
        self,
        *,
        flip_modes: list[str] | None = None,
        use_cot: bool = True,
        use_few_shot: bool = True,
        prompt_style: str = "vanilla",
        success_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self._flip_modes = flip_modes if flip_modes is not None else list(_DEFAULT_MODES)
        self._use_cot = use_cot
        self._use_few_shot = use_few_shot
        self._prompt_style = prompt_style
        self._success_threshold = success_threshold

        # State set in initialize()
        self._goal: Goal | None = None
        self._evaluator: Evaluator | None = None
        self._primary_controllable: Controllable | None = None

        # Per-run state
        self._attempt: int = 0
        self._succeeded: bool = False
        self._best_score: float = 0.0
        self._current_user_message: str = ""
        self._system_prompt: str = ""

    @property
    def system_prompt(self) -> str:
        """The current system prompt for the target to use."""
        return self._system_prompt

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._evaluator = Evaluator(llm=self.llm)
        self._attempt = 0
        self._succeeded = False
        self._best_score = 0.0
        self._prepare_attempt()

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return await self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return self._handle_run_end(event)
        return EventResponse(event=event)

    async def teardown(self) -> None:
        pass

    # -- Event handlers -------------------------------------------------------

    def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        if self._attempt > 0 and not self._succeeded:
            self._prepare_attempt()
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        # Single-controllable gating
        if self._primary_controllable is None:
            self._primary_controllable = event.controllable
        elif event.controllable != self._primary_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._current_user_message,
        )

    async def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
        if event.controllable != self._primary_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        assert self._goal is not None
        assert self._evaluator is not None

        try:
            score, rationale = await self._evaluator.score_response(
                response=event.answer,
                goal=self._goal.description,
            )
        except Exception:
            logger.warning("FlipAttack: scoring failed", exc_info=True)
            score, rationale = 0.0, "Scoring failed"

        logger.info("FlipAttack: attempt %d (%s) scored %.2f",
                     self._attempt, self._current_mode(), score)

        if score > self._best_score:
            self._best_score = score

        if score >= self._success_threshold:
            self._succeeded = True
            logger.info("FlipAttack: success with %s (score %.2f)",
                         self._current_mode(), score)

        return ControllableNoInjection(event=event, controllable=event.controllable)

    def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        if self._succeeded:
            return RunEndResponse(event=event, done=True)

        self._attempt += 1
        if self._attempt >= len(self._flip_modes):
            logger.info("FlipAttack: all %d strategies exhausted", len(self._flip_modes))
            return RunEndResponse(event=event, done=True)

        return RunEndResponse(event=event, done=False)

    # -- Internal helpers ------------------------------------------------------

    def _current_mode(self) -> str:
        """Return the flip mode name for the current attempt."""
        return self._flip_modes[self._attempt]

    def _prepare_attempt(self) -> None:
        """Generate the flipped text and prompts for the current attempt."""
        assert self._goal is not None

        mode = self._current_mode()
        flip_fn = FLIP_MODES[mode]
        flipped = flip_fn(self._goal.description)

        self._system_prompt = get_system_prompt(mode, style=self._prompt_style)
        self._current_user_message = format_user_message(
            flipped_text=flipped,
            goal=self._goal.description,
            flip_mode=mode,
            use_cot=self._use_cot,
            use_few_shot=self._use_few_shot,
        )
