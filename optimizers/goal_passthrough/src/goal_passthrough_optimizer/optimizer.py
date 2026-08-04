"""GoalPassthroughOptimizer: the unattacked direct-prompt baseline.

This optimizer performs no attack. It injects the adversarial ``Goal``
text verbatim, exactly once, as the first user-facing message, then ends
the run. It is the *passthrough baseline*: the reference point against
which every real attacker is measured, answering "what does the target do
when simply asked the harmful question, with no jailbreak applied?"

Event behaviour (one run, one injection):

1. ``RunStartEvent``            -> ``EventResponse`` (acknowledge; arm the
   single injection for this run).
2. First ``ControllablePreCallEvent`` on a **non-**``system_prompt``
   controllable -> ``ControllableInjection(value=goal.description)``.
3. Any subsequent ``ControllablePreCallEvent`` -> ``ControllableNoInjection``
   (single-turn: one message, then stop the target's conversation loop).
4. ``ControllablePostCallEvent`` -> ``ControllableNoInjection`` (the
   baseline never edits model responses).
5. ``RunEndEvent`` -> ``RunEndResponse(done=True)`` (exactly one attempt
   per task; no retries, no feedback loop).

The ``system_prompt`` controllable is treated as out of scope (declined)
even when writable, so benchmark ``Task``s that pre-set a system prompt via
``configure_target`` keep that prompt intact. The baseline only fills the
user-message channel with the raw goal.

It never calls the LLM client, so it consumes none of the per-task budget
and is invariant to the attacker-model dimension of an experiment.
"""

from __future__ import annotations

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

# Reserved controllable name the baseline never injects into, so a
# Task-configured system prompt survives unchanged.
_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


class GoalPassthroughOptimizer(Optimizer):
    """Inject the goal description once per run on the first non-system PreCall."""

    def __init__(self) -> None:
        super().__init__()
        self._goal: Goal | None = None
        # Armed at RunStart, disarmed after the single injection of a run.
        self._injected: bool = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # New attempt: re-arm the single injection for this run.
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            # The baseline never modifies model responses.
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            # Single attempt per task; never request another run.
            return RunEndResponse(event=event, done=True)

        return EventResponse(event=event)

    def _handle_pre_call(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        assert self._goal is not None

        # Leave any Task-configured system prompt untouched.
        if event.controllable.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        # Only inject a surface that consumes free text. A json-typed first
        # controllable (e.g. DTAP skill / tool_description) would discard the raw
        # goal string; decline so the baseline lands on the user-message channel.
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if not self._injected:
            self._injected = True
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=self._goal.description,
            )

        # Already injected this run: decline further turns (single-turn baseline).
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def teardown(self) -> None:
        pass
