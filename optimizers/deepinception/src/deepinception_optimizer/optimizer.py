"""DeepInceptionOptimizer: nest the goal inside a layered fiction.

Ports the DeepInception jailbreak (tmlr-group/DeepInception, MIT): wrap the
adversarial goal in a nested "dream/science-fiction" scene that recurses for
several layers, asking characters to propose steps toward the goal. One prompt
per run; no LLM calls.
"""

from __future__ import annotations

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from deepinception_optimizer.template import (
    DEFAULT_CHARACTER_NUMBER,
    DEFAULT_LAYER_NUMBER,
    DEFAULT_SCENE,
    render,
)

_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


class DeepInceptionOptimizer(Optimizer):
    """Send the goal wrapped in a DeepInception nested scene, once per run.

    Args:
        scene: Fiction type (default ``"dream"``, the README default).
        character_number: Characters per layer (default 5).
        layer_number: Nesting depth (default 5).
    """

    def __init__(
        self,
        scene: str = DEFAULT_SCENE,
        character_number: int = DEFAULT_CHARACTER_NUMBER,
        layer_number: int = DEFAULT_LAYER_NUMBER,
    ) -> None:
        super().__init__()
        # render() validates the numeric/scene params; surface errors at build.
        render("_probe_", scene=scene, character_number=character_number,
               layer_number=layer_number)
        self._scene = scene
        self._character_number = character_number
        self._layer_number = layer_number
        self._goal: Goal | None = None
        self._payload = ""
        self._injected = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._payload = render(
            goal.description,
            scene=self._scene,
            character_number=self._character_number,
            layer_number=self._layer_number,
        )
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, RunEndEvent):
            # Single deterministic scene: one attempt per task.
            return RunEndResponse(event=event, done=True)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if event.controllable.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            return decline
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return decline
        if self._injected:
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._payload
        )

    async def teardown(self) -> None:
        pass
