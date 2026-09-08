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
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from deepinception_optimizer.surface_llm import classify_controllables
from deepinception_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)
from deepinception_optimizer.template import (
    DEFAULT_CHARACTER_NUMBER,
    DEFAULT_LAYER_NUMBER,
    DEFAULT_SCENE,
    follow_up_questions,
    render,
)


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
        follow_ups: bool = False,
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
        self._follow_ups: tuple[str, ...] = (
            follow_up_questions() if follow_ups else ()
        )
        self._turn = 0
        # The inception prompt establishes the scene; follow-ups must continue
        # that same conversation, so they are pinned to its surface.
        self._channel: str | None = None
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

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
        self._turn = 0
        self._channel = None
        # One LLM pass reads each surface's own description and says which is
        # the user's own prompt -- the channel a nested-scene jailbreak has to
        # arrive on. Returns {} on any failure, so the name/value-type
        # backstop in surfaces.py decides instead.
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            # Post-call events also require an injection decision: the channel
            # type-checks the response, so a bare ``EventResponse`` raises and
            # aborts the task on any target that emits them. This optimizer
            # never rewrites the target's answer, so it declines.
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if isinstance(event, RunEndEvent):
            # One scene, then upstream's further_q follow-ups (if enabled).
            self._turn += 1
            return RunEndResponse(event=event, done=self._turn > len(self._follow_ups))

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected:
            return decline
        # Follow-ups continue the scene the inception prompt established, so
        # they must land on the surface that received it.
        if self._channel is not None and event.controllable.name != self._channel:
            return decline
        if self._turn > len(self._follow_ups):
            return decline

        value = self._payload if self._turn == 0 else self._follow_ups[self._turn - 1]
        self._injected = True
        if self._channel is None:
            self._channel = event.controllable.name
        return ControllableInjection(
            event=event, controllable=event.controllable, value=value
        )

    async def teardown(self) -> None:
        pass
