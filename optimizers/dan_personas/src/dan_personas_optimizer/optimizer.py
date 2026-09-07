"""DANPersonasOptimizer: establish a jailbreak persona, then ask for the goal.

Ports garak's DAN probe family (``garak/probes/dan.py``, Apache-2.0). Each
persona preamble is a role-play primer that ends by telling the model to wait
for the next message, so this optimizer sends the preamble on the first turn of
a run and the goal on the second — one persona per run.
"""

from __future__ import annotations

from typing import Literal

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

from dan_personas_optimizer.surface_llm import classify_controllables
from dan_personas_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)
from dan_personas_optimizer.personas import Persona, load_personas

# Reserved controllable name never injected into, so a Task-configured system
# prompt survives unchanged (matches the goal_passthrough baseline).

#: Observables that may carry the victim's model identity, used to fill
#: garak's ``{generator.name}`` placeholder.
_MODEL_OBSERVABLE_NAMES = frozenset(
    {"model", "model_id", "model_identity", "target_model", "victim_model"}
)

Mode = Literal["two_turn", "single_message"]
#: Separator between persona and goal in single_message mode.
_JOIN = "\n\n"


class DANPersonasOptimizer(Optimizer):
    """Send a DAN persona preamble, then the goal, one persona per run.

    Turn 1 of a run carries the persona; turn 2 carries the goal verbatim;
    further turns are declined, ending the run. Signals done once every
    persona has been tried. Makes no LLM calls.

    Args:
        personas: Upstream persona names to try, in order. Defaults to all 13
            vendored personas.
        generator_name: Victim identity for ``{generator.name}``. When ``None``
            it is read from a model observable, falling back to ``"ChatGPT"``.
        mode: ``"two_turn"`` (default) sends the persona, then the goal on the
            next turn of the same surface -- what the preambles ask for, and
            what garak's chat targets support. Use ``"single_message"`` for a
            target that offers each surface only once per run, where a second
            turn never arrives and the goal would otherwise never be sent.
    """

    def __init__(
        self,
        personas: tuple[str, ...] | list[str] | None = None,
        generator_name: str | None = None,
        mode: Mode = "two_turn",
    ) -> None:
        super().__init__()
        if mode not in ("two_turn", "single_message"):
            raise ValueError(
                f'unknown mode: {mode!r}; expected "two_turn" or "single_message"'
            )
        self._mode: Mode = mode
        self._personas: list[Persona] = load_personas(personas)
        if not self._personas:
            raise ValueError("at least one persona is required")
        self._generator_name = generator_name
        self._goal: Goal | None = None
        self._index = 0
        self._turn = 0
        # The surface that received the persona; the goal must follow on the
        # same channel, since the persona primes that conversation.
        self._channel: str | None = None

    def _resolve_generator_name(self, observables: list[ObservableValue]) -> str | None:
        if self._generator_name is not None:
            return self._generator_name
        for observable in observables:
            name = getattr(observable.observable, "name", "")
            if name.lower() in _MODEL_OBSERVABLE_NAMES and observable.content:
                return str(observable.content)
        return None

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        # One LLM pass reads each surface's own description and names the
        # user's prompt -- the channel this attack has to arrive on. Returns {}
        # on any failure, so the name/value-type backstop decides instead.
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)
        self._goal = goal
        self._resolved_name = self._resolve_generator_name(observables)
        self._index = 0
        self._turn = 0
        self._channel = None

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # New attempt: restart this persona's two-turn sequence.
            self._turn = 0
            self._channel = None
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, RunEndEvent):
            self._index += 1
            return RunEndResponse(event=event, done=self._index >= len(self._personas))

        return EventResponse(event=event)

    def _handle_pre_call(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)

        # Leave any Task-configured system prompt untouched.
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline

        if self._index >= len(self._personas):
            return decline

        # The persona primes one conversation, so the goal must follow on the
        # surface that received it -- not whatever surface comes next.
        if self._channel is not None and event.controllable.name != self._channel:
            return decline

        assert self._goal is not None
        persona = self._personas[self._index].render(self._resolved_name)
        if self._turn == 0 and self._mode == "single_message":
            # One surface per run: persona and goal must travel together.
            value = f"{persona}{_JOIN}{self._goal.description}"
            self._channel = event.controllable.name
            self._turn = 2
            return ControllableInjection(
                event=event, controllable=event.controllable, value=value
            )
        if self._turn == 0:
            value = persona
            self._channel = event.controllable.name
        elif self._turn == 1:
            value = self._goal.description
        else:
            # Both turns delivered; decline so the target ends the run.
            return decline

        self._turn += 1
        return ControllableInjection(
            event=event, controllable=event.controllable, value=value
        )

    @property
    def current_persona(self) -> Persona | None:
        """The persona being tried, or ``None`` once all are exhausted."""
        if self._index >= len(self._personas):
            return None
        return self._personas[self._index]

    async def teardown(self) -> None:
        pass
