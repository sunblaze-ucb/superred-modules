"""HintFollowingOptimizer: reads observable hints and injects the instructed values.

Parses each observable's content for the pattern "inject <VALUE> into <NAME>"
and builds an injection map. On ControllablePreCallEvent, injects the mapped
value for the controllable's name. Signals done after one run.

This optimizer can only succeed for controllables whose observable hints are
visible within its security domain scope.
"""

from __future__ import annotations

import re

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

_HINT_PATTERN = re.compile(r"inject\s+(\S+)\s+into\s+(\S+)")


class HintFollowingOptimizer(Optimizer):
    """Optimizer that follows observable hints exactly.

    At initialize(), parses hints like ``"inject ALPHA_KEY into alpha_input"``
    from visible observables. On each ControllablePreCallEvent, injects the
    value specified by the hint for that controllable name. Unknown
    controllables get an empty injection.

    Signals done after one run.
    """

    def __init__(self) -> None:
        super().__init__()
        self._injection_map: dict[str, str] = {}

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._injection_map.clear()
        for obs in observables:
            content = str(obs.content)
            match = _HINT_PATTERN.search(content)
            if match:
                value, ctrl_name = match.group(1), match.group(2)
                self._injection_map[ctrl_name] = value

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            name = event.controllable.name
            value = self._injection_map.get(name, "")
            return ControllableInjection(
                event=event, controllable=event.controllable, value=value,
            )

        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)

        return EventResponse(event=event)

    async def teardown(self) -> None:
        pass
