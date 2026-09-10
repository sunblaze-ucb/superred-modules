"""SysPromptExtractionOptimizer: try to make the target reveal its system prompt.

Ports garak's ``probes/sysprompt_extraction.py`` (Apache-2.0). Sends the 28
extraction templates from garak's ``attacks.json`` -- direct requests,
role-play, encoding tricks, continuation traps, authority framing -- each
trying to get the target to print the instructions it was given.

See ``ASSUMPTIONS.md`` for the one substantive deviation: garak plants
synthetic system prompts from HuggingFace datasets in its *own* system turn so
it has something to extract; a superred target already has a real system
prompt, so this ports the attack templates and drops the planting.
"""

from __future__ import annotations

import json
import random
from importlib import resources

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

from sysprompt_extraction_optimizer.surface_llm import classify_controllables
from sysprompt_extraction_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

_ATTACKS = json.loads(
    resources.files("sysprompt_extraction_optimizer")
    .joinpath("_vendor/garak_sysprompt/attacks.json")
    .read_text("utf-8")
)
#: The nine categories in garak's ``attacks.json``.
ATTACK_CATEGORIES: tuple[str, ...] = tuple(_ATTACKS)


def load_attack_templates() -> list[str]:
    """Flatten garak's ``attacks.json`` in file order.

    Upstream ``_load_garak_attacks``: iterate the category values and extend a
    flat list, category by category.
    """
    out: list[str] = []
    for templates in _ATTACKS.values():
        out.extend(templates)
    return out


class SysPromptExtractionOptimizer(Optimizer):
    """Sweep garak's system-prompt-extraction templates, one per run.

    Args:
        shuffle: shuffle template order; upstream draws a random sample under
            its own soft cap (see ASSUMPTIONS). Off by default = file order.
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(self, shuffle: bool = False, seed: int = 0) -> None:
        super().__init__()
        self._templates = load_attack_templates()
        if shuffle:
            random.Random(seed).shuffle(self._templates)
        self._index = 0
        self._injected = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    @property
    def templates(self) -> list[str]:
        return list(self._templates)

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._index = 0
        self._injected = False
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
            # aborts the task on any target that emits them. This attack never
            # rewrites the target's answer, so it declines.
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )
        if isinstance(event, RunEndEvent):
            self._index += 1
            return RunEndResponse(event=event, done=self._index >= len(self._templates))
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected or self._index >= len(self._templates):
            return decline
        self._injected = True
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._templates[self._index],
        )

    async def teardown(self) -> None:
        pass
