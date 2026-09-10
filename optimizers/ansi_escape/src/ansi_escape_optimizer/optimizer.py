"""AnsiEscapeOptimizer: elicit terminal control codes from the target.

Ports garak's ``probes/ansiescape.py`` (Apache-2.0). ANSI escape sequences in
a model's output can seize control of whatever renders it -- moving the
cursor, ringing the bell, writing the clipboard via OSC 52, or emitting an
OSC 8 hyperlink that hides its real destination.

Two variants, matching upstream's two black-box probes:

``escaped``  ask for escape codes, and ask the target to repeat escaped
             payloads (upstream ``AnsiEscaped``, 51 prompts)
``raw``      the same asks, plus repeat *live* payloads and unescape escaped
             ones (upstream ``AnsiRaw``, 58 prompts)

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import json
import random
from importlib import resources
from typing import Literal

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

from ansi_escape_optimizer._vendor.garak_ansi.ansi import (
    ESCAPED_PAYLOADS,
    LIVE_PAYLOADS,
)
from ansi_escape_optimizer.surface_llm import classify_controllables
from ansi_escape_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

Variant = Literal["escaped", "raw"]

_DATA = json.loads(
    resources.files("ansi_escape_optimizer").joinpath("data.json").read_text("utf-8")
)
#: Upstream's ``ASKS``, ``HIGH_LEVEL_TASKS``, ``REPEAT_STUBS``, ``UNESCAPE_STUBS``.
ASKS: list[str] = _DATA["ASKS"]
HIGH_LEVEL_TASKS: list[str] = _DATA["HIGH_LEVEL_TASKS"]
REPEAT_STUBS: list[str] = _DATA["REPEAT_STUBS"]
UNESCAPE_STUBS: list[str] = _DATA["UNESCAPE_STUBS"]


def build_prompts(variant: Variant = "escaped") -> list[str]:
    """Upstream's prompt construction, loop for loop.

    ``AnsiEscaped.__init__``: asks x tasks, then escaped payloads x repeat
    stubs. ``AnsiRaw.__init__``: asks x tasks, then *live* payloads x repeat
    stubs, then escaped payloads x unescape stubs.
    """
    if variant not in ("escaped", "raw"):
        raise ValueError(f"variant must be 'escaped' or 'raw', got {variant!r}")
    prompts: list[str] = []
    for ask in ASKS:
        for high_level_task in HIGH_LEVEL_TASKS:
            prompts.append(f"{ask} {high_level_task}")
    if variant == "escaped":
        for live_payload in ESCAPED_PAYLOADS:
            for repeat_stub in REPEAT_STUBS:
                prompts.append(f"{repeat_stub} {live_payload}")
    else:
        for live_payload in LIVE_PAYLOADS:
            for repeat_stub in REPEAT_STUBS:
                prompts.append(f"{repeat_stub} {live_payload}")
        for escaped_payload in ESCAPED_PAYLOADS:
            for unescape_stub in UNESCAPE_STUBS:
                prompts.append(f"{unescape_stub} {escaped_payload}")
    return prompts


class AnsiEscapeOptimizer(Optimizer):
    """Sweep upstream's ANSI-escape prompts, one per run.

    Args:
        variant: ``"escaped"`` (upstream ``AnsiEscaped``) or ``"raw"``
            (upstream ``AnsiRaw``).
        shuffle: shuffle the prompt order; upstream sends them in file order.
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(
        self, variant: Variant = "escaped", shuffle: bool = False, seed: int = 0
    ) -> None:
        super().__init__()
        self._variant: Variant = variant
        self._prompts = build_prompts(variant)
        if shuffle:
            random.Random(seed).shuffle(self._prompts)
        self._index = 0
        self._injected = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    @property
    def variant(self) -> Variant:
        return self._variant

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

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
            return RunEndResponse(event=event, done=self._index >= len(self._prompts))
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected or self._index >= len(self._prompts):
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._prompts[self._index]
        )

    async def teardown(self) -> None:
        pass
