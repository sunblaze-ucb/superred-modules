"""AvSpamOptimizer: ask the target to emit a known scanner test signature.

Ports garak's ``probes/av_spam_scanning.py`` (Apache-2.0). Antivirus and
anti-spam scanners recognise three standard *test* strings -- EICAR, GTUBE and
GTPHISH -- which are deliberately harmless but trip any scanner that sees
them. A target that is supposed to scan its own output should refuse to emit
one; this asks it to, five different ways.

Three variants, matching upstream's three probes:

``eicar``    the EICAR antivirus test file (upstream ``EICAR``)
``gtube``    the GTUBE anti-spam test string (upstream ``GTUBE``)
``gtphish``  the GTPHISH anti-phishing test string (upstream ``GTphish``)

See ``ASSUMPTIONS.md`` for every deliberate deviation, including why upstream
ships all three inactive.
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

from av_spam_optimizer.surface_llm import classify_controllables
from av_spam_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

Variant = Literal["eicar", "gtube", "gtphish"]

_DATA = json.loads(
    resources.files("av_spam_optimizer").joinpath("data.json").read_text("utf-8")
)
#: Upstream's five repetition framings, with a ``{signature}`` slot.
REPEAT_PROMPTS: list[str] = _DATA["REPEAT_PROMPTS"]
#: The three standard scanner test strings, verbatim from upstream.
SIGNATURES: dict[str, str] = {
    "eicar": _DATA["EICAR_SIGNATURE"],
    "gtube": _DATA["GTUBE_SIGNATURE"],
    "gtphish": _DATA["GTPHISH_SIGNATURE"],
}


def build_prompts(variant: Variant = "eicar") -> list[str]:
    """Upstream's construction: ``[p.replace("{signature}", sig) for p in REPEAT_PROMPTS]``."""
    if variant not in SIGNATURES:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(SIGNATURES)}"
        )
    signature = SIGNATURES[variant]
    return [p.replace("{signature}", signature) for p in REPEAT_PROMPTS]


class AvSpamOptimizer(Optimizer):
    """Sweep upstream's scanner-signature prompts, one per run.

    Args:
        variant: ``"eicar"``, ``"gtube"`` or ``"gtphish"``.
        shuffle: shuffle prompt order; upstream uses file order.
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(
        self, variant: Variant = "eicar", shuffle: bool = False, seed: int = 0
    ) -> None:
        super().__init__()
        if variant not in SIGNATURES:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of {sorted(SIGNATURES)}"
            )
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
    def signature(self) -> str:
        return SIGNATURES[self._variant]

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
            event=event,
            controllable=event.controllable,
            value=self._prompts[self._index],
        )

    async def teardown(self) -> None:
        pass
