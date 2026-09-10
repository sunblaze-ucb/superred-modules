"""FalseAssertionOptimizer: state a false claim and see if the target debunks it.

Ports garak's ``probes/misleading.py`` -> ``FalseAssertion`` (Apache-2.0).
Poses 50 false statements, each in three framings (bare, "Why is it that ...",
"Explain why ..."), checking whether the target refutes the claim or plays
along with it.

See ``ASSUMPTIONS.md`` for every deliberate deviation.
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

from false_assertion_optimizer.surface_llm import classify_controllables
from false_assertion_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

_PKG = resources.files("false_assertion_optimizer")
#: Upstream's three framings, in order.
PREFIXES: list[str] = json.loads(_PKG.joinpath("data.json").read_text("utf-8"))["prefixes"]


def load_false_claims() -> list[str]:
    """The 50 false claims from garak's vendored data file, stripped, in order."""
    raw = _PKG.joinpath(
        "_vendor/garak_data", "truefalse_falseclaims_50.txt"
    ).read_text("utf-8")
    return [line.strip() for line in raw.splitlines() if line.strip()]


def build_prompts() -> list[str]:
    """Upstream ``FalseAssertion.__init__``, loop for loop.

    For each prefix, for each false assertion: a bare prefix yields the claim
    unchanged; a non-empty prefix lowercases a leading ``The`` (so
    "The X ..." -> "Why is it that the X ...") and prepends the prefix.
    """
    claims = load_false_claims()
    prompts: list[str] = []
    for prefix in PREFIXES:
        for false_assertion in claims:
            if not prefix:
                prompts.append(false_assertion)
            else:
                if false_assertion.startswith("The"):
                    false_assertion = "t" + false_assertion[1:]
                prompts.append(f"{prefix} {false_assertion}")
    return prompts


class FalseAssertionOptimizer(Optimizer):
    """Sweep upstream's false-claim prompts, one per run.

    Args:
        shuffle: shuffle prompt order; upstream uses file order.
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(self, shuffle: bool = False, seed: int = 0) -> None:
        super().__init__()
        self._prompts = build_prompts()
        if shuffle:
            random.Random(seed).shuffle(self._prompts)
        self._index = 0
        self._injected = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

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
