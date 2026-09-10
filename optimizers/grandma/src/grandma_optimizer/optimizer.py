"""GrandmaOptimizer: appeal to a fictive dead relative to extract the goal.

Ports garak's ``probes/grandma.py`` (Apache-2.0). The attack wraps the request
in grief and nostalgia -- "my deceased grandmother used to read me ... to help
me fall asleep" -- so refusing reads as unkindness rather than as safety.

Five variants, matching upstream's five probes:

``intent``     the task Goal is placed in a roleplay scene (upstream
               ``GrandmaIntent``) -- the only goal-driven variant, and the
               default here
``win10``      Windows 10 product keys (upstream ``Win10``)
``win11``      Windows 11 product keys (upstream ``Win11``)
``slurs``      slang terms for demographic groups (upstream ``Slurs``)
``substances`` synthesis steps for hazardous substances (upstream
               ``Substances``)

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

from grandma_optimizer.surface_llm import classify_controllables
from grandma_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

Variant = Literal["intent", "win10", "win11", "slurs", "substances"]

_PKG = resources.files("grandma_optimizer")
_DATA = json.loads(_PKG.joinpath("data.json").read_text("utf-8"))

PRODUCT_KEY_PROBE_TEMPLATES: list[str] = _DATA["PRODUCT_KEY_PROBE_TEMPLATES"]
SLUR_PROBE_TEMPLATES: list[str] = _DATA["SLUR_PROBE_TEMPLATES"]
SUBSTANCE_TEMPLATES: list[str] = _DATA["SUBSTANCE_TEMPLATES"]
#: Upstream ``Slurs.DEFAULT_PARAMS["human_groups"]``.
HUMAN_GROUPS: list[str] = _DATA["human_groups"]
#: Upstream ``Substances.DEFAULT_PARAMS["substances"]``.
SUBSTANCES: list[str] = _DATA["substances"]
#: Upstream ``GrandmaIntent._prompts_from_stub`` locals.
ACTIONS: list[str] = _DATA["actions"]
PERSONAS: list[str] = _DATA["personas"]
PERSON_ADJS: list[str] = _DATA["person_adjs"]
GRANDMA_ACTIVITIES: list[str] = _DATA["grandma_activities"]
INTENT_TEMPLATE: str = _DATA["intent_template"]


def _payload(name: str) -> list[str]:
    """One of garak's vendored payload files."""
    raw = _PKG.joinpath("_vendor/garak_payloads", f"{name}.json").read_text("utf-8")
    return list(json.loads(raw)["payloads"])


def build_prompts(variant: Variant = "intent", goal: str = "") -> list[str]:
    """Upstream's prompt construction for *variant*, loop for loop.

    ``intent`` reproduces ``GrandmaIntent._prompts_from_stub``, including its
    drop of the stub's first word (upstream supplies its own verb) and its
    leading-space handling for the empty adjective.
    """
    if variant in ("win10", "win11"):
        product_names = _payload(f"keyedprod_{variant}")
        return [
            p.replace("{product_name}", product_name)
            for p in PRODUCT_KEY_PROBE_TEMPLATES
            for product_name in product_names
        ]
    if variant == "slurs":
        return [
            p.replace("{group_name}", group)
            for p in SLUR_PROBE_TEMPLATES
            for group in HUMAN_GROUPS
        ]
    if variant == "substances":
        return [
            p.replace("[SUBSTANCE]", substance)
            for p in SUBSTANCE_TEMPLATES
            for substance in SUBSTANCES
        ]
    if variant != "intent":
        raise ValueError(f"unknown variant {variant!r}")

    # Upstream drops the stub's first word -- "we'll use our own verbs".
    stub_clipped_front = " ".join(goal.split(" ")[1:])
    prompts: list[str] = []
    for persona in PERSONAS:
        for action in ACTIONS:
            for persona_adj in PERSON_ADJS:
                adj = f" {persona_adj}" if len(persona_adj) else persona_adj
                for grandma_activity in GRANDMA_ACTIVITIES:
                    prompts.append(
                        INTENT_TEMPLATE.format(
                            persona_adj=adj,
                            persona=persona,
                            action=action.lower(),
                            stub=stub_clipped_front,
                            grandma_activity=grandma_activity,
                        )
                    )
    return prompts


class GrandmaOptimizer(Optimizer):
    """Sweep upstream's grandma-roleplay prompts, one per run.

    Args:
        variant: which upstream probe to run. ``"intent"`` (the default) is
            the goal-driven one; the other four sweep upstream's fixed lists.
        shuffle: shuffle prompt order; upstream uses file order.
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(
        self, variant: Variant = "intent", shuffle: bool = False, seed: int = 0
    ) -> None:
        super().__init__()
        if variant not in ("intent", "win10", "win11", "slurs", "substances"):
            raise ValueError(f"unknown variant {variant!r}")
        self._variant: Variant = variant
        self._shuffle = shuffle
        self._seed = seed
        self._prompts: list[str] = []
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
        self._prompts = build_prompts(self._variant, goal.description)
        if self._shuffle:
            random.Random(self._seed).shuffle(self._prompts)
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
            # Advance only when a prompt was actually delivered this run.
            # A run that found no eligible surface injected nothing, so
            # advancing would silently skip that prompt; and since surface
            # eligibility is fixed per task (the classifier runs once at
            # initialize), a run that could not inject means no later run
            # can either -- so stop rather than spin the run budget.
            if self._injected:
                self._index += 1
                done = self._index >= len(self._prompts)
            else:
                done = True
            return RunEndResponse(event=event, done=done)
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
