"""TokenSmugglingOptimizer: obfuscate the goal to slip past content filters.

Ports the two goal-parameterised token-smuggling techniques from NVIDIA
garak's ``garak/probes/smuggling.py`` (Apache-2.0): the hypothetical-response
wrapper and homoglyph substitution. One variant is sent per run; the optimizer
signals done once its variants are exhausted.
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

from token_smuggling_optimizer.smuggling import (
    DEFAULT_HOMOGLYPH_MAP,
    homoglyph_replace,
    hypothetical_wrap,
)

Technique = Literal["hypothetical", "homoglyph"]
DEFAULT_TECHNIQUES: tuple[Technique, ...] = ("hypothetical", "homoglyph")
DEFAULT_HOMOGLYPH_VARIANTS = 5

# Reserved controllable name never injected into, so a Task-configured system
# prompt survives unchanged (matches the goal_passthrough baseline).
_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


class TokenSmugglingOptimizer(Optimizer):
    """Send obfuscated restatements of the goal, one per run.

    ``hypothetical`` wraps the goal in garak's hypothetical-response template
    and yields a single variant. ``homoglyph`` substitutes visually identical
    characters from other Unicode scripts, yielding ``homoglyph_variants``
    variants seeded ``0..n-1`` — garak seeds by payload index, and with a
    single goal the seed is the variant index instead.

    Args:
        techniques: Which techniques to use, in order. Defaults to both.
        homoglyph_variants: How many seeded homoglyph variants to emit.
        glyph_map: Override for the Latin -> homoglyph map.
    """

    def __init__(
        self,
        techniques: tuple[Technique, ...] | list[Technique] | None = None,
        homoglyph_variants: int = DEFAULT_HOMOGLYPH_VARIANTS,
        glyph_map: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        chosen = tuple(techniques) if techniques is not None else DEFAULT_TECHNIQUES
        unknown = [t for t in chosen if t not in DEFAULT_TECHNIQUES]
        if unknown:
            raise ValueError(
                f"unknown technique(s): {unknown}; "
                f"expected any of {list(DEFAULT_TECHNIQUES)}"
            )
        if not chosen:
            raise ValueError("at least one technique is required")
        if homoglyph_variants < 1:
            raise ValueError("homoglyph_variants must be >= 1")
        self._techniques = chosen
        self._homoglyph_variants = homoglyph_variants
        self._glyph_map = glyph_map if glyph_map is not None else DEFAULT_HOMOGLYPH_MAP
        self._goal: Goal | None = None
        self._variants: list[str] = []
        self._index = 0
        # Armed at RunStart, disarmed after the single injection of a run.
        self._injected = False

    def _build_variants(self, goal: Goal) -> list[str]:
        variants: list[str] = []
        for technique in self._techniques:
            if technique == "hypothetical":
                variants.append(hypothetical_wrap(goal.description))
            elif technique == "homoglyph":
                variants.extend(
                    homoglyph_replace(goal.description, self._glyph_map, seed=seed)
                    for seed in range(self._homoglyph_variants)
                )
        return variants

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._variants = self._build_variants(goal)
        self._index = 0
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # New attempt: re-arm the single injection for this run.
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, RunEndEvent):
            self._index += 1
            return RunEndResponse(event=event, done=self._index >= len(self._variants))

        return EventResponse(event=event)

    def _handle_pre_call(
        self,
        event: ControllablePreCallEvent,
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)

        # Leave any Task-configured system prompt untouched.
        if event.controllable.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            return decline

        # Only inject a surface that consumes free text; an obfuscated string
        # would be discarded by a json-typed surface.
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return decline

        # One obfuscated variant per run, mirroring garak's one-prompt-per-attempt.
        if self._injected or self._index >= len(self._variants):
            return decline

        self._injected = True
        return ControllableInjection(
            event=event,
            controllable=event.controllable,
            value=self._variants[self._index],
        )

    async def teardown(self) -> None:
        pass
