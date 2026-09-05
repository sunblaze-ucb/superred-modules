"""PolicyPuppetryOptimizer: smuggle the goal into a fake TV-script config.

Ports garak's Policy Puppetry bypass (``garak/probes/doctor.py``,
Apache-2.0): an XML-ish "interaction-config" block that redefines the model's
allowed modes and blocked strings, wrapping the request as a line in a Dr.
House scene. Upstream fills the template's ``{{HARMFUL_BEHAVIOR}}`` slot from
a fixed list; here it takes the task's goal.
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

from policy_puppetry_optimizer.templates import leetspeak, render

Technique = Literal["plain", "leet"]
DEFAULT_TECHNIQUES: tuple[Technique, ...] = ("plain", "leet")
DEFAULT_LEET_VARIANTS = 3

# Reserved controllable name never injected into, so a Task-configured system
# prompt survives unchanged (matches the goal_passthrough baseline).
_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})


class PolicyPuppetryOptimizer(Optimizer):
    """Send the goal wrapped in the Policy Puppetry scene, one variant per run.

    ``plain`` renders the template with the goal in the slot, as upstream
    ``Bypass`` does. ``leet`` adds upstream ``BypassLeet``'s aside and
    leetspeaks the whole prompt, yielding ``leet_variants`` variants seeded
    ``0..n-1`` — upstream leetspeaks from an unseeded RNG, so seeding is what
    makes the variants reproducible.

    Args:
        techniques: Which techniques to use, in order. Defaults to both.
        leet_variants: How many seeded leetspeak variants to emit.
    """

    def __init__(
        self,
        techniques: tuple[Technique, ...] | list[Technique] | None = None,
        leet_variants: int = DEFAULT_LEET_VARIANTS,
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
        if leet_variants < 1:
            raise ValueError("leet_variants must be >= 1")
        self._techniques = chosen
        self._leet_variants = leet_variants
        self._goal: Goal | None = None
        self._variants: list[str] = []
        self._index = 0
        # Armed at RunStart, disarmed after the single injection of a run.
        self._injected = False

    def _build_variants(self, goal: Goal) -> list[str]:
        variants: list[str] = []
        for technique in self._techniques:
            if technique == "plain":
                variants.append(render(goal.description))
            elif technique == "leet":
                rendered = render(goal.description, leet_suffix=True)
                variants.extend(
                    leetspeak(rendered, seed=seed)
                    for seed in range(self._leet_variants)
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

        # Leave any Task-configured system prompt untouched: the scene config
        # is the attack, and it is delivered as a user message upstream.
        if event.controllable.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            return decline

        # The prompt is markup-flavoured prose; a json-typed surface would
        # discard it.
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return decline

        # One scene per run, mirroring garak's one-prompt-per-attempt.
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
