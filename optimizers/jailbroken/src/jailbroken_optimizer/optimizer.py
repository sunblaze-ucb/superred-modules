"""JailbrokenOptimizer: wrap the goal in a prefix-injection jailbreak template.

Ports the Jailbroken attack as implemented in Tencent AI-Infra-Guard
(``AIG-PromptSecurity/deepteam/attacks/single_turn/jailbroken``, Apache-2.0):
a collection of prefix-injection templates that instruct the target to begin
its reply with an affirmative phrase and then answer the request. Deterministic
— no attacker LLM.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).

Technique: Wei et al., "Jailbroken: How Does LLM Safety Training Fail?" (2023).

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import random

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

from jailbroken_optimizer._vendor.aig_jailbroken.template import JAILBROKEN_TEMPLATES
from jailbroken_optimizer.surface_llm import classify_controllables
from jailbroken_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

#: Number of templates upstream ships.
TEMPLATE_COUNT = len(JAILBROKEN_TEMPLATES)


def render(goal: str, index: int) -> str:
    """Fill template *index* (0-based) with *goal*.

    Byte-faithful to upstream ``Jailbroken.enhance``: take the single value of
    the chosen template dict and ``.format(query=...)`` the goal into it.
    """
    if not 0 <= index < TEMPLATE_COUNT:
        raise ValueError(f"index must be 0..{TEMPLATE_COUNT - 1}, got {index!r}")
    template = JAILBROKEN_TEMPLATES[index]
    prompt = next(iter(template.values()))
    return prompt.format(query=goal)


class JailbrokenOptimizer(Optimizer):
    """Send the goal wrapped in a Jailbroken prefix-injection template.

    Upstream picks one template at random per call. To exercise the whole set,
    this optimizer sends one template per run and ends after all of them; the
    order is a seeded shuffle so runs are reproducible.

    Args:
        seed: Shuffle seed for the template order (upstream is unseeded random;
            seeding makes a run reproducible).
        shuffle: Shuffle the template order (default). ``False`` uses upstream
            file order.
    """

    def __init__(self, seed: int = 0, shuffle: bool = True) -> None:
        super().__init__()
        self._seed = seed
        self._shuffle = shuffle
        self._goal: Goal | None = None
        self._order: list[int] = []
        self._index = 0
        self._injected = False
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
        self._order = list(range(TEMPLATE_COUNT))
        if self._shuffle:
            random.Random(self._seed).shuffle(self._order)
        self._index = 0
        self._injected = False
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)

    def _current_payload(self) -> str:
        assert self._goal is not None
        return render(self._goal.description, self._order[self._index])

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
            return RunEndResponse(event=event, done=self._index >= len(self._order))
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected or self._index >= len(self._order):
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._current_payload()
        )

    async def teardown(self) -> None:
        pass
