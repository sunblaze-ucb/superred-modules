"""CodeAttackOptimizer: wrap the goal as a code-completion task.

Ports CodeAttack (renqibing/CodeAttack, MIT): the goal is tokenised into a data
structure inside a code snippet (python list / stack / string) and the model is
asked to "complete the code", which slips the request past refusal. One prompt
per run; no LLM calls.
"""

from __future__ import annotations

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

from codeattack_optimizer.surface_llm import classify_controllables
from codeattack_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)
from codeattack_optimizer.codeattack import VARIANTS, render



class CodeAttackOptimizer(Optimizer):
    """Send the goal wrapped in a code-completion prompt, once per run.

    Args:
        variant: Code structure to encode the goal in — ``"python_stack"``
            (upstream's headline), ``"python_list"`` or ``"python_string"``.
    """

    def __init__(self, variant: str = "python_stack") -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of {list(VARIANTS)}"
            )
        self._variant = variant
        self._goal: Goal | None = None
        self._payload = ""
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
        # One LLM pass reads each surface's own description and names the
        # user's prompt -- the channel this attack has to arrive on. Returns {}
        # on any failure, so the name/value-type backstop decides instead.
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)
        self._goal = goal
        self._payload = render(goal.description, self._variant)
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            # Post-call events also require an injection decision: the channel
            # type-checks the response, so a bare ``EventResponse`` raises and
            # aborts the task on any target that emits them. This optimizer
            # never rewrites the target's answer, so it declines.
            return ControllableNoInjection(
                event=event, controllable=event.controllable
            )

        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected:
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._payload
        )

    async def teardown(self) -> None:
        pass
