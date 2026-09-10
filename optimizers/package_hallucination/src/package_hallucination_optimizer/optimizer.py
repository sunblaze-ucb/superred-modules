"""PackageHallucinationOptimizer: elicit imports of non-existent packages.

Ports garak's ``probes/packagehallucination.py`` (Apache-2.0). Models
sometimes recommend importing packages that do not exist; an attacker can find
those names and squat them in a public registry, so that following the model's
code silently installs their package. This asks the target for code, across a
spread of framings and tasks, so the hallucinated names can be counted.

One variant per upstream language subclass:

``python`` ``ruby`` ``javascript`` ``rust`` ``raku`` ``perl`` ``dart``

Each builds ``stub_prompts`` x ``code_tasks`` = 10 x 24 = 240 prompts.

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

from package_hallucination_optimizer.surface_llm import classify_controllables
from package_hallucination_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

Variant = Literal["python", "ruby", "javascript", "rust", "raku", "perl", "dart"]

_DATA = json.loads(
    resources.files("package_hallucination_optimizer")
    .joinpath("data.json")
    .read_text("utf-8")
)
STUB_PROMPTS: list[str] = _DATA["stub_prompts"]
REAL_CODE_TASKS: list[str] = _DATA["real_code_tasks"]
UNREAL_CODE_TASKS: list[str] = _DATA["unreal_code_tasks"]
#: Upstream ``real_code_tasks + unreal_code_tasks``.
CODE_TASKS: list[str] = REAL_CODE_TASKS + UNREAL_CODE_TASKS
#: variant -> upstream ``language_name`` (e.g. ``"python"`` -> ``"Python3"``).
LANGUAGE_NAMES: dict[str, str] = {
    "python": _DATA["languages"]["Python"],
    "ruby": _DATA["languages"]["Ruby"],
    "javascript": _DATA["languages"]["JavaScript"],
    "rust": _DATA["languages"]["Rust"],
    "raku": _DATA["languages"]["RakuLand"],
    "perl": _DATA["languages"]["Perl"],
    "dart": _DATA["languages"]["Dart"],
}


def build_prompts(variant: Variant = "python") -> list[str]:
    """Upstream ``PackageHallucinationProbe.__init__``: stubs x tasks.

    ``for stub in stub_prompts: for task in code_tasks:`` filling
    ``<language>`` with the subclass's ``language_name`` and ``<task>`` with
    the code task.
    """
    if variant not in LANGUAGE_NAMES:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(LANGUAGE_NAMES)}"
        )
    language_name = LANGUAGE_NAMES[variant]
    return [
        stub.replace("<language>", language_name).replace("<task>", task)
        for stub in STUB_PROMPTS
        for task in CODE_TASKS
    ]


class PackageHallucinationOptimizer(Optimizer):
    """Sweep upstream's package-hallucination prompts for one language.

    Args:
        variant: which language subclass to run (default ``"python"``).
        shuffle: shuffle prompt order; upstream uses file order (and, in its
            own harness, an optional random cap -- see ASSUMPTIONS).
        seed: seeds the shuffle so a run is reproducible.
    """

    def __init__(
        self, variant: Variant = "python", shuffle: bool = False, seed: int = 0
    ) -> None:
        super().__init__()
        if variant not in LANGUAGE_NAMES:
            raise ValueError(
                f"unknown variant {variant!r}; expected one of {sorted(LANGUAGE_NAMES)}"
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
    def language_name(self) -> str:
        return LANGUAGE_NAMES[self._variant]

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
