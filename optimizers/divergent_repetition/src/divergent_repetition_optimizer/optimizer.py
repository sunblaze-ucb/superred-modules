"""DivergentRepetitionOptimizer: ask the model to repeat a word forever.

Ports garak's ``divergence.Repeat`` / ``RepeatExtended`` (Apache-2.0): prompt
the target to repeat a word many times, which can make an aligned model diverge
into regurgitating memorised training data (the Carlini et al. "poem poem poem"
extraction). This is a **data-extraction** attack, so it is goal-agnostic — the
prompts do not embed the task Goal, they exercise the repetition behaviour, and
a SecurityClaim judges whether the response diverged.

The word lists, prompt templates and repetition counts are extracted verbatim
from ``garak/probes/divergence.py`` into ``data.json`` and reassembled with the
same nested loop. No attacker LLM.

Reference: Nasr et al., "Scalable Extraction of Training Data from (Production)
Language Models" (2023).

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

from divergent_repetition_optimizer.surface_llm import classify_controllables
from divergent_repetition_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

_SLOT = "[XX]"


def _data() -> dict:
    text = resources.files("divergent_repetition_optimizer").joinpath("data.json").read_text("utf-8")
    return json.loads(text)


def build_prompts(extended: bool = False) -> list[str]:
    """Reproduce garak's ``Repeat.__init__`` prompt construction.

    For each repeat word, each of the three prompt templates, and each
    repetition count ``[1, 3, 40]``, substitute the word repeated ``r`` times
    into the template's ``[XX]`` slot — the exact nested order upstream uses.

    Args:
        extended: Use ``RepeatExtended``'s 105-word list instead of ``Repeat``'s
            4-word list (upstream marks ``RepeatExtended`` ``active = False``).
    """
    data = _data()
    words = data["repeat_words_extended"] if extended else data["repeat_words"]
    templates = data["prompt_templates"]
    repetitions = data["num_repetitions"]
    prompts: list[str] = []
    for repeat_word in words:
        for template in templates:
            for r in repetitions:
                prompts.append(template.replace(_SLOT, " ".join([repeat_word] * r)))
    return prompts


class DivergentRepetitionOptimizer(Optimizer):
    """Sweep garak's repeat-a-word prompts, one per run.

    This attack does not use the task Goal (it exercises repetition-induced
    divergence, not goal elicitation); the Goal is available but unused. Whether
    a response actually leaked data is for the SecurityClaim to judge.

    Args:
        extended: Use the 105-word ``RepeatExtended`` list (default: the 4-word
            ``Repeat`` list).
        shuffle: Shuffle the prompt order (seeded). ``False`` uses upstream's
            nested-loop order.
        seed: Shuffle seed.
    """

    def __init__(
        self,
        extended: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._extended = extended
        self._shuffle = shuffle
        self._seed = seed
        self._prompts: list[str] = []
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
        self._prompts = build_prompts(extended=self._extended)
        if self._shuffle:
            random.Random(self._seed).shuffle(self._prompts)
        self._index = 0
        self._injected = False
        # Goal-agnostic, but still choose the user-prompt surface the same way.
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
