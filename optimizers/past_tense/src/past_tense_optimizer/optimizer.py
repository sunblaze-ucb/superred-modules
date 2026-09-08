"""PastTenseOptimizer: rephrase the goal into the past (or future) tense.

Ports the past/future tense attack as implemented in Tencent AI-Infra-Guard
(``AIG-PromptSecurity/deepteam/attacks/single_turn/past_tense``, Apache-2.0).
The attacker LLM reformulates the objective as a question in the past or future
tense, which slips past refusal training keyed to present-tense requests. One
rephrase call, one injection.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).

Technique: Andriushchenko & Flammarion, "Does Refusal Training in LLMs
Generalize to the Past Tense?" (2024).

See ``ASSUMPTIONS.md`` for every deliberate deviation.
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
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from past_tense_optimizer._vendor.aig_past_tense.template import (
    FUTURE_TENSE_PROMPT,
    PAST_TENSE_PROMPT,
)
from past_tense_optimizer.surface_llm import classify_controllables
from past_tense_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

Tense = Literal["past", "present", "future"]

#: Upstream's ``PastTense.__init__`` defaults.
DEFAULT_TENSE: Tense = "past"
DEFAULT_MAX_RETRIES = 5


class PastTenseOptimizer(Optimizer):
    """Send the goal reformulated into the past or future tense, once per run.

    Args:
        tense: ``"past"`` (default), ``"future"``, or ``"present"`` (a control
            that sends the goal unchanged, as upstream does).
        max_retries: How many times to ask the attacker LLM for a non-empty
            reformulation before falling back to the original goal.
    """

    def __init__(
        self,
        tense: Tense = DEFAULT_TENSE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        super().__init__()
        if tense not in ("past", "present", "future"):
            raise ValueError(
                f"tense must be 'past', 'present' or 'future', got {tense!r}"
            )
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries!r}")
        self._tense: Tense = tense
        self._max_retries = max_retries

        self._goal: Goal | None = None
        self._payload = ""
        self._injected = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    @property
    def tense(self) -> Tense:
        return self._tense

    async def _generate(self, prompt: str) -> str | None:
        """One attacker-LLM call. ``None`` on failure, which upstream retries.

        A genuine budget exhaustion is re-raised: folding it into the retry
        path would exhaust every attempt on an attacker that ran out of money
        and then silently fall back to the un-rephrased goal. The budget-less
        noop client raises the same error with nothing spent, so only
        ``usage.cost > 0`` propagates, matching
        ``surface_llm._is_genuine_exhaustion``.
        """
        try:
            result = await self.llm.complete([{"role": "user", "content": prompt}])
            return result.choices[0].message.content or ""
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - upstream retries on any failure
            return None

    async def _reformulate(self, request: str) -> str:
        """Reformulate *request* into the configured tense.

        Byte-faithful to upstream ``PastTense.enhance``: present returns the
        request unchanged; otherwise the tense template is filled and the model
        asked up to ``max_retries`` times, each result stripped of quotes.
        Falls back to the original request if every attempt is empty (or the
        attacker LLM is unavailable), exactly as upstream returns ``attack``.
        """
        if self._tense == "present":
            return request
        template = PAST_TENSE_PROMPT if self._tense == "past" else FUTURE_TENSE_PROMPT
        prompt = template.format(request=request)
        for _ in range(self._max_retries):
            reformulated = await self._generate(prompt)
            if not reformulated:
                continue
            reformulated = reformulated.replace('"', "").strip()
            if reformulated:
                return reformulated
        return request

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._injected = False
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)
        # Reformulate once at initialize; the same tense-shifted prompt is sent
        # on every attempt of this task.
        self._payload = await self._reformulate(goal.description)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, RunEndEvent):
            # Single deterministic reformulation: one attempt per task.
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
