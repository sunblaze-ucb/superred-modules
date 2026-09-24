"""ContextComplianceOptimizer: the Context Compliance Attack (CCA) for superred.

Ports the ``context_compliance`` technique from microsoft/PyRIT (MIT), pinned at
commit ``2016c4a``. CCA (Russinovich & Salem, "Jailbreaking is (Mostly) Simpler
Than You Think", arXiv:2503.05264) fabricates a benign prior exchange -- a
neutral user question about the objective's topic and an affirmative assistant
answer that offers to carry out the objective -- then sends the real request, so
the target "complies with its own prior context".

At ``initialize`` the fabricated exchange is built once, from the two vendored
PyRIT templates, with both adversarial generations routed through the
constrained ``self.llm``:

* the adversarial persona (``context_compliance.yaml``) turns the objective into
  the benign opening user question ``Q``;
* the simulated target (``context_compliance_target.yaml``) answers ``Q`` with
  the fabricated affirmative assistant turn ``A``.

The framework exposes no primitive for injecting prior conversation turns, so
the fabricated ``[user Q][assistant A][user real-request]`` history is encoded
into a single free-text user-message injection (a labelled transcript) on a
conversation surface -- never the system prompt. See ``ASSUMPTIONS.md`` for this
and every other deliberate deviation, including how the final real request is
chosen.

The prompt bodies are loaded from the byte-identical vendored YAML; none are
reproduced here.
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
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from context_compliance_optimizer import conversation
from context_compliance_optimizer.surfaces import preferred_surface_names, should_inject

#: Upstream (PyRIT ``context_compliance`` technique) simulated-conversation length.
DEFAULT_NUM_TURNS = conversation.DEFAULT_NUM_TURNS

#: The fixed affirmation PyRIT's technique wiring sends as the final user turn.
#: Pass ``final_user_message=UPSTREAM_FINAL_USER_MESSAGE`` to reproduce it.
UPSTREAM_FINAL_USER_MESSAGE = conversation.UPSTREAM_FINAL_USER_MESSAGE


class ContextComplianceOptimizer(Optimizer):
    """Inject a fabricated prior exchange, then the real request (CCA).

    The fabricated exchange is generated once at ``initialize`` and injected
    once per run into a single free-text conversation surface. CCA is a single
    deterministic attempt per task.

    Args:
        num_turns: Number of simulated turns used to render the templates.
            Upstream uses ``1``. Clamped to ``>= 1``.
        final_user_message: The real request that follows the fabricated
            exchange. ``None`` (default) uses the objective itself as the real
            request -- the fabricated assistant turn is followed by the goal.
            Set to a fixed string (e.g. :data:`UPSTREAM_FINAL_USER_MESSAGE`,
            ``"yes."``) to reproduce PyRIT's wiring, where the objective is
            carried by the fabricated assistant turn's offer instead. See
            ASSUMPTIONS.md.
    """

    def __init__(
        self,
        *,
        num_turns: int = DEFAULT_NUM_TURNS,
        final_user_message: str | None = None,
    ) -> None:
        super().__init__()
        if final_user_message is not None and not final_user_message.strip():
            raise ValueError("final_user_message must be non-empty when provided")
        self._num_turns = max(int(num_turns), 1)
        self._final_user_message = final_user_message

        # Per-task state (rebuilt in initialize()).
        self._goal: Goal | None = None
        self._payload: str | None = None
        self._preferred: frozenset[str] = frozenset()
        # Per-run state (re-armed on RunStart).
        self._injected = False

    @property
    def payload(self) -> str | None:
        """The fabricated-conversation transcript, once built (else ``None``)."""
        return self._payload

    @property
    def num_turns(self) -> int:
        return self._num_turns

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    async def _ask(self, system: str, user: str) -> str | None:
        """One adversarial-generation LLM call. ``None`` on failure.

        A genuine budget exhaustion (cost already spent) is re-raised so a spent
        run is never reported as a quietly finished attack; the zero-cost noop
        client the controller hands non-LLM optimizers degrades quietly instead.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            result = await self.llm.complete(messages)
            return result.choices[0].message.content or ""
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - degrade like the sibling single-turn ports
            return None

    async def _build_payload(self, objective: str) -> str | None:
        """Build the fabricated ``[user Q][assistant A][user real-request]`` history.

        Returns the encoded transcript, or ``None`` if either adversarial
        generation is unavailable/empty (so the optimizer declines cleanly
        rather than injecting a malformed context).
        """
        adversarial_system = conversation.adversarial_system_prompt(
            objective, max_turns=self._num_turns
        )
        question = conversation.clean_generation(await self._ask(adversarial_system, objective))
        if not question:
            return None

        target_system = conversation.simulated_target_system_prompt(
            objective, num_turns=self._num_turns
        )
        answer = conversation.clean_generation(await self._ask(target_system, question))
        if not answer:
            return None

        final_request = (
            self._final_user_message if self._final_user_message is not None else objective
        )
        return conversation.assemble_transcript(question, answer, final_request)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
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
        self._preferred = preferred_surface_names(controllables)
        self._payload = await self._build_payload(goal.description)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            # PostCall requires an injection decision: the channel type-checks
            # the response, so a bare EventResponse would abort the task on any
            # target that emits one. CCA never rewrites the target's answer.
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if isinstance(event, RunEndEvent):
            # CCA is a single deterministic attempt: one fabricated context,
            # one injection.
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if self._payload is None or self._injected:
            return decline
        if not should_inject(event.controllable, self._preferred):
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._payload
        )

    async def teardown(self) -> None:
        pass
