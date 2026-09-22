"""SkeletonKeyOptimizer: Microsoft's Skeleton Key jailbreak, one shot per run.

Ports ``SkeletonKeyAttack`` from Microsoft PyRIT
(``pyrit/executor/attack/single_turn/skeleton_key.py``). Skeleton Key primes
the target with a fabricated exchange -- a "skeleton key" preamble (user) that
instructs the model to update its behaviour and comply with any request, plus a
simulated assistant acceptance -- and then sends the real objective. Learn more
from Microsoft's disclosure "Mitigating Skeleton Key, a new type of generative
AI jailbreak technique" (Mark Russinovich, Microsoft, 2024).

The preamble and acceptance bodies are loaded from the byte-identical vendored
``skeleton_key.prompt`` / ``skeleton_key_acceptance.prompt`` seed files; none
are reproduced here. Skeleton Key is prompt-only: it makes no model calls of its
own (``self.llm`` is available but unused), pulls in no heavy dependency, and is
fully deterministic.

superred's controllable model injects a single string per turn and exposes no
editable conversation history, so the fabricated preamble/acceptance exchange
is rendered ahead of the objective exactly as PyRIT itself renders a prepended
conversation for a target *without* editable history -- the turn-based
``ConversationContextNormalizer`` transcript. See ``ASSUMPTIONS.md``.
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

from skeleton_key_optimizer.surfaces import preferred_surface_names, should_inject
from skeleton_key_optimizer.vendored import (
    load_skeleton_key_acceptance,
    load_skeleton_key_prompt,
)

#: Upstream default: retries only on failure, and 0 of them (one send). Total
#: sends across runs = ``1 + max_attempts_on_failure``.
DEFAULT_MAX_ATTEMPTS_ON_FAILURE = 0

# --- Prepended-conversation transcript tokens ------------------------------
# The fabricated skeleton-key exchange is flattened ahead of the objective in
# PyRIT's own turn-based format for a target without editable history
# (``pyrit.message_normalizer.ConversationContextNormalizer``:
# "Turn N:\n<role>: <content>"). These are structural tokens, not prompt text.
_TURN_PREFIX = "Turn"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"


def render(objective: str, skeleton_key_prompt: str, skeleton_key_acceptance: str) -> str:
    """Flatten the primed skeleton-key exchange and the objective into one turn.

    Mirrors PyRIT's ``SkeletonKeyAttack`` -- which prepends
    ``[user: skeleton_key_prompt, assistant: skeleton_key_acceptance]`` and then
    sends ``objective`` -- rendered as PyRIT's ``ConversationContextNormalizer``
    transcript (the format used when the target has no editable history):

        Turn 1:
        user: <skeleton key preamble>
        assistant: <simulated acceptance>
        Turn 2:
        user: <objective>

    The skeleton-key preamble therefore always precedes the objective.
    """
    return (
        f"{_TURN_PREFIX} 1:\n"
        f"{_USER_ROLE}: {skeleton_key_prompt}\n"
        f"{_ASSISTANT_ROLE}: {skeleton_key_acceptance}\n"
        f"{_TURN_PREFIX} 2:\n"
        f"{_USER_ROLE}: {objective}"
    )


class SkeletonKeyOptimizer(Optimizer):
    """Send the objective primed by a fabricated skeleton-key exchange.

    One deterministic prompt per run, injected into the first eligible free-text
    controllable. No model calls (Skeleton Key is prompt-only).

    Args:
        skeleton_key_prompt: Override the skeleton-key preamble (the user turn
            that instructs the model to update its behaviour). ``None`` (default)
            loads the vendored ``skeleton_key.prompt`` body. Mirrors upstream.
        skeleton_key_acceptance: Override the simulated assistant acceptance.
            ``None`` (default) loads the vendored ``skeleton_key_acceptance.prompt``
            body. Mirrors upstream.
        max_attempts_on_failure: Retry budget mirroring upstream's constructor.
            The deterministic payload is (re)sent across up to
            ``1 + max_attempts_on_failure`` runs; the controller/claim stops the
            task early on success (this optimizer runs no self-judge). Clamped
            to >= 0.
    """

    def __init__(
        self,
        *,
        skeleton_key_prompt: str | None = None,
        skeleton_key_acceptance: str | None = None,
        max_attempts_on_failure: int = DEFAULT_MAX_ATTEMPTS_ON_FAILURE,
    ) -> None:
        super().__init__()
        # Load vendored bodies unless the caller overrides (upstream semantics).
        self._skeleton_key_prompt = (
            skeleton_key_prompt if skeleton_key_prompt is not None else load_skeleton_key_prompt()
        )
        self._skeleton_key_acceptance = (
            skeleton_key_acceptance
            if skeleton_key_acceptance is not None
            else load_skeleton_key_acceptance()
        )
        if not self._skeleton_key_prompt.strip():
            raise ValueError("skeleton_key_prompt must not be empty")
        if not self._skeleton_key_acceptance.strip():
            raise ValueError("skeleton_key_acceptance must not be empty")
        self._max_attempts = 1 + max(max_attempts_on_failure, 0)

        # Per-task state (reset in initialize()).
        self._goal: Goal | None = None
        self._payload = ""
        self._preferred: frozenset[str] = frozenset()
        self._attempts_made = 0
        # Per-run state (re-armed on RunStart).
        self._injected = False

    @property
    def skeleton_key_prompt(self) -> str:
        """The skeleton-key preamble body in use."""
        return self._skeleton_key_prompt

    @property
    def skeleton_key_acceptance(self) -> str:
        """The simulated acceptance body in use."""
        return self._skeleton_key_acceptance

    @property
    def payload(self) -> str:
        """The rendered injection for the current goal (empty until initialize)."""
        return self._payload

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._payload = render(
            goal.description, self._skeleton_key_prompt, self._skeleton_key_acceptance
        )
        self._preferred = preferred_surface_names(controllables)
        self._attempts_made = 0
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # Re-arm ALL per-run state so a fresh run injects again.
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            # PostCall requires an injection decision: the channel type-checks
            # the response, so a bare EventResponse raises and aborts the task
            # on any target that emits PostCall. Skeleton Key never rewrites the
            # target's answer, so it declines.
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            self._attempts_made += 1
            done = self._attempts_made >= self._max_attempts
            return RunEndResponse(event=event, done=done)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred):
            return decline
        if self._injected:
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._payload
        )

    async def teardown(self) -> None:
        pass
