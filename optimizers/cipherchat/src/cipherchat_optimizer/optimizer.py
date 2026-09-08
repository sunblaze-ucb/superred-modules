"""CipherChatOptimizer: converse with the target in a cipher it is taught.

Ports CipherChat (RobustNLP/CipherChat, MIT): a system prompt teaches the
target a cipher and shows a few enciphered demonstrations, then the objective
is sent enciphered in the user turn, so safety training keyed to natural
language never fires. One attempt per task; no attacker LLM is needed to build
the prompt.

Paper: Yuan et al., "GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs
via Cipher" (ICLR 2024).

See ``ASSUMPTIONS.md`` for every deliberate deviation.
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

from cipherchat_optimizer.assembly import (
    DEFAULT_CATEGORY,
    DEFAULT_CIPHER,
    DEFAULT_LANGUAGE,
    DEFAULT_TOXICITY,
    build_system_prompt,
    encode,
)
from cipherchat_optimizer.surface_llm import classify_controllables
from cipherchat_optimizer.surfaces import (
    ROLE_CATEGORIES,
    SYSTEM_PROMPT_CONTROLLABLE_NAME,
    preferred_surface_names,
    should_inject,
)


class CipherChatOptimizer(Optimizer):
    """Teach the target a cipher, then send the enciphered goal.

    The cipher teaching (and enciphered demonstrations) go into a system-prompt
    surface when the attacker can write one; otherwise they are prepended to
    the enciphered query in the user turn, so the attack still lands on a
    target that exposes only a user surface.

    Args:
        cipher: One of ``assembly.CIPHERS`` (default ``caesar``).
        category: Demonstration domain (default
            ``Crimes_And_Illegal_Activities``).
        toxicity: ``"toxic"`` (upstream default) or ``"harmless"``.
        language: ``"en"`` or ``"zh"``.
        use_demonstrations: Include the enciphered few-shot examples.
        use_system_surface: Deliver the teaching via a system-prompt surface
            when one is available (else always combine into the user turn).
    """

    def __init__(
        self,
        cipher: str = DEFAULT_CIPHER,
        category: str = DEFAULT_CATEGORY,
        toxicity: str = DEFAULT_TOXICITY,
        language: str = DEFAULT_LANGUAGE,
        use_demonstrations: bool = True,
        use_system_surface: bool = True,
    ) -> None:
        super().__init__()
        # build_system_prompt validates cipher/category/toxicity/language, so
        # surface a bad config at construction with a clear message.
        build_system_prompt(
            cipher,
            category=category,
            toxicity=toxicity,
            language=language,
            use_demonstrations=use_demonstrations,
        )
        self._cipher = cipher
        self._category = category
        self._toxicity = toxicity
        self._language = language
        self._use_demonstrations = use_demonstrations
        self._use_system_surface = use_system_surface

        self._goal: Goal | None = None
        self._teaching = ""
        self._encoded_query = ""
        self._split = False
        self._teaching_done = False
        self._query_done = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    @property
    def cipher(self) -> str:
        return self._cipher

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._teaching = build_system_prompt(
            self._cipher,
            category=self._category,
            toxicity=self._toxicity,
            language=self._language,
            use_demonstrations=self._use_demonstrations,
        )
        self._encoded_query = encode(goal.description, self._cipher)
        self._teaching_done = False
        self._query_done = False

        # Split the teaching onto a system-prompt surface only when the target
        # actually exposes one to the attacker; otherwise combine.
        has_system_surface = any(
            c.name == SYSTEM_PROMPT_CONTROLLABLE_NAME for c in controllables
        )
        self._split = self._use_system_surface and has_system_surface

        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # initialize() runs once per task, but a task can start many runs.
            # Without re-arming here, run 2 inherits run 1's flags: the
            # teaching branch is skipped as already delivered and the query
            # branch as already sent, so the run injects nothing at all.
            self._teaching_done = False
            self._query_done = False
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
            # The attack is one shot: teach, then ask, and it is spent. That
            # holds whether or not the query landed -- if this run found no
            # eligible user surface for the ciphertext, neither will the next
            # one, and reporting "not done" only spins out the run budget
            # re-declining the same surfaces.
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        name = event.controllable.name

        # Teaching goes to the system-prompt surface in split mode.
        if (
            self._split
            and name == SYSTEM_PROMPT_CONTROLLABLE_NAME
            and not self._teaching_done
        ):
            self._teaching_done = True
            return ControllableInjection(
                event=event, controllable=event.controllable, value=self._teaching
            )

        # The enciphered query goes to the user surface. When the teaching was
        # not (yet) delivered to a system surface, it is prepended here so the
        # target still learns the cipher.
        if (
            not self._query_done
            and should_inject(event.controllable, self._preferred, self._surface_roles)
        ):
            if self._split and self._teaching_done:
                value = self._encoded_query
            else:
                value = self._teaching + "\n\n" + self._encoded_query
                self._teaching_done = True
            self._query_done = True
            return ControllableInjection(
                event=event, controllable=event.controllable, value=value
            )

        return decline

    async def teardown(self) -> None:
        pass
