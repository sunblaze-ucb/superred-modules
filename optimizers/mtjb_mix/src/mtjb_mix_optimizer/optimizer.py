"""MtjbMixOptimizer: MT-JailBench's Mix composite attack.

Mix picks three roles independently -- a *generator*, an *updater*, and a
*judge/flow* controller -- each one of {Crescendo, ActorBreaker, ChainOfAttack,
FootInTheDoor, XTeaming}, and drives them through the shared engine. Unlike the
CoA and X-Teaming ports, Mix reuses upstream's own five-family
generator/updater/judge/flow **verbatim** (the vendored engine), because
re-implementing all five families natively would duplicate the whole ecosystem.

The vendored generator/updater modules import ``textgrad`` at load, so the real
driving path requires the ``mix`` extra (``textgrad`` + ``tiktoken``) and is
**env-gated**. This module itself imports with no heavy dependency: the vendored
engine is imported lazily inside the bridge (``_engine_bridge``), which runs the
synchronous engine in a worker thread and routes every model call to ``self.llm``
via a stdlib client shim. Without ``textgrad`` the optimizer degrades cleanly
(it ends without sending). See ``ASSUMPTIONS.md``.
"""

from __future__ import annotations

import logging

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

from mtjb_mix_optimizer.surfaces import preferred_surface_names, should_inject

logger = logging.getLogger(__name__)

#: Valid role values (mirror engine/attack_type.py AttackType values; INTERACTIVE
#: and MIX are not selectable as a Mix role).
VALID_ROLES: frozenset[str] = frozenset(
    {"ActorBreaker", "ChainOfAttack", "Crescendo", "FootInTheDoor", "XTeaming"}
)

DEFAULT_GENERATOR = "ChainOfAttack"
DEFAULT_UPDATER = "XTeaming"
DEFAULT_JUDGE_AND_FLOW = "XTeaming"
DEFAULT_MAX_TOTAL_TURNS = 10
DEFAULT_MAX_TURNS = 5
DEFAULT_MAX_REFINES_PER_TURN = 1
DEFAULT_MAX_RESTARTS = 0


class MtjbMixOptimizer(Optimizer):
    """Mix composite optimizer: one conversation, one turn per superred run.

    Args:
        generator: Family that produces each turn's prompt.
        updater: Family that refines a prompt when the flow retries.
        judge_and_flow: Family whose judge scores responses and whose flow
            controller decides continue/retry/stop.
        max_total_turns: Hard cap on conversation turns (runs) per task.
        max_turns: Planned turn count handed to the generator (plan length).
        max_refines_per_turn: Refines allowed per plan step before advancing.
        max_restarts: Strategy restarts allowed (``JUMP_TO 1``); not rewound
            here, so a restart ends the attack (see ASSUMPTIONS.md).
        use_llm_for_similarity: CoA similarity via ``self.llm`` (no endpoint).

    Raises:
        ValueError: If any role is not one of :data:`VALID_ROLES`.
    """

    def __init__(
        self,
        *,
        generator: str = DEFAULT_GENERATOR,
        updater: str = DEFAULT_UPDATER,
        judge_and_flow: str = DEFAULT_JUDGE_AND_FLOW,
        max_total_turns: int = DEFAULT_MAX_TOTAL_TURNS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_refines_per_turn: int = DEFAULT_MAX_REFINES_PER_TURN,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        use_llm_for_similarity: bool = True,
    ) -> None:
        super().__init__()
        for role_name, role in (
            ("generator", generator),
            ("updater", updater),
            ("judge_and_flow", judge_and_flow),
        ):
            if role not in VALID_ROLES:
                raise ValueError(
                    f"{role_name}={role!r} is not a valid Mix role; "
                    f"choose one of {sorted(VALID_ROLES)}"
                )
        self._generator = generator
        self._updater = updater
        self._judge_and_flow = judge_and_flow
        self._max_total_turns = max(max_total_turns, 1)
        self._max_turns = max(max_turns, 1)
        self._max_refines_per_turn = max(max_refines_per_turn, 0)
        self._max_restarts = max(max_restarts, 0)
        self._use_llm_for_similarity = use_llm_for_similarity

        self._engine: object | None = None
        self._goal: Goal | None = None
        self._pending: str | None = None
        self._done = False
        self._preferred: frozenset[str] = frozenset()
        self._injected = False
        self._saw_post_call = False
        self._scored_this_run = False
        self._channel: str | None = None

    # ------------------------------------------------------------------
    def _config(self) -> dict:
        """Build the vendored-engine config. Model ids are placeholders: the
        client shim ignores them and routes to ``self.llm``."""
        routed = {"model": "routed-via-self-llm", "provider": None, "base_url": None}
        return {
            "generator": self._generator,
            "updater": self._updater,
            "judge_and_flow": self._judge_and_flow,
            "attacker_model": routed,
            "judge_model": routed,
            "mirror_target_config": routed,
            "similarity_model": routed,
            "use_llm_for_similarity": self._use_llm_for_similarity,
            "max_refines_per_turn": self._max_refines_per_turn,
            "max_restarts": self._max_restarts,
            "max_update_retries": 5,
            "n_init_chains": 1,
            "semantic_update_slack": 0.1,
            "enable_attack_update": True,
            "num_sets": 1,
            "use_multiple_strategies": False,
        }

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._engine = None
        self._pending = None
        self._done = False
        self._injected = False
        self._saw_post_call = False
        self._scored_this_run = False
        self._channel = None
        self._preferred = preferred_surface_names(controllables)

        self._pending = await self._start_engine(goal.description)
        if self._pending is None:
            self._done = True

    async def _start_engine(self, behavior: str) -> str | None:
        """Build + start the vendored engine; return the first prompt or None.

        Returns None (and the optimizer ends) when the engine cannot run --
        most importantly when ``textgrad`` is not installed.
        """
        try:
            from mtjb_mix_optimizer._engine_bridge import MixEngineBridge

            self._engine = MixEngineBridge(
                config=self._config(),
                behavior=behavior,
                max_total_turns=self._max_total_turns,
                max_turns=self._max_turns,
            )
            return await self._engine.start(self.llm)
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - textgrad/deps missing or setup failed
            logger.warning("Mix engine could not start (needs the 'refine' extra?); ending")
            return None

    async def _observe(self, response: str) -> str | None:
        """Feed the response to the engine; return the next prompt or None."""
        if self._engine is None:
            return None
        try:
            next_prompt, done, _success = await self._engine.observe(self.llm, response)  # type: ignore[attr-defined]
            if done:
                self._done = True
            return next_prompt
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            self._done = True
            return None
        except Exception:  # noqa: BLE001 - engine step failed; end the attack
            logger.warning("Mix engine step failed; ending")
            self._done = True
            return None

    # ------------------------------------------------------------------
    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            self._saw_post_call = False
            self._scored_this_run = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            self._saw_post_call = await self._handle_post_call(event) or self._saw_post_call
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            if self._injected and not self._saw_post_call:
                self._done = True
            elif self._channel is not None and not self._injected and self._pending is not None:
                self._done = True
            return RunEndResponse(event=event, done=self._done)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred):
            return decline
        if self._injected or self._done or self._pending is None:
            return decline
        if self._channel is not None and event.controllable.name != self._channel:
            return decline
        value = self._pending
        self._injected = True
        self._pending = None
        if self._channel is None:
            self._channel = event.controllable.name
        return ControllableInjection(event=event, controllable=event.controllable, value=value)

    async def _handle_post_call(self, event: ControllablePostCallEvent) -> bool:
        if self._done or self._goal is None:
            return False
        if self._scored_this_run:
            return False
        if self._channel is None or event.controllable.name != self._channel:
            return False
        self._scored_this_run = True
        self._pending = await self._observe(event.answer)
        if self._pending is None:
            self._done = True
        return True

    async def teardown(self) -> None:
        pass
