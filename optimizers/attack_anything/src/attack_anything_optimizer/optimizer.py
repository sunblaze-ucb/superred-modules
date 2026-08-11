"""Attack Anything (SEATS) optimizer for superred — the planner pump.

The upstream SEATS engine drives the victim synchronously; superred inverts
control, so the search lives in :mod:`attack_anything_optimizer.planner` as a
generator that yields one victim message (`Unit`) at a time. This module is the
**pump**: it drives that generator across superred events, mapping each `Unit`
onto a superred conversation.

Run mapping. A `Unit.fresh` message must start a NEW conversation, which is a
fresh superred run (the chatbot target resets between runs); a non-fresh message
continues the current run's conversation. So a fresh-per-subtask / turn-fresh /
recursive-leaf / goal-as-root protocol maps naturally onto runs, and the final
RDRT-lineage **assembly** unit is one more fresh run whose transcript the
SecurityClaim judges. The framework verdict on that assembly run (or on any run
the judge scores as goal-achieving) is the authoritative `TaskResult.success`;
the planner's own internal dual-judge only steers the search.

The planner is advanced on a worker thread (`asyncio.to_thread`) so its vendored,
synchronous LLM helper calls bridge back to the event loop through
:class:`VendorLLMBridge`; unit *execution* (injecting a message, reading the
reply) happens on the loop via events.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from attack_anything_optimizer._llm import VendorLLMBridge, _BudgetSignal, _NoLLMSignal
from attack_anything_optimizer.config import AttackAnythingConfig
from attack_anything_optimizer.planner import Planner, Unit

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_NAME = "system_prompt"
_RESPONSE_NAME = "response"
_RESPONSE_OBSERVABLE_NAMES = frozenset({"response", "model_response", "assistant_response"})
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})
_VENDOR_MODEL = "attacker"

# Sentinel returned when the planner generator is exhausted.
_DONE = object()


def _safe_send(gen: Any, value: Any) -> Any:
    """Advance a generator, returning ``_DONE`` on StopIteration.

    Wrapping keeps ``StopIteration`` out of the async/thread boundary (raising it
    across ``asyncio.to_thread`` would be mishandled).
    """
    try:
        return gen.send(value)
    except StopIteration:
        return _DONE


class AttackAnythingOptimizer(Optimizer):
    """Self-evolving attack-tree-search jailbreak optimizer (full v2 parity)."""

    def __init__(
        self,
        *,
        config: AttackAnythingConfig | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        base = config or AttackAnythingConfig()
        if overrides:
            from dataclasses import replace

            base = replace(base, **overrides)
        self._cfg = base

        self._goal: Goal | None = None
        self._planner: Planner | None = None
        self._gen: Any = None
        self._primed = False
        self._done = False
        self._framework_succeeded = False
        self._llm_available = False
        self._primary_pre_controllable: Controllable | None = None

        # Cross-run cursor.
        self._current_unit: Unit | None = None
        self._pending_fresh_unit: Unit | None = None

        # Per-run state.
        self._awaiting_reply = False
        self._pending_post_answer: str | None = None
        self._last_pre_request: str | None = None
        self._last_injected_value: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
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
        if self._cfg.judge_mode:
            os.environ["JUDGE_MODE"] = self._cfg.judge_mode

        loop = asyncio.get_running_loop()
        # A noop client (non-LLM run) has cost_cap_usd == 0; a real attacker has
        # None (unlimited) or a positive cap. No-LLM -> the planner uses its
        # vendored rule-based (client is None) paths and never calls the bridge.
        self._llm_available = getattr(llm_client, "_cost_cap_usd", None) != 0
        attacker = VendorLLMBridge(llm_client, loop) if self._llm_available else None

        judge, judge_model = attacker, (_VENDOR_MODEL if self._llm_available else "")
        if self._cfg.judge_llm_config is not None:
            judge = VendorLLMBridge(LLMClient(self._cfg.judge_llm_config), loop)
            judge_model = _VENDOR_MODEL
        validator = None
        validator_model = ""
        if self._cfg.validator_llm_config is not None:
            validator = VendorLLMBridge(LLMClient(self._cfg.validator_llm_config), loop)
            validator_model = _VENDOR_MODEL

        self._planner = Planner(
            self._cfg,
            rng=random.Random(self._cfg.seed),
            attacker=attacker,
            attacker_model=_VENDOR_MODEL if self._llm_available else "",
            judge=judge if judge is not attacker else None,
            judge_model=judge_model if judge is not attacker else "",
            validator=validator,
            validator_model=validator_model,
        )
        self._gen = self._planner.run(goal.description)
        self._primed = False
        self._done = False
        self._framework_succeeded = False
        self._current_unit = None
        self._pending_fresh_unit = None
        self._primary_pre_controllable = None
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return await self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        if self._done:
            return EventResponse(event=event)
        if not self._primed:
            self._primed = True
            first = await self._advance(None)
            if first is _DONE or first is None:
                self._done = True
                return EventResponse(event=event)
            self._current_unit = first
        elif self._pending_fresh_unit is not None:
            self._current_unit = self._pending_fresh_unit
            self._pending_fresh_unit = None
        # If neither branch set a unit, the run has nothing to inject (the loop
        # will decline and RunEnd will end the task).
        return EventResponse(event=event)

    async def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        name = event.controllable.name
        if name == _SYSTEM_PROMPT_NAME:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if name == _RESPONSE_NAME:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if self._primary_pre_controllable is None:
            self._primary_pre_controllable = event.controllable
        elif event.controllable != self._primary_pre_controllable:
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._done or self._current_unit is None:
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if self._awaiting_reply:
            # Resolve the reply for the message we last injected, then advance the
            # planner to decide what comes next.
            reply = self._read_reply()
            self._awaiting_reply = False
            self._pending_post_answer = None
            nxt = await self._advance(reply)
            if nxt is _DONE:
                self._done = True
                self._current_unit = None
                return ControllableNoInjection(event=event, controllable=event.controllable)
            assert isinstance(nxt, Unit)
            if nxt.fresh:
                # A fresh conversation needs a new run: end this one; the next
                # RunStart injects nxt.
                self._pending_fresh_unit = nxt
                self._current_unit = None
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._current_unit = nxt  # continuation of this run's conversation

        return self._inject_current(event)

    def _inject_current(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        unit = self._current_unit
        if unit is None or not unit.user_message.strip():
            return ControllableNoInjection(event=event, controllable=event.controllable)
        self._awaiting_reply = True
        self._last_pre_request = event.request
        self._last_injected_value = unit.user_message
        self._pending_post_answer = None
        return ControllableInjection(
            event=event, controllable=event.controllable, value=unit.user_message
        )

    def _handle_post_call(self, event: ControllablePostCallEvent) -> ControllableNoInjection:
        if self._awaiting_reply and self._matches_active_turn(event):
            self._pending_post_answer = event.answer
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        if event.evaluation is not None and event.evaluation.success:
            self._framework_succeeded = True

        # If a reply is still unresolved (the target ended without a follow-up
        # PreCall), resolve it and advance so the next run has a unit.
        if self._awaiting_reply and self._current_unit is not None:
            reply = self._read_reply()
            self._awaiting_reply = False
            self._pending_post_answer = None
            nxt = await self._advance(reply)
            self._current_unit = None
            if nxt is _DONE:
                self._done = True
            elif isinstance(nxt, Unit):
                # Whether fresh or a continuation, the run has ended, so it starts
                # the next run.
                self._pending_fresh_unit = nxt

        done = self._done or self._framework_succeeded
        return RunEndResponse(event=event, done=done)

    # ------------------------------------------------------------------
    # Planner advance (worker thread + budget/no-LLM handling)
    # ------------------------------------------------------------------

    async def _advance(self, reply: str | None) -> Any:
        """Advance the generator one step on a worker thread; returns Unit or _DONE."""
        try:
            return await asyncio.to_thread(_safe_send, self._gen, reply)
        except _BudgetSignal as sig:
            raise sig.original from sig
        except _NoLLMSignal:
            # Should not occur (no-LLM is detected up front), but degrade safely.
            logger.info("AttackAnything: unexpected no-LLM signal; ending task")
            return _DONE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_reply(self) -> str:
        recovered = self._read_response_from_trajectory()
        if recovered is None:
            recovered = self._pending_post_answer
        return recovered if recovered is not None else ""

    def _read_response_from_trajectory(self) -> str | None:
        traj = self.current_trajectory
        if traj is None:
            return None
        recovered: str | None = None
        for item in traj.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            if (
                name in _RESPONSE_OBSERVABLE_NAMES or name.lower() in _RESPONSE_OBSERVABLE_NAMES
            ) and isinstance(item.content, str):
                recovered = item.content
        return recovered

    def _matches_active_turn(self, event: ControllablePostCallEvent) -> bool:
        if self._last_pre_request is not None and event.request == self._last_pre_request:
            return True
        if self._last_injected_value is not None and event.request == self._last_injected_value:
            return True
        return event.controllable == self._primary_pre_controllable

    def _reset_run_state(self) -> None:
        self._awaiting_reply = False
        self._pending_post_answer = None
        self._last_pre_request = None
        self._last_injected_value = None
        self._primary_pre_controllable = None


__all__ = ["AttackAnythingOptimizer"]
