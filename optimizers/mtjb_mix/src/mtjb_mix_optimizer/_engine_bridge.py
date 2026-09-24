"""Drives MT-JailBench's vendored Mix engine inside superred's event model.

This module is imported lazily by the optimizer. It does **not** import the
vendored engine (or ``textgrad``) at module load; that happens inside the worker
thread in :meth:`MixEngineBridge._start_sync`, so a missing ``textgrad`` surfaces
only when the engine is actually started (and the optimizer degrades cleanly).

The vendored generator / outcome-evaluator / flow-controller are synchronous and
expect to call the model directly; superred's ``LLMClient`` is async and the
target is external. So:

* the whole engine step runs in a worker thread (:func:`asyncio.to_thread`);
* model calls route through the ``client.unified_llm_client`` shim, whose facade
  bridges back to the running loop via ``run_coroutine_threadsafe`` (bound per
  task with :func:`_client_shim.bind_llm`);
* the target is never called here -- the optimizer injects the prompt and feeds
  the observed response back via :meth:`observe`.

Because superred cannot rewind the target conversation, upstream ``RETRY`` sends
the refined prompt as the next turn (no rewind) and ``JUMP_TO`` ends the attack.
The rest of the per-attempt bookkeeping mirrors ``jailbreak_engine`` faithfully.

This path is exercised only with the ``refine`` extra installed; its test is
env-gated.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from superred.core.llm import LLMClient

from mtjb_mix_optimizer._client_shim import Conversation, bind_llm, install

logger = logging.getLogger(__name__)

_DUMMY_REFUSAL = "I'm sorry, I can't help with that request."


class MixEngineBridge:
    """Owns the vendored engine objects + AttackContext and advances them."""

    def __init__(
        self, *, config: dict, behavior: str, max_total_turns: int, max_turns: int
    ) -> None:
        self._config = config
        self._behavior = behavior
        self._max_total_turns = max_total_turns
        self._max_turns = max_turns
        self._gen: Any = None
        self._eval: Any = None
        self._flow: Any = None
        self._ctx: Any = None
        self._inflight: Any = None
        self._turns_sent = 0
        # Vendored types, filled on start.
        self._AttackAttempt: Any = None
        self._AttackTurn: Any = None
        self._ActionType: Any = None

    # ------------------------------------------------------------------
    async def start(self, llm: LLMClient) -> str | None:
        loop = asyncio.get_running_loop()
        with bind_llm(llm, loop):
            return await asyncio.to_thread(self._start_sync)

    async def observe(self, llm: LLMClient, response: str) -> tuple[str | None, bool, bool]:
        loop = asyncio.get_running_loop()
        with bind_llm(llm, loop):
            return await asyncio.to_thread(self._observe_sync, response)

    # ------------------------------------------------------------------
    def _start_sync(self) -> str | None:
        install()  # register the client.unified_llm_client shim

        from mtjb_mix_optimizer._vendor.mtjb.engine.attack_type import AttackType
        from mtjb_mix_optimizer._vendor.mtjb.engine.attacks.mix.mix_flow_controller import (
            MixFlowController,
        )
        from mtjb_mix_optimizer._vendor.mtjb.engine.attacks.mix.mix_outcome_evaluator import (
            MixOutcomeEvaluator,
        )
        from mtjb_mix_optimizer._vendor.mtjb.engine.attacks.mix.mix_prompt_generator import (
            MixPromptGenerator,
        )
        from mtjb_mix_optimizer._vendor.mtjb.engine.core.action import ActionType
        from mtjb_mix_optimizer._vendor.mtjb.engine.core.context import AttackContext
        from mtjb_mix_optimizer._vendor.mtjb.engine.core.outcome import AttackOutcome, OutcomeCode
        from mtjb_mix_optimizer._vendor.mtjb.engine.core.turn import AttackAttempt, AttackTurn

        self._AttackAttempt = AttackAttempt
        self._AttackTurn = AttackTurn
        self._ActionType = ActionType

        self._ctx = AttackContext(
            harmful_behavior_id="mtjb-0",
            harmful_behavior=self._behavior,
            max_turns=self._max_turns,
            max_epochs=self._max_total_turns,
            history=[AttackTurn(turn=1)],
            full_history=[],
            current_turn=1,
            current_epoch=0,
            conversation=None,
            attack_outcome=AttackOutcome(False, OutcomeCode.MAX_TURN_REACHED),
            custom={},
        )
        # Constructing the generator runs plan generation (attacker-LLM calls).
        self._gen = MixPromptGenerator(self._ctx, self._config)
        self._eval = MixOutcomeEvaluator(self._ctx, self._config)
        self._flow = MixFlowController(self._ctx, self._config)
        _ = AttackType  # imported for provenance / config validation parity
        return self._produce_prompt()

    def _produce_prompt(self) -> str | None:
        ctx = self._ctx
        ctx.current_epoch += 1
        turn = ctx.history[ctx.current_turn - 1]
        attempt = self._AttackAttempt(epoch=ctx.current_epoch)
        turn.all_attempts.append(attempt)
        turn.attempt_in_effect = attempt
        ctx.full_history.append(attempt)

        attempt.system_prompt = self._gen.system_prompt(ctx)
        if len(turn.all_attempts) > 1:
            attempt.prompt = self._gen.refine_prompt(ctx)
        else:
            attempt.prompt = self._gen.next_prompt(ctx)
        self._inflight = attempt
        return attempt.prompt

    def _observe_sync(self, response: str) -> tuple[str | None, bool, bool]:
        ctx = self._ctx
        attempt = self._inflight
        if attempt is None:
            return None, True, False
        attempt.response = response if (response and response.strip()) else _DUMMY_REFUSAL

        # Record the exchange into the conversation (no-defense canonical form).
        system_prompt = attempt.system_prompt
        if system_prompt is None and ctx.conversation is not None:
            system_prompt = ctx.conversation.system_prompt
        conv = Conversation(system_prompt=system_prompt)
        if ctx.conversation is not None:
            conv.history = list(ctx.conversation.history)
        conv.add_user_message(str(attempt.prompt or ""))
        conv.add_assistant_message(str(attempt.response))
        attempt._conv_after_response = conv
        ctx.conversation = conv

        attempt.evaluation = self._eval.evaluate(ctx)
        attempt.next_action = self._flow.next_action(ctx)
        self._turns_sent += 1

        at = self._ActionType
        action = attempt.next_action.type
        turn = ctx.history[ctx.current_turn - 1]

        if action == at.END_SUCCESS:
            return None, True, True
        if action == at.END_FAILURE:
            return None, True, False
        if self._turns_sent >= self._max_total_turns:
            return None, True, False

        if action == at.CONTINUE:
            payload = attempt.next_action.payload
            if payload is not None:
                match = next((a for a in turn.all_attempts if a.epoch == payload), None)
                if match is not None:
                    turn.attempt_in_effect = match
                    ctx.conversation = match._conv_after_response
            if ctx.current_turn >= self._max_turns:
                return None, True, False  # plan exhausted
            ctx.current_turn += 1
            ctx.history.append(self._AttackTurn(turn=ctx.current_turn))
            return self._produce_prompt(), False, False

        if action == at.RETRY:
            # No rewind: the refined prompt becomes the next turn; the generator's
            # refine_prompt fires because the current turn now has >1 attempts.
            return self._produce_prompt(), False, False

        # JUMP_TO (or anything else): no rewind is possible, so end the attack.
        return None, True, False
