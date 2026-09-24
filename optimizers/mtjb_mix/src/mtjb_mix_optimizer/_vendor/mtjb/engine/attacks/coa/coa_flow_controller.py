import logging
import random
from typing import override

from ...core.action import ActionType, NextAction
from ...core.context import AttackContext
from ...interfaces.attack.flow_controller import FlowController

logger = logging.getLogger(__name__)


"""
Adapted from https://github.com/YancyKahn/CoA

The original control flow implementation is a little sloppy. We've made some minor
changes to keep it safe under our engine while trying to be faithful.

CoA flow control compares semantic similarity metrics across three contexts:
  - response_sem: similarity of current response (with conversation history)
  - base_response_sem: similarity of current response (w/o conversation history)
  - posted_response_sem: similarity of conversation history only

Decision logic (from original RoundManager.get_action with methods="dp"):
  - response_sem >= base_response_sem & response_sem >= posted_response_sem
    → NEXT (advance to next round)
  - response_sem < posted_response_sem & response_sem < base_response_sem
    → BACK (go back one or two round)
  - otherwise
    → REGEN (regenerate prompt for current round)

When at max round and wanting to advance, randomly regen or go back.
When judge_score == 1 (binary success), terminate immediately.
"""

class CoAFlowController(FlowController):
    # probability thresholds matching original CoA's RoundManager
    BACK_WALK_RATE = 0.7
    REGEN_WALK_RATE = 0.8

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        self.use_attack_update = config["enable_attack_update"]

    @override
    def next_action(self, context: AttackContext) -> NextAction:
        current_attempt = context.get_current_attempt()
        evaluation = current_attempt.evaluation

        if evaluation is None:
            return NextAction(ActionType.CONTINUE)

        judge_score = evaluation.score

        # binary judge success: score == 1 means jailbreak succeeded
        if judge_score is not None and judge_score == 1:
            return self._end_walk()

        if not self.use_attack_update:
            return self._sequence_walk(context)

        metrics = evaluation.metrics or {}
        response_sem = metrics.get("response_sem", 0.0)
        base_response_sem = metrics.get("base_response_sem", 0.0)
        posted_response_sem = metrics.get("posted_response_sem", 0.0)

        # three-way DP comparison
        if response_sem >= base_response_sem and response_sem >= posted_response_sem:
            return self._next_walk(context)
        elif response_sem < posted_response_sem and response_sem < base_response_sem:
            return self._back_walk(context, step=1)
        else:
            return self._regen_walk()

    def _next_walk(self, context: AttackContext) -> NextAction:
        """original: RoundManager.next_walk()"""
        if context.current_turn >= context.max_turns:
            # At max round, retry or jump back
            # original logic: 80% regen, 20% back
            if random.random() <= self.REGEN_WALK_RATE:
                return self._regen_walk()
            else:
                # original logic: 70% back 1 step, 30% back 2 steps
                step = 1 if random.random() <= self.BACK_WALK_RATE else 2
                return self._back_walk(context, step)

        return NextAction(ActionType.CONTINUE)

    def _end_walk(self) -> NextAction:
        """original: RoundManager.end_walk()"""
        return NextAction(ActionType.END_SUCCESS)

    def _back_walk(self, context: AttackContext, step: int = 1) -> NextAction:
        """original: RoundManager.back_walk()"""
        # behavior is slightly different from original COA to be safe under this framework
        target_turn = max(1, context.current_turn - step)
        return NextAction(ActionType.JUMP_TO, target_turn)

    def _regen_walk(self) -> NextAction:
        """original: RoundManager.regen_walk()"""
        return NextAction(ActionType.RETRY)

    def _sequence_walk(self, context: AttackContext) -> NextAction:
        """original: RoundManager.sequence_walk()"""
        if context.current_turn >= context.max_turns:
            return NextAction(ActionType.END_FAILURE)
        return NextAction(ActionType.CONTINUE)
