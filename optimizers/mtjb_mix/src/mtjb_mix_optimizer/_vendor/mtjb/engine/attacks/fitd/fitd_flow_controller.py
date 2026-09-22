from typing import override

from client.unified_llm_client import UnifiedLLMClient

from ...core import ActionType, AttackContext, NextAction
from ...interfaces.attack.flow_controller import FlowController
from .format_template import format_prompt


class FitdFlowController(FlowController):
    """
    Flow controller for FITD.

    The main difference between our flow controller and original FITD is around
    simulation of Align and SlipperySlope. Our engine doesn't support modifying
    history, so we use JUMP_TO(previous) to align previous turn, and RETRY to do
    slippery slope (i.e. intermediate turn).
    """

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        attacker_cfg = config["attacker_model"]
        self.client = UnifiedLLMClient(
            attacker_cfg["model"],
            attacker_cfg.get("provider"),
            attacker_cfg.get("base_url"),
            deterministic_mode=False
        )

    def _detect_align(self, prev_turn_prompt, prev_turn_response) -> bool:
        """
        Mirrors fitd/utils.py detect_align:
        Returns True if the response is aligned with the prompt's intent.
        """
        user_content = format_prompt("detect", prompt=prev_turn_prompt, response=prev_turn_response)
        resp, _ = self.client.generate(user_input=user_content, temperature=0.0)
        return (resp or "").strip().lower() == "yes"

    @override
    def next_action(self, context: AttackContext) -> NextAction:
        evaluation = context.get_current_attempt().evaluation

        if evaluation.score == 1:
            return NextAction(ActionType.END_SUCCESS)

        # if final turn fails, do a final align; if it still fails, terminate 
        if context.current_turn == context.max_turns:
            if context.current_turn in context.custom.get("skip_align", []):
                return NextAction(ActionType.END_FAILURE)

            if len(context.get_current_turn().all_attempts) == 1:
                prev = context.get_turn(context.current_turn - 1).attempt_in_effect
                context.custom["align"] = True
                context.custom["last_prompt"] = prev.prompt
                context.custom["last_response"] = prev.response
                return NextAction(ActionType.JUMP_TO, context.current_turn - 1)                

        is_refusal = evaluation.metrics.get("is_refusal", False)
        if is_refusal:
            # if force_continue (i.e. slippery slope failed), don't retry
            if context.custom.get("force_continue", False):
                del context.custom["force_continue"]
                return NextAction(ActionType.CONTINUE)
            
            # we can't align previous turn at turn 1
            if context.current_turn > 1:
                # if previous turn needs alignment, JUMP_TO previous turn
                prev = context.get_turn(context.current_turn - 1).attempt_in_effect
                is_aligned = self._detect_align(prev.prompt, prev.response)
                if not is_aligned and context.current_turn not in context.custom.get("skip_align", []):
                    # since JUMP_TO will wipe the previous turn, save these values
                    context.custom["align"] = True
                    context.custom["last_prompt"] = prev.prompt
                    context.custom["last_response"] = prev.response
                    return NextAction(ActionType.JUMP_TO, context.current_turn - 1)

            return NextAction(ActionType.RETRY)

        return NextAction(ActionType.CONTINUE)
