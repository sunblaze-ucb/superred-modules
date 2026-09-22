"""
X-Teaming outcome evaluator: scores target responses with a judge (xteaming 1-5 scale).
Identical to CrescendoOutcomeEvaluator — both use the same xteaming judge type.
"""
import json
import logging
import os
from typing import override

import yaml

from client.unified_llm_client import UnifiedLLMClient

from ...core import AttackContext, TurnEvaluation
from ...interfaces.attack.outcome_evaluator import OutcomeEvaluator

logger = logging.getLogger(__name__)


class XTeamingOutcomeEvaluator(OutcomeEvaluator):
   
    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        jc = config.get("judge_model", {})
        self.client = UnifiedLLMClient(
            model=jc["model"],
            provider=jc.get("provider"),
            base_url=jc.get("base_url"),
        )
        prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        path = os.path.join(prompts_dir, "evaluation_prompt.yaml")
        with open(path) as f:
            self.template = yaml.safe_load(f)["prompt"]

    @override
    def evaluate(self, context: AttackContext) -> TurnEvaluation:
        attempt = context.get_current_attempt()
        response = attempt.response or ""
        behavior = context.harmful_behavior

        if not response.strip():
            raise RuntimeError("No response found")
        
        prompt = self._build_judge_prompt(behavior, response)
        response, _ = self.client.generate(prompt, response_format={"type": "json_object"})
        result = json.loads(response)
        score = int(result["score"])
        reason = result["reason"]
        if score < 1 or score > 5:
            raise RuntimeError("Invalid score from judge")
        return TurnEvaluation(score=score, reason=reason or "")

    def _build_judge_prompt(self, behavior: str, response: str) -> str:
        """
        Formats judge prompt by replacing `{behavior}` and `{response}` in template.
        """
        return self.template.format(
            behavior=behavior.strip(),
            response=response.strip()
        )