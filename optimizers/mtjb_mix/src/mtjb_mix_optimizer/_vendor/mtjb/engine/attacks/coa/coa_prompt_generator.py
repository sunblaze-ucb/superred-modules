import ast
import json
import logging
from pathlib import Path
from typing import override

import yaml

from client.unified_llm_client import UnifiedLLMClient

from ...core import ActionType, AttackAttempt, AttackContext, AttackTurn
from ...interfaces.attack.prompt_generator import PromptGenerator
from .similarity_scorer import SimilarityScorer

logger = logging.getLogger(__name__)


class CoAPromptGenerator(PromptGenerator):
    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()

        with open(Path(__file__).parent / "prompts.yaml") as f:
            prompts = yaml.safe_load(f)
        self.mr_attack_chain_system_prompt = prompts["mr_attack_chain_system_prompt"]
        self.mr_attack_chain_init_message = prompts["mr_attack_chain_init_message"]
        self.update_system_prompt_tpl = prompts["update_single_round_system_prompt"]
        self.update_user_message_tpl = prompts["update_single_round_user_message"]

        self.use_llm_approx = config["use_llm_for_similarity"]
        self.similarity_scorer = SimilarityScorer(self.use_llm_approx, config["similarity_model"])

        self.enable_attack_update = config.get("enable_attack_update", True)
        self.n_init_chains = config.get("n_init_chains", 3)
        self.max_update_retries = config.get("max_update_retries", 5)
        self.theta = config.get("semantic_update_slack", 0.1)

        attacker_cfg = config["attacker_model"]
        self.client = UnifiedLLMClient(
            attacker_cfg["model"],
            attacker_cfg.get("provider"),
            attacker_cfg.get("base_url"),
            deterministic_mode=False
        )

        self.harmful_behavior = context.harmful_behavior
        self.max_turns = context.max_turns

        # prepare attack plan
        self.mr_init_chain: list[dict[str, str]] = self._get_mr_init_chain()
        self.current_prompts: list[str] = [item["prompt"] for item in self.mr_init_chain]

    @override
    def next_prompt(self, context: AttackContext) -> str | None:
        # special case: if this turn started after JUMP_TO, need to update the query
        # to prevent repeating the history
        last_epoch = context.full_history[-2] if len(context.full_history) >= 2 else None
        if last_epoch and last_epoch.next_action.type == ActionType.JUMP_TO:
            target = context.harmful_behavior
            turn = context.current_turn
            max_round = context.max_turns
            preset_prompt = self.mr_init_chain[context.current_turn - 1]["prompt"]
            response_text = last_epoch.response
            score = last_epoch.evaluation.score
            self.current_prompts[context.current_turn - 1] = self._update_attack(
                target, turn, max_round, preset_prompt, response_text, score
            )
    
        return self.current_prompts[context.current_turn - 1]

    @override
    def refine_prompt(self, context: AttackContext) -> str | None:
        if not self.enable_attack_update:
            raise RuntimeError("Attack update not enabled; shouldn't RETRY")

        target = context.harmful_behavior
        turn = context.current_turn
        max_round = context.max_turns
        # original COA repo uses query from init chain rather than current chain 
        preset_prompt = self.mr_init_chain[context.current_turn - 1]["prompt"]
        best_attempt = self._get_best_attempt_in_turn(context.get_current_turn())
        response_text = best_attempt.response
        score = best_attempt.evaluation.score

        self.current_prompts[context.current_turn - 1] = self._update_attack(
            target, turn, max_round, preset_prompt, response_text, score
        )
        return self.current_prompts[context.current_turn - 1]

    @override
    def system_prompt(self, context: AttackContext) -> str | None:
        return super().system_prompt(context)

    def _get_mr_init_chain(self) -> list[dict[str, str]]:
        """Generate the initial multi-round attack chain."""
        target = self.harmful_behavior
        max_round = self.max_turns

        system_prompt = self.mr_attack_chain_system_prompt.format(
            target=target,
            max_round=max_round,
        )
        init_msg_mr = self.mr_attack_chain_init_message.format(
            target=target,
            max_round=max_round,
        )

        init_chains: list[dict] = []

        retry_mr_init_chain = self.n_init_chains + 5     # buffer for failure
        for attempt in range(1, retry_mr_init_chain + 1):
            logger.debug("Init chain generation attempt %s", attempt)
            try:
                response, _ = self.client.generate(
                    init_msg_mr,
                    system_prompt=system_prompt,
                )
            except Exception:
                logger.exception("Error during chain generation attempt")
                continue

            try:
                mr_conv, evaluation, _ = self._extract_init_chain(response)
            except ValueError as exc:
                logger.warning("Discarding invalid chain response: %s", exc)
                continue

            # Require exact chain length to match the configured number of rounds.
            if len(mr_conv) != max_round:
                logger.warning(
                    "Discarding chain with wrong length: expected %s rounds, got %s",
                    max_round,
                    len(mr_conv),
                )
                continue

            init_chains.append({"mr_conv": mr_conv, "evaluation": evaluation})

            if len(init_chains) >= self.n_init_chains:
                break

        if not init_chains:
            raise RuntimeError("Failed to generate initial chain")

        ranked_init_chains = self._process_mr_init_chain(
            init_chains,
            target,
        )
        if not ranked_init_chains:
            raise RuntimeError("Failed to generate initial chain")

        best_chain = ranked_init_chains[0]["mr_conv"]

        logger.debug("Selected best chain:")
        for i, item in enumerate(best_chain, start=1):
            preview = item["prompt"][:80] + ("..." if len(item["prompt"]) > 80 else "")
            logger.debug("  Round %s: %s", i, preview)

        return best_chain

    def _process_mr_init_chain(self, mr_init_chain: list[dict], target: str) -> list[dict]:
        """Score and rank candidate multi-round chains."""
        processed_chain_list = []

        for item in mr_init_chain:
            processed_conv = []
            sem_scores = []

            # Preserve original round order.
            for conv_item in item["mr_conv"]:
                prompt = conv_item["prompt"]
                sem_score = self._compute_text_similarity(target, prompt)
                sem_scores.append(sem_score)

                processed_conv.append(
                    {
                        **conv_item,
                        "sem_score": sem_score,
                    }
                )

            eval_score = 0.0
            if item["evaluation"]:
                first_eval = item["evaluation"][0]
                if isinstance(first_eval, dict):
                    eval_score = float(first_eval.get("score", 0.0))

            processed_chain_list.append(
                {
                    "mr_conv": processed_conv,
                    "evaluation": item["evaluation"],
                    "sem_score": sem_scores,
                    "range_score": max(sem_scores) - min(sem_scores),
                    "eval_score": eval_score,
                }
            )

        processed_chain_list.sort(
            key=lambda x: (x["range_score"], x["eval_score"]),
            reverse=True,
        )
        return processed_chain_list[:1]

    def _extract_init_chain(self, response: str) -> tuple[list[dict[str, str]], list[dict], str]:
        """Extract structured chain data from a model response."""
        response = response.strip()
        start_pos = response.find("{")
        end_pos = response.rfind("}") + 1

        if start_pos == -1 or end_pos == 0:
            raise ValueError("No JSON structure found in chain response")

        json_str = response[start_pos:end_pos]

        parsed = None
        try:
            parsed = ast.literal_eval(json_str)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                raise ValueError("Failed to parse chain response JSON") from None

        if parsed is None or not isinstance(parsed, dict):
            raise ValueError("Chain response must parse to a dictionary")

        multi_round_conv: list[dict[str, str]] = []
        evaluation: list[dict] = []

        for key, item in parsed.items():
            if key == "evaluation":
                if isinstance(item, list):
                    evaluation.extend(item)
                else:
                    evaluation.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(f"Malformed chain item for key={key}")

            if not all(field in item for field in ["improvement", "prompt"]):
                raise ValueError(f"Missing keys in chain item for key={key}")

            improvement = item["improvement"]
            prompt = item["prompt"]

            if isinstance(improvement, list):
                improvement = improvement[0] if improvement else ""
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            prompt = str(prompt).strip()
            improvement = str(improvement).strip()

            if not prompt:
                raise ValueError(f"Empty prompt in chain item for key={key}")

            multi_round_conv.append(
                {
                    "improvement": improvement,
                    "prompt": prompt,
                }
            )

        if not multi_round_conv:
            raise ValueError("Chain response did not contain any prompts")

        return multi_round_conv, evaluation, json_str

    def _update_attack(self, target, turn, max_round, preset_prompt, response_text, score) -> str:
        """Generate an updated attack prompt for the current round."""
        system_prompt = self.update_system_prompt_tpl.format(
            preset_prompt=preset_prompt,
            target=target,
            round=turn,
            max_round=max_round,
        )
        prompt = self.update_user_message_tpl.format(
            response=response_text,
            preset_prompt=preset_prompt,
            target=target,
            round=round,
            max_round=max_round,
            score=score,
        )

        valid_output = None
        # candidate tracks the prompt with highest max similarity score
        candidate_output = None
        candidate_sem = 0.0

        pre_prompt_sem = self._compute_text_similarity(target, preset_prompt)

        for _ in range(self.max_update_retries):
            resp, _ = self.client.generate(prompt, system_prompt=system_prompt)

            try:
                attack_dict = self._extract_json(resp)
            except ValueError as exc:
                logger.warning("Discarding invalid updated prompt response: %s", exc)
                continue

            attack_prompt = attack_dict["prompt"]
            if isinstance(attack_prompt, list):
                attack_prompt = attack_prompt[0] if attack_prompt else ""
            attack_prompt = self._clean_prompt(attack_prompt)

            if not attack_prompt:
                continue

            prompt_sem = self._compute_text_similarity(target, attack_prompt)
            if prompt_sem > candidate_sem:
                candidate_output = attack_prompt
                candidate_sem = prompt_sem

            if prompt_sem >= pre_prompt_sem * (1 - self.theta):
                valid_output = attack_prompt
                break

        # Keep only the fallback that matches original behavior:
        # best generated candidate if no valid output passes threshold.
        if valid_output is not None:
            return valid_output

        if candidate_output is not None:
            logger.warning(
                "Fell back to best generated prompt "
                "(pre_prompt_sem=%.4f, candidate_sem=%.4f)",
                pre_prompt_sem,
                candidate_sem,
            )
            return candidate_output

        raise RuntimeError(
            f"Failed to generate updated prompt after {self.max_update_retries} retries"
        )

    def _get_best_attempt_in_turn(self, turn: AttackTurn) -> AttackAttempt:
        """Select the best attempt in a turn based on similarity."""
        attempts = turn.all_attempts
        best_attempt = None
        best_similarity = float("-inf")

        for attempt in attempts:
            if not attempt.response:
                continue

            similarity = None
            if attempt.evaluation and attempt.evaluation.similarity is not None:
                similarity = float(attempt.evaluation.similarity)

            if similarity is None:
                continue

            if similarity >= best_similarity:
                best_similarity = similarity
                best_attempt = attempt

        if best_attempt is None:
            raise RuntimeError("No valid attempt found for the current turn")

        return best_attempt

    def _extract_json(self, response: str) -> dict:
        """Extract the expected JSON payload from an update response."""
        start_pos = response.find("{")
        end_pos = response.rfind("}") + 1
        if start_pos == -1 or end_pos == 0:
            raise ValueError("No JSON structure found in update response")

        json_str = response[start_pos:end_pos].replace("\n", "")

        try:
            parsed = ast.literal_eval(json_str)
            if isinstance(parsed, dict) and all(k in parsed for k in ["improvement", "prompt"]):
                return parsed
        except (SyntaxError, ValueError):
            pass

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and all(k in parsed for k in ["improvement", "prompt"]):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        raise ValueError("Update response must be a JSON object with improvement and prompt")

    def _clean_prompt(self, prompt: str) -> str:
        """Normalize a generated prompt string."""
        prompt = str(prompt).strip()
        if prompt.startswith("['") and prompt.endswith("']"):
            prompt = prompt[2:-2]
        elif prompt.startswith("[") and prompt.endswith("]"):
            prompt = prompt[1:-1]
        return prompt.strip()

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        return self.similarity_scorer.similarity(text1, text2)