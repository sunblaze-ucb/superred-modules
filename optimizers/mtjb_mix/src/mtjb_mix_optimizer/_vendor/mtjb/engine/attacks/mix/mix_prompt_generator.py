import logging
from typing import override

from client.unified_llm_client import UnifiedLLMClient

from ...attack_type import AttackType
from ...attacks.coa.similarity_scorer import SimilarityScorer
from ...attacks.xteaming.updater_utils import CustomEngine
from ...core import ActionType, AttackContext
from ...interfaces.attack.prompt_generator import PromptGenerator
from .plan_generation import generate_plan, serve_crecendo_prompt, serve_xteaming_prompt
from .prompt_update import (
    update_actor,
    update_coa,
    update_cresendo,
    update_fitd,
    update_xteaming,
)

logger = logging.getLogger(__name__)


class MixPromptGenerator(PromptGenerator):

    def __init__(self, context: AttackContext, config: dict) -> None:
        super().__init__()
        self.generator_type = AttackType(config["generator"])
        self.updater_type = AttackType(config["updater"])

        attacker_cfg = config["attacker_model"]
        self.attacker_client = UnifiedLLMClient(
            model=attacker_cfg["model"],
            provider=attacker_cfg.get("provider"),
            base_url=attacker_cfg.get("base_url")
        )

        target_cfg = config.get("mirror_target_config")
        self.mirror_target_client = UnifiedLLMClient(
            model=target_cfg["model"],
            provider=target_cfg.get("provider"),
            base_url=target_cfg.get("base_url"),
        )

        # for COA, initialize similarity scorer
        self.similarity_scorer = None
        if self.generator_type == AttackType.COA or self.updater_type == AttackType.COA:
            self.use_llm_approx = config["use_llm_for_similarity"]
            self.similarity_scorer = SimilarityScorer(
                self.use_llm_approx,
                config["similarity_model"]
            )

        # for XTeaming, set up textgrad engine
        if self.updater_type == AttackType.X_TEAMING:
            import textgrad as tg
            tg.set_backward_engine(CustomEngine(attacker_cfg), override=True)

        # initialize attack plans
        num_strategies = config.get("max_restarts", 0) + 1
        self._init_attack_plan(context.harmful_behavior, context.max_turns, num_strategies)

    @override
    def next_prompt(self, context: AttackContext) -> str:
        match self.generator_type:
            # crescendo and xteaming dynamically modify query based on history
            case AttackType.CRESCENDO:
                return serve_crecendo_prompt(
                    context=context,
                    client=self.attacker_client,
                    restarted=self._restarted(context),
                )
            case AttackType.X_TEAMING:
                # if attack was restarted from turn 1, rotate to next available strategy
                if self._restarted(context):
                    strategy = next(self.backup)
                    self.attack_plan = list(strategy["conversation_plan"].values())
                    self.metadata = strategy

                return serve_xteaming_prompt(
                    planned_prompt=self.attack_plan[context.current_turn - 1],
                    strategy=self.metadata,
                    context=context,
                    client=self.attacker_client,
                )
            # actor, COA, FITD simply return planned prompt
            case AttackType.COA | AttackType.FITD | AttackType.ACTOR:
                return self.attack_plan[context.current_turn - 1]

    @override
    def refine_prompt(self, context: AttackContext) -> str:
        match self.updater_type:
            case AttackType.CRESCENDO:
                return update_cresendo(context, self.attacker_client)
            case AttackType.ACTOR:
                return update_actor(context, self.attacker_client, self.attack_plan)
            case AttackType.COA:
                return update_coa(context, self.attacker_client, self.similarity_scorer)
            case AttackType.FITD:
                return update_fitd(context, self.attacker_client, self.mirror_target_client, self.attack_plan)
            case AttackType.X_TEAMING:
                return update_xteaming(context)

    @override
    def system_prompt(self, context: AttackContext) -> str | None:
        return super().system_prompt(context)

    def _init_attack_plan(self, behavior: str, max_turns: int, num_strategies: int = 1) -> None:
        # self.attack_plan: basic turn-by-turn plan
        # self.metadata: some attacks may need to store additional info besides basic plan
        # self.backup: used for store backup plans (if generated)
        self.attack_plan, self.metadata, self.backup = generate_plan(
            attack=AttackType(self.generator_type),
            behavior=behavior,
            max_turns=max_turns,
            client=self.attacker_client,
            target_client=self.mirror_target_client,
            sim_scorer=self.similarity_scorer,
            num_strategies=num_strategies
        )

    def _restarted(self, context: AttackContext) -> bool:
        # checks if the attack was restarted from turn 1
        if len(context.full_history) <= 1:
            return False
        last_action = context.full_history[-2].next_action
        return last_action.type == ActionType.JUMP_TO and last_action.payload == 1
