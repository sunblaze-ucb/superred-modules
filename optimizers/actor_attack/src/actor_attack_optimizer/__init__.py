"""ActorAttack multi-turn optimizer for superred.

Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard).
"""

from actor_attack_optimizer.optimizer import (
    DEFAULT_MAX_TURNS_PER_ACTOR,
    DEFAULT_NUM_ACTORS,
    ActorAttackOptimizer,
)
from actor_attack_optimizer.parsing import Actor

__all__ = [
    "DEFAULT_MAX_TURNS_PER_ACTOR",
    "DEFAULT_NUM_ACTORS",
    "Actor",
    "ActorAttackOptimizer",
]
