"""Policy Puppetry jailbreak optimizer for superred (ported from NVIDIA garak)."""

from policy_puppetry_optimizer.optimizer import PolicyPuppetryOptimizer
from policy_puppetry_optimizer.templates import (
    HARMFUL_BEHAVIOR_SLOT,
    dr_house_template,
    leetspeak,
    render,
)

__all__ = [
    "HARMFUL_BEHAVIOR_SLOT",
    "PolicyPuppetryOptimizer",
    "dr_house_template",
    "leetspeak",
    "render",
]
