"""Policy Puppetry jailbreak optimizer for superred (ported from NVIDIA garak)."""

from policy_puppetry_optimizer.optimizer import PolicyPuppetryOptimizer
from policy_puppetry_optimizer.templates import (
    HARMFUL_BEHAVIOR_SLOT,
    TEMPLATE_COUNT,
    bypass_template,
    dr_house_template,
    leetspeak,
    render,
)

__all__ = [
    "HARMFUL_BEHAVIOR_SLOT",
    "TEMPLATE_COUNT",
    "PolicyPuppetryOptimizer",
    "bypass_template",
    "dr_house_template",
    "leetspeak",
    "render",
]
