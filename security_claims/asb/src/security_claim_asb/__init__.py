"""security_claim_asb: the Agent Security Bench claim for the ASB target.

One Task per (agent, benign task, attacker tool); ASB's attack-success /
utility / refusal predicates ported verbatim; hierarchical factories by
scenario, aggressiveness, and attack type. Pairs with the ``asb_target``
package (the injection method is chosen by the optimizer's scope; the
attack template by the optimizer's strategy).
"""

from __future__ import annotations

from security_claim_asb.categories import (
    ATTACK_TYPES,
    agent_names,
    load_agent_tasks,
    load_attack_tools,
    load_normal_tools,
)
from security_claim_asb.factory import (
    asb_agent_claim,
    asb_aggressive_claim,
    asb_claim,
    asb_combined_claim,
    asb_disruptive_claim,
    asb_non_aggressive_claim,
    asb_stealthy_claim,
    asb_target_factory,
)
from security_claim_asb.judge import (
    ASB_REFUSAL_SYSTEM_PROMPT,
    AsbRefusalJudge,
    RefusalJudge,
    StubRefusalJudge,
)
from security_claim_asb.task import AsbAttackTask

__all__ = [
    "AsbAttackTask",
    "asb_claim",
    "asb_agent_claim",
    "asb_aggressive_claim",
    "asb_non_aggressive_claim",
    "asb_stealthy_claim",
    "asb_disruptive_claim",
    "asb_combined_claim",
    "asb_target_factory",
    "AsbRefusalJudge",
    "StubRefusalJudge",
    "RefusalJudge",
    "ASB_REFUSAL_SYSTEM_PROMPT",
    "ATTACK_TYPES",
    "agent_names",
    "load_agent_tasks",
    "load_attack_tools",
    "load_normal_tools",
]
