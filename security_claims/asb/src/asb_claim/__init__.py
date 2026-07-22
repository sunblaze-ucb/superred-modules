"""asb_claim: the Agent Security Bench claim for the ASB target.

One Task per (agent, benign task, attacker tool); ASB's attack-success /
utility / refusal predicates ported verbatim; hierarchical factories by
scenario, aggressiveness, and attack type. Pairs with the ``asb_target``
package (the injection method is selected by the experiment's scope; the
attack content is an external attacker's concern, not the claim's).
"""

from __future__ import annotations

from asb_claim.categories import (
    ATTACK_TYPES,
    agent_names,
    load_agent_tasks,
    load_attack_tools,
    load_normal_tools,
)
from asb_claim.factory import (
    asb_agent_claim,
    asb_aggressive_claim,
    asb_claim,
    asb_combined_claim,
    asb_disruptive_claim,
    asb_full_claim,
    asb_non_aggressive_claim,
    asb_stealthy_claim,
    asb_target_factory,
)
from asb_claim.judge import (
    ASB_ADVERSARIAL_SYSTEM_PROMPT,
    ASB_REFUSAL_SYSTEM_PROMPT,
    AsbAdversarialJudge,
    AsbRefusalJudge,
    RefusalJudge,
    StubAdversarialJudge,
    StubRefusalJudge,
    SuccessJudge,
)
from asb_claim.task import AsbAttackTask

__all__ = [
    "AsbAttackTask",
    "asb_claim",
    "asb_agent_claim",
    "asb_full_claim",
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
    "AsbAdversarialJudge",
    "StubAdversarialJudge",
    "SuccessJudge",
    "ASB_ADVERSARIAL_SYSTEM_PROMPT",
    "ATTACK_TYPES",
    "agent_names",
    "load_agent_tasks",
    "load_attack_tools",
    "load_normal_tools",
]
