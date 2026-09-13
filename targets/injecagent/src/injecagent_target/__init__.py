"""injecagent_target: a superred Target driving the agent-under-test through the
InjecAgent indirect-prompt-injection benchmark.

Builds the ReAct (or native function-calling) prompt for one InjecAgent test case,
calls the model, and runs the data-stealing two-step against a fully simulated
tool environment (no Docker) — the poisoned tool observation is baked into the
case and the ds step-2 exfiltration response is served cache-first from the
vendored cache. Pairs with the ``injecagent-claim`` package, whose pure judge
grades the captured outputs (and whose evaluator this target reuses to gate ds
step 2).

Provenance: ported from ``uiuc-kang-lab/InjecAgent`` @ ``f19c9f2`` (MIT). The
vendored tool schemas (``tools.json``) are adapted from ToolEmu (Ruan et al.,
2023, Apache-2.0); see NOTICE.
"""

from __future__ import annotations

from injecagent_target.factory import injecagent_target_factory
from injecagent_target.security_tags import (
    DIMENSION_TAG,
    DOMAIN,
    EXTERNAL_DATA_TAG,
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from injecagent_target.target import (
    ATTACK_TYPE_OBS,
    ATTACKER_INSTRUCTION_CTRL,
    MODE_OBS,
    MODEL_IDENTITY_OBS,
    PROMPT_TYPE_OBS,
    SETTING_OBS,
    USER_INSTRUCTION_CTRL,
    InjecAgentTarget,
)

__all__ = [
    "InjecAgentTarget",
    "injecagent_target_factory",
    "DOMAIN",
    "DIMENSION_TAG",
    "SYSTEM_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "TOOLS_TAG",
    "ATTACKER_INSTRUCTION_CTRL",
    "USER_INSTRUCTION_CTRL",
    "MODEL_IDENTITY_OBS",
    "ATTACK_TYPE_OBS",
    "SETTING_OBS",
    "PROMPT_TYPE_OBS",
    "MODE_OBS",
]
