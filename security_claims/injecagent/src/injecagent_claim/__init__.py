"""injecagent_claim: the InjecAgent benchmark claim.

One Task per InjecAgent test case (1,054 base cases across direct-harm and
data-stealing; an ``enhanced`` setting reinforces each injection with a hacking
prompt). Upstream's deterministic ReAct / function-calling attack-success
evaluators and ASR scoring are ported verbatim into a pure, LLM-free judge.
Pairs with the ``injecagent_target`` package (the ReAct agent-under-test driver
+ tool environment + ds two-step).

Provenance: ported from ``uiuc-kang-lab/InjecAgent`` @ ``f19c9f2`` (MIT,
(c) 2023 Qiusi Zhan). Cite Zhan et al., ACL Findings 2024 (arXiv:2403.02691).
"""

from __future__ import annotations

from injecagent_claim.categories import (
    ATTACKS,
    SETTINGS,
    load_cases,
    task_counts,
)
from injecagent_claim.factory import (
    injecagent_claim,
    injecagent_dh_claim,
    injecagent_ds_claim,
    injecagent_enhanced_claim,
)
from injecagent_claim.judge import (
    OutputParseError,
    Verdict,
    evaluate_output_finetuned,
    evaluate_output_prompted,
    evaluate_task,
    get_score,
    output_parser,
)
from injecagent_claim.security_tags import (
    DIMENSION_TAG,
    DOMAIN,
    EXTERNAL_DATA_TAG,
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from injecagent_claim.task import InjecAgentTask

__all__ = [
    "ATTACKS",
    "SETTINGS",
    "load_cases",
    "task_counts",
    "InjecAgentTask",
    "injecagent_claim",
    "injecagent_dh_claim",
    "injecagent_ds_claim",
    "injecagent_enhanced_claim",
    "OutputParseError",
    "Verdict",
    "output_parser",
    "evaluate_output_prompted",
    "evaluate_output_finetuned",
    "get_score",
    "evaluate_task",
    "DOMAIN",
    "SYSTEM_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "TOOLS_TAG",
    "DIMENSION_TAG",
]
