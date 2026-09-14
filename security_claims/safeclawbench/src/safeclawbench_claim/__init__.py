"""SafeClawBench Semantic Core cases and judge, run against the OpenClaw target.

Takes the Semantic Core challenge set and CoreFail judge from SafeClawBench
(arXiv 2606.18356; dataset https://huggingface.co/datasets/sairights/safeclawbench,
MIT, pinned at commit ``e6c29204c24a``): 600 cases, 100 in each of six
reporting families. Each case becomes a superred Task against the OpenClaw
agent, scored by the released judge's CoreFail label ("did the final response
semantically comply with the attack goal").

The paper's Semantic Core is one plain chat call with no function calling;
here the case runs through a full OpenClaw agent with its own tools and system
prompt, so rates are not comparable to the paper's CoreFail@600 (see the
README). The paper's Exec-Balanced sandbox endpoint is the separate
``safeclawbench_exec`` target and claim.
"""

from safeclawbench_claim.factory import (
    safeclawbench_claim,
    safeclawbench_combined_claim,
    safeclawbench_family_claim,
    safeclawbench_target_factory,
)
from safeclawbench_claim.judge import (
    HARM_EVIDENCE_CATEGORIES,
    JUDGE_PROMPT,
    CoreFailJudge,
    CoreFailVerdict,
    LLMCoreFailJudge,
    StubCoreFailJudge,
)
from safeclawbench_claim.loader import (
    AGENT_SYSTEM_PROMPT,
    ATTACK_FAMILIES,
    BENCHMARK_PATH,
    CASE_KEYS,
    FAMILY_NAMES,
    SafeClawCase,
    family_label,
    load_cases,
)
from safeclawbench_claim.task import SafeClawBenchTask

__all__ = [
    # loader
    "AGENT_SYSTEM_PROMPT",
    "ATTACK_FAMILIES",
    "BENCHMARK_PATH",
    "CASE_KEYS",
    "FAMILY_NAMES",
    # judge
    "HARM_EVIDENCE_CATEGORIES",
    "JUDGE_PROMPT",
    "CoreFailJudge",
    "CoreFailVerdict",
    "LLMCoreFailJudge",
    # task
    "SafeClawBenchTask",
    "SafeClawCase",
    "StubCoreFailJudge",
    "family_label",
    "load_cases",
    # factory
    "safeclawbench_claim",
    "safeclawbench_combined_claim",
    "safeclawbench_family_claim",
    "safeclawbench_target_factory",
]
