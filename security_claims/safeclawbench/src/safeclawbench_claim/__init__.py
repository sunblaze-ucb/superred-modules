"""SafeClawBench Semantic Core security claims for the OpenClaw target.

Grounds the OpenClaw security claims in SafeClawBench (arXiv 2606.18356;
dataset https://huggingface.co/datasets/sairights/safeclawbench, MIT): a
600-case, six-family staged benchmark for tool-using-agent security. This
package wires the **Semantic Core** endpoint — the prompt-level benchmark
whose primary metric is CoreFail@600 (an LLM judge's "did the response
semantically comply with the attack goal" label) — as superred Tasks against
the OpenClaw target.

The Exec-Balanced sandbox endpoint is a separate containerized protocol and is
intentionally out of scope here.
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
