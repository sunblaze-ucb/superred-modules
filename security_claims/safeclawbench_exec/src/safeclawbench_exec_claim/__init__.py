"""SafeClawBench Exec-Balanced (sandbox-harm) security claims for superred.

Grounds SafeClawBench's **sandbox-observed harm** endpoint (arXiv 2606.18356) in
the released 600-case executable set: each case runs the model-under-test through
a vendored offline mock sandbox (files / email / db / memory / web / code) via
:class:`~safeclawbench_exec_target.SafeClawBenchExecTarget`, and a deterministic
state oracle (``HarmCompletion-ASR``) scores whether observable harm occurred —
complementing the prompt-level Semantic Core claim (``safeclawbench-claim``).
"""

from __future__ import annotations

from safeclawbench_exec_claim.factory import (
    safeclawbench_exec_claim,
    safeclawbench_exec_combined_claim,
    safeclawbench_exec_family_claim,
    safeclawbench_exec_target_factory,
)
from safeclawbench_exec_claim.task import SafeClawBenchExecTask

__all__ = [
    "SafeClawBenchExecTask",
    "safeclawbench_exec_claim",
    "safeclawbench_exec_combined_claim",
    "safeclawbench_exec_family_claim",
    "safeclawbench_exec_target_factory",
]
