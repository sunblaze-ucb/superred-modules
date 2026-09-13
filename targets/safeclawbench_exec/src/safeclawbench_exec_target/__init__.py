"""SafeClawBench Exec-Balanced target for superred.

Runs one executable SafeClawBench scenario against a model-under-test inside a
vendored, fully-offline mock sandbox (files / email / db / memory / web / code
stores + permissioned tools), capturing the final world state and trajectory.
The paired ``safeclawbench-exec-claim`` package scores the vendored deterministic
oracle over that capture — the paper's **sandbox-observed harm** endpoint,
complementing the prompt-level Semantic Core (``safeclawbench-claim``).

The vendored sandbox + oracle (schema / state / tools / trajectory / metrics)
and the 600-case dataset are re-exported here so the claim package can score
without a second copy.
"""

from __future__ import annotations

from safeclawbench_exec_target._vendor.executable.metrics import (
    CaseMetrics,
    evaluate_case_metrics,
)
from safeclawbench_exec_target._vendor.executable.schema import Scenario, StateOracle
from safeclawbench_exec_target._vendor.executable.trajectory import (
    TrajectoryEvent,
    TrajectoryLog,
)
from safeclawbench_exec_target.factory import safeclawbench_exec_target_factory
from safeclawbench_exec_target.loader import (
    ATTACK_FAMILIES,
    EXEC_FULL_PATH,
    FAMILY_NAMES,
    TINY_SUBSET_PATH,
    family_label,
    load_scenarios,
)
from safeclawbench_exec_target.security_tags import (
    DOMAIN,
    EXTERNAL_DATA_TAG,
    MEMORY_TAG,
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_INPUT_TAG,
)
from safeclawbench_exec_target.target import (
    DEFENSE_CHOICES,
    EXTERNAL_DATA_PATH,
    AgentModel,
    SafeClawBenchExecTarget,
)

__all__ = [
    # loader
    "ATTACK_FAMILIES",
    "DEFENSE_CHOICES",
    # security domain
    "DOMAIN",
    "EXEC_FULL_PATH",
    "EXTERNAL_DATA_PATH",
    "EXTERNAL_DATA_TAG",
    "FAMILY_NAMES",
    "MEMORY_TAG",
    "SYSTEM_TAG",
    "TINY_SUBSET_PATH",
    "TOOLS_TAG",
    "USER_INPUT_TAG",
    # target
    "AgentModel",
    # vendored sandbox + oracle (re-exported for the claim)
    "CaseMetrics",
    "SafeClawBenchExecTarget",
    "Scenario",
    "StateOracle",
    "TrajectoryEvent",
    "TrajectoryLog",
    "evaluate_case_metrics",
    "family_label",
    "load_scenarios",
    "safeclawbench_exec_target_factory",
]
