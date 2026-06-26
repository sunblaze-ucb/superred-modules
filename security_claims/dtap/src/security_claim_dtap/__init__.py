"""security_claim_dtap: the DecodingTrust-Agent (DTAP-BENCH) security claim.

One :class:`DtapTask` per DTAP per-task ``config.yaml`` (benign or malicious,
direct/indirect threat model), driving either DTAP agent target (Claude Code,
OpenClaw) through the target-agnostic DTAP config/query surface and the
byte-faithful out-of-band env-state judge. Hierarchical factories cover the
natural DTAP axes (per domain, threat model, risk category). The attack content
is an external attacker's (optimizer's) concern; the injection method is the
experiment's scope. Pairs with the ``dtap_scaffold`` package and the two
``dtap_*`` target packages.
"""

from __future__ import annotations

from security_claim_dtap.dataset import (
    GOLDEN_HASHES_PATH,
    build_golden_hashes,
    hash_task,
    iter_task_config_paths,
    load_golden_hashes,
    parse_task_config,
    resolve_dataset_root,
)
from security_claim_dtap.factory import (
    dtap_benign_claim,
    dtap_claim,
    dtap_claudecode_target_factory,
    dtap_combined_claim,
    dtap_direct_claim,
    dtap_domain_claim,
    dtap_indirect_claim,
    dtap_openclaw_target_factory,
    dtap_risk_claim,
)
from security_claim_dtap.task import REQUIRED_CONFIG_SLOTS, DtapTask

__version__ = "0.1.0"
__all__ = [
    # task
    "DtapTask",
    "REQUIRED_CONFIG_SLOTS",
    # factories
    "dtap_claim",
    "dtap_domain_claim",
    "dtap_direct_claim",
    "dtap_indirect_claim",
    "dtap_benign_claim",
    "dtap_risk_claim",
    "dtap_combined_claim",
    "dtap_claudecode_target_factory",
    "dtap_openclaw_target_factory",
    # dataset helpers
    "resolve_dataset_root",
    "iter_task_config_paths",
    "parse_task_config",
    "hash_task",
    "build_golden_hashes",
    "load_golden_hashes",
    "GOLDEN_HASHES_PATH",
]
