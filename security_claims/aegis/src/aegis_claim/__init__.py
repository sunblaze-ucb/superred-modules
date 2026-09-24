"""NVIDIA Aegis AI Content Safety benchmark as a superred ``SecurityClaim``.

The claim sends each unsafe user prompt from the Aegis 1.0 **test split** to a
:class:`chatbot_target.ChatbotTarget` and asks a bound safety judge whether the
target's *response* is unsafe per NVIDIA's Aegis content-safety taxonomy. Attack
success = the target produced unsafe content.

The judge is a **bound, generic safety judge** (flagged ``generic-judge``), NOT
NVIDIA's native Aegis-Guard classifier (see ``ASSUMPTIONS.md``).

Quick start::

    from chatbot_target import ChatbotTarget, USER_TAG, RESPONSE_READABLE_TAG
    from aegis_claim import aegis_claim
    from superred.core.controller import Controller, TargetFactory
    from superred.core.types.llm import LLMConfig

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(model="gpt-4o-mini", api_base=BASE, api_key=KEY),
        concurrency=8,
    )
    judge_cfg = LLMConfig(model="openai/gpt-4o", api_base=BASE, api_key=KEY)
    claim = aegis_claim(judge_llm_config=judge_cfg)   # 236 unsafe-prompt tasks
    controller = Controller(
        optimizer_factory=PassthroughOptimizer,       # any optimizer
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()                   # -> ThreatModelResult
"""

from __future__ import annotations

from aegis_claim.categories import (
    ALL_LABELS,
    HARM_CATEGORIES,
    NEEDS_CAUTION,
    SAFE,
    atomic_categories,
    is_unsafe,
    slugify,
)
from aegis_claim.factory import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    UNSAFE_COUNT_HARM_ONLY,
    UNSAFE_COUNT_WITH_NEEDS_CAUTION,
    aegis_claim,
)
from aegis_claim.judge import (
    AEGIS_SAFETY_RUBRIC,
    DEFAULT_JUDGE_MODEL,
    AegisSafetyJudge,
    Judge,
    RefusalHeuristicJudge,
    is_basic_refusal,
)
from aegis_claim.judge_failure import (
    JudgeFailedError,
    JudgeUnavailableError,
    JudgeUnparseableError,
)
from aegis_claim.loader import (
    AegisRow,
    load_all_rows,
    load_unsafe_user_prompts,
)
from aegis_claim.task import AegisTask

__all__ = [
    # factory
    "aegis_claim",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "DATASET_SPLIT",
    "UNSAFE_COUNT_WITH_NEEDS_CAUTION",
    "UNSAFE_COUNT_HARM_ONLY",
    # task
    "AegisTask",
    # judge
    "Judge",
    "AegisSafetyJudge",
    "RefusalHeuristicJudge",
    "AEGIS_SAFETY_RUBRIC",
    "DEFAULT_JUDGE_MODEL",
    "is_basic_refusal",
    # judge-failure policy (a judge that cannot answer raises these)
    "JudgeFailedError",
    "JudgeUnavailableError",
    "JudgeUnparseableError",
    # loader
    "AegisRow",
    "load_all_rows",
    "load_unsafe_user_prompts",
    # categories
    "HARM_CATEGORIES",
    "NEEDS_CAUTION",
    "SAFE",
    "ALL_LABELS",
    "atomic_categories",
    "is_unsafe",
    "slugify",
]
