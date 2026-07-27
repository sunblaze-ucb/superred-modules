"""SORRY-Bench safety-refusal benchmark as a superred ``SecurityClaim``.

Quick start::

    from chatbot_target import ChatbotTarget, USER_TAG, MODEL_TAG
    from sorry_bench_claim import sorry_bench_claim
    from superred.core.controller import Controller, TargetFactory
    from superred.core.types.llm import LLMConfig

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model="gpt-4o-2024-05-13", api_base=BASE, api_key=KEY,
        ),
        concurrency=8,
    )
    judge_cfg = LLMConfig(model="gpt-4-1106-preview", api_base=BASE,
                          api_key=KEY)
    claim = sorry_bench_claim(
        target_model_id="gpt-4o-2024-05-13",
        judge_llm_config=judge_cfg,
        prompts_per_category=2,           # smoke run
    )
    controller = Controller(
        optimizer_factory=PassthroughOptimizer,    # any optimizer
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG}),
        read_only=frozenset({MODEL_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()                # -> ThreatModelResult
"""

from __future__ import annotations

from sorry_bench_claim.categories import (
    CATEGORY_DOMAINS,
    CATEGORY_NAMES,
    CATEGORY_NAMES_SHORTEST,
    category_domain,
    category_name,
    category_name_shortest,
)
from sorry_bench_claim.factory import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_JUDGE_MODEL,
    SUBSET_A_QUESTION_IDS,
    SUBSET_B_QUESTION_IDS,
    Subset,
    sorry_bench_claim,
)
from sorry_bench_claim.judge import (
    BASE_SCOREONLY_TEMPLATE,
    BASE_THESCORE_TEMPLATE,
    Judge,
    LLMAsJudge,
    RefusalRegexJudge,
)
from sorry_bench_claim.system_prompts import sorry_bench_system_prompt
from sorry_bench_claim.task import SorryBenchTask

__all__ = [
    # factory
    "sorry_bench_claim",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "DATASET_FILENAME",
    "DEFAULT_JUDGE_MODEL",
    "SUBSET_A_QUESTION_IDS",
    "SUBSET_B_QUESTION_IDS",
    "Subset",
    # task
    "SorryBenchTask",
    # judge
    "Judge",
    "LLMAsJudge",
    "RefusalRegexJudge",
    "BASE_THESCORE_TEMPLATE",
    "BASE_SCOREONLY_TEMPLATE",
    # categories
    "CATEGORY_NAMES",
    "CATEGORY_NAMES_SHORTEST",
    "CATEGORY_DOMAINS",
    "category_name",
    "category_name_shortest",
    "category_domain",
    # system prompts
    "sorry_bench_system_prompt",
]
