"""SORRY-Bench safety-refusal benchmark as a superred ``SecurityClaim``.

Quick start::

    from chatbot_target import ChatbotTarget, USER_TAG, RESPONSE_READABLE_TAG
    from security_claim_sorry_bench import sorry_bench_claim
    from superred.core.controller import Controller, TargetFactory
    from superred.core.types.llm import LLMConfig

    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model="gpt-4o-2024-05-13", api_base=BASE, api_key=KEY,
        ),
        concurrency=8,
    )
    judge_cfg = LLMConfig(model="gpt-4-1106-preview", api_base=BASE,
                          api_key=KEY, max_cost=10.0)
    claim = sorry_bench_claim(
        target_model_id="gpt-4o-2024-05-13",
        judge_llm_config=judge_cfg,
        prompts_per_category=2,           # smoke run
    )
    controller = Controller(
        optimizer_factory=PassthroughOptimizer,    # any optimizer
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()                # -> ThreatModelResult
"""

from __future__ import annotations

from security_claim_sorry_bench.categories import (
    CATEGORY_DOMAINS,
    CATEGORY_NAMES,
    CATEGORY_NAMES_SHORTEST,
    category_domain,
    category_name,
    category_name_shortest,
)
from security_claim_sorry_bench.factory import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEFAULT_JUDGE_MODEL,
    SUBSET_A_QUESTION_IDS,
    SUBSET_B_QUESTION_IDS,
    Subset,
    sorry_bench_claim,
)
from security_claim_sorry_bench.judge import (
    BASE_SCOREONLY_TEMPLATE,
    BASE_THESCORE_TEMPLATE,
    Judge,
    LLMAsJudge,
    RefusalRegexJudge,
)
from security_claim_sorry_bench.system_prompts import sorry_bench_system_prompt
from security_claim_sorry_bench.task import SorryBenchTask

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
