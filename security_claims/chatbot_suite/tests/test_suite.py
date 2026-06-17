"""Integration tests for chatbot_suite_claim against the real benchmarks.

HarmBench and StrongREJECT ship their data in-package and build offline
(judges only fire at evaluation time, never at construction). SORRY-Bench
needs its dataset (HF cache or an explicit path), so the full-suite test is
skipped when that dataset is unavailable.
"""

from __future__ import annotations

from collections import Counter

import pytest

from security_claim_chatbot_suite import TaskRecord, chatbot_suite_claim

_JUDGE_BASE = "http://example.invalid"  # never called during construction
_JUDGE_KEY = "sk-test"


def test_harmbench_plus_strongreject_offline() -> None:
    manifest: list[TaskRecord] = []
    claim = chatbot_suite_claim(
        target_model_id="openai/gpt-5.4-mini-2026-03-17",
        judge_api_base=_JUDGE_BASE,
        judge_api_key=_JUDGE_KEY,
        per_category=1,
        include_sorrybench=False,
        manifest_out=manifest,
    )
    tasks = list(claim)
    # 7 HarmBench semantic categories + 6 StrongREJECT categories.
    assert len(tasks) == 13
    by_source = Counter(r.source for r in manifest)
    assert by_source["harmbench"] == 7
    assert by_source["strongreject"] == 6
    # categories are namespaced by attribute, never collide across benchmarks
    assert all(("=" in r.category) for r in manifest)


def test_disabling_all_sources_raises() -> None:
    with pytest.raises(ValueError):
        chatbot_suite_claim(
            target_model_id="m",
            judge_api_base=_JUDGE_BASE,
            judge_api_key=_JUDGE_KEY,
            include_harmbench=False,
            include_sorrybench=False,
            include_strongreject=False,
        )


def test_per_category_scales_subset_size() -> None:
    claim1 = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JUDGE_BASE,
        judge_api_key=_JUDGE_KEY,
        per_category=1,
        include_sorrybench=False,
    )
    claim2 = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JUDGE_BASE,
        judge_api_key=_JUDGE_KEY,
        per_category=2,
        include_sorrybench=False,
    )
    assert len(list(claim2)) > len(list(claim1))


def _sorry_dataset_available() -> bool:
    try:
        from security_claim_sorry_bench import sorry_bench_claim
        from superred.core.types.llm import LLMConfig

        sorry_bench_claim(
            target_model_id="openai/gpt-4o-mini",
            judge_llm_config=LLMConfig(
                model="openai/gpt-4o-mini",
                api_base=_JUDGE_BASE,
                api_key=_JUDGE_KEY,
                max_cost=None,
            ),
            prompts_per_category=1,
        )
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _sorry_dataset_available(),
    reason="SORRY-Bench dataset not available (HF cache / path)",
)
def test_full_suite_with_sorrybench() -> None:
    manifest: list[TaskRecord] = []
    claim = chatbot_suite_claim(
        target_model_id="openai/gpt-5.4-mini-2026-03-17",
        judge_api_base=_JUDGE_BASE,
        judge_api_key=_JUDGE_KEY,
        per_category=1,
        manifest_out=manifest,
    )
    tasks = list(claim)
    by_source = Counter(r.source for r in manifest)
    assert by_source["sorrybench"] == 44
    assert by_source["harmbench"] == 7
    assert by_source["strongreject"] == 6
    assert len(tasks) == 57
