"""Integration tests for chatbot_suite_claim (the canonical static build).

HarmBench and StrongREJECT ship their data in-package and build offline (judges
fire only at evaluation time, never at construction; the static dedup needs no
embedding). The full canonical suite additionally needs the SORRY-Bench dataset,
so that test is skipped when it is unavailable.
"""

from __future__ import annotations

from collections import Counter

import pytest

from security_claim_chatbot_suite import (
    CombineStats,
    TaskRecord,
    chatbot_suite_claim,
)

_JB = "http://example.invalid"  # never called during construction
_JK = "sk-test"


class _FakeIdenticalEmbedder:
    """One vector for everything: every task after the first is a semantic dup."""

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_disabling_all_sources_raises() -> None:
    with pytest.raises(ValueError):
        chatbot_suite_claim(
            target_model_id="m",
            judge_api_base=_JB,
            judge_api_key=_JK,
            include_harmbench=False,
            include_sorrybench=False,
            include_strongreject=False,
        )


def test_canonical_offline_composition() -> None:
    # HarmBench + StrongREJECT at the canonical per-benchmark defaults; static
    # dedup off to see the raw composition. (SORRY omitted to stay offline.)
    man: list[TaskRecord] = []
    chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_sorrybench=False,
        apply_static_dedup=False,
        manifest_out=man,
    )
    by = Counter(r.source for r in man)
    assert by["harmbench"] == 18  # 3 x 6 semantic categories, copyright excluded
    assert by["strongreject"] == 18  # 3 x 6 categories from the small set
    assert all("copyright" not in r.category for r in man if r.source == "harmbench")


def test_static_dedup_drops_known_duplicate() -> None:
    common = dict(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_sorrybench=False,
        include_strongreject=False,
    )
    raw: list[TaskRecord] = []
    chatbot_suite_claim(**common, apply_static_dedup=False, manifest_out=raw)
    deduped: list[TaskRecord] = []
    stats: list[CombineStats] = []
    chatbot_suite_claim(
        **common, apply_static_dedup=True, manifest_out=deduped, stats_out=stats
    )
    # The committed static set removes exactly the one HarmBench near-duplicate,
    # without the per-category cap back-filling its slot.
    assert len(raw) == 18
    assert len(deduped) == 17
    assert stats[0].dropped_static == 1
    assert not any("graphic detail" in r.goal_preview for r in deduped)


def test_copyright_exclusion_default() -> None:
    incl = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_sorrybench=False,
        include_strongreject=False,
        harmbench_per_category=1,
        harmbench_exclude_copyright=False,
        apply_static_dedup=False,
    )
    excl = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_sorrybench=False,
        include_strongreject=False,
        harmbench_per_category=1,
        apply_static_dedup=False,  # harmbench_exclude_copyright defaults True
    )
    assert len(list(incl)) == 7  # 7 semantic categories incl copyright
    assert len(list(excl)) == 6  # copyright excluded by default


def test_embedder_recompute_path() -> None:
    # Passing an embedder recomputes semantic dups (the regeneration path).
    claim = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_sorrybench=False,
        apply_static_dedup=False,
        embedder=_FakeIdenticalEmbedder(),
        similarity_threshold=0.5,
    )
    assert len(list(claim)) == 1  # identical embeddings collapse to one


def _sorry_dataset_available() -> bool:
    try:
        from security_claim_sorry_bench import sorry_bench_claim
        from superred.core.types.llm import LLMConfig

        sorry_bench_claim(
            target_model_id="openai/gpt-4o-mini",
            judge_llm_config=LLMConfig(
                model="openai/gpt-4o-mini", api_base=_JB, api_key=_JK
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
def test_canonical_suite_is_exactly_79() -> None:
    # All defaults -> the canonical claim, built offline (static dedup, no embed).
    man: list[TaskRecord] = []
    stats: list[CombineStats] = []
    chatbot_suite_claim(
        target_model_id="openai/gpt-5.4-mini-2026-03-17",
        judge_api_base=_JB,
        judge_api_key=_JK,
        manifest_out=man,
        stats_out=stats,
    )
    by = Counter(r.source for r in man)
    assert by["sorrybench"] == 44
    assert by["harmbench"] == 17
    assert by["strongreject"] == 18
    assert len(man) == 79
    assert stats[0].dropped_static == 1


def test_strongreject_judge_model_defaults_to_faithful() -> None:
    # The StrongREJECT judge model defaults to the paper-faithful gpt-4o-mini.
    claim = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_harmbench=False,
        include_sorrybench=False,
        dedup=False,
        apply_static_dedup=False,
    )
    tasks = list(claim)
    assert tasks
    assert all(t._judge_config.model == "openai/gpt-4o-mini" for t in tasks)  # type: ignore[attr-defined]


def test_strongreject_judge_model_override_threads() -> None:
    # strongreject_judge_model reaches every StrongREJECT task's JudgeConfig.
    claim = chatbot_suite_claim(
        target_model_id="m",
        judge_api_base=_JB,
        judge_api_key=_JK,
        include_harmbench=False,
        include_sorrybench=False,
        strongreject_judge_model="bedrock/custom-judge",
        dedup=False,
        apply_static_dedup=False,
    )
    tasks = list(claim)
    assert tasks
    assert all(t._judge_config.model == "bedrock/custom-judge" for t in tasks)  # type: ignore[attr-defined]
