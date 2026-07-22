"""Tests for ``sorry_bench_claim`` factory and dataset loader.

These tests use the real ``question.jsonl`` resolved by the
``question_jsonl_path`` fixture (``conftest.py``). No synthetic fixture
is shipped — tests fail loudly if the dataset is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superred.core.types.llm import LLMConfig

from sorry_bench_claim.categories import CATEGORY_NAMES
from sorry_bench_claim.factory import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
    SUBSET_A_QUESTION_IDS,
    SUBSET_B_QUESTION_IDS,
    _filter_rows,
    _load_dataset,
    sorry_bench_claim,
)
from sorry_bench_claim.judge import RefusalRegexJudge
from sorry_bench_claim.task import SorryBenchTask


# ---------------------------------------------------------------------------
# _load_dataset
# ---------------------------------------------------------------------------


class TestLoadDatasetFromPath:
    def test_loads_440_rows(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        assert len(rows) == 440

    def test_each_row_has_required_fields(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        for row in rows[:5]:
            assert {"question_id", "category", "turns"} <= set(row.keys())

    def test_question_ids_are_contiguous_1_to_440(
        self, question_jsonl_path: str
    ) -> None:
        rows = _load_dataset(question_jsonl_path)
        qids = sorted(int(r["question_id"]) for r in rows)
        assert qids == list(range(1, 441))

    def test_category_field_is_a_string(self, question_jsonl_path: str) -> None:
        # Plan correction: 'category' is a numeric string, not int.
        rows = _load_dataset(question_jsonl_path)
        assert isinstance(rows[0]["category"], str)

    def test_categories_cover_1_through_44(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        cats = {int(r["category"]) for r in rows}
        assert cats == set(range(1, 45))

    def test_turns_is_length_one(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        assert all(len(r["turns"]) == 1 for r in rows)


# ---------------------------------------------------------------------------
# _filter_rows
# ---------------------------------------------------------------------------


class TestFilterRows:
    def test_no_filter_keeps_all_rows(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset=None,
        )
        assert len(out) == 440

    def test_categories_filter_keeps_only_listed(
        self, question_jsonl_path: str
    ) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=[1, 8, 42],
            prompts_per_category=None,
            question_ids=None,
            subset=None,
        )
        assert len(out) == 30  # 3 cats × 10 prompts each
        assert all(int(r["category"]) in {1, 8, 42} for r in out)

    def test_prompts_per_category_truncates(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=2,
            question_ids=None,
            subset=None,
        )
        assert len(out) == 88  # 44 cats × 2

    def test_categories_and_prompts_per_category_compose(
        self, question_jsonl_path: str
    ) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=[8, 9, 20],
            prompts_per_category=3,
            question_ids=None,
            subset=None,
        )
        assert len(out) == 9

    def test_question_ids_is_exclusive_filter(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=[1, 2, 3],  # ignored when question_ids provided
            prompts_per_category=1,  # also ignored
            question_ids=[1, 50, 100, 440],
            subset="a",  # also ignored
        )
        assert len(out) == 4
        ids = {int(r["question_id"]) for r in out}
        assert ids == {1, 50, 100, 440}

    def test_negative_prompts_per_category_raises(
        self, question_jsonl_path: str
    ) -> None:
        rows = _load_dataset(question_jsonl_path)
        with pytest.raises(ValueError, match="prompts_per_category must be >= 0"):
            _filter_rows(
                rows,
                categories=None,
                prompts_per_category=-1,
                question_ids=None,
                subset=None,
            )


# ---------------------------------------------------------------------------
# sorry_bench_claim factory
# ---------------------------------------------------------------------------


def _stub_judge_config() -> LLMConfig:
    """An LLMConfig that does not get used because we pass `judge=` explicitly."""
    return LLMConfig(
        model="gpt-4-1106-preview",
        api_base="http://example.invalid",
        api_key="sk-test",
    )


class TestFactory:
    def test_full_claim_has_440_tasks(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
        )
        tasks = list(claim)
        assert len(tasks) == 440

    def test_each_task_is_a_sorry_bench_task(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            prompts_per_category=1,
        )
        tasks = list(claim)
        assert all(isinstance(t, SorryBenchTask) for t in tasks)

    def test_subset_by_categories(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            categories=[1, 5, 12],
        )
        tasks = list(claim)
        assert len(tasks) == 30
        assert {t.category_id for t in tasks} == {1, 5, 12}

    def test_subset_by_prompts_per_category(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            prompts_per_category=2,
        )
        tasks = list(claim)
        assert len(tasks) == 88

    def test_subset_by_question_ids(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            question_ids=[1, 50, 100],
        )
        tasks = list(claim)
        assert {t.question_id for t in tasks} == {1, 50, 100}

    def test_each_task_uses_correct_canonical_category_name(
        self, question_jsonl_path: str
    ) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            prompts_per_category=1,
        )
        for task in claim:
            assert task.category_name == CATEGORY_NAMES[task.category_id - 1]

    def test_system_prompt_is_target_specific(self, question_jsonl_path: str) -> None:
        # GPT-4o gets the helpful-assistant prompt (no period); via the Task.
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-2024-05-13",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            question_ids=[1],
        )
        task = next(iter(claim))
        # Internal access for the test — the system_prompt is private but stable.
        assert task._system_prompt == "You are a helpful assistant"  # noqa: SLF001

    def test_factory_is_re_iterable(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            prompts_per_category=1,
        )
        first = list(claim)
        second = list(claim)
        assert len(first) == len(second) == 44

    def test_requires_judge_or_judge_llm_config(self, question_jsonl_path: str) -> None:
        with pytest.raises(ValueError, match="`judge` or `judge_llm_config`"):
            sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=question_jsonl_path,
            )

    def test_accepts_judge_llm_config_alone(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge_llm_config=_stub_judge_config(),
            prompts_per_category=1,
        )
        # Tasks were created (judge is built lazily from the config).
        assert len(list(claim)) == 44

    def test_empty_filter_result_raises(self, question_jsonl_path: str) -> None:
        with pytest.raises(ValueError, match="No tasks produced"):
            sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=question_jsonl_path,
                judge=RefusalRegexJudge(),
                question_ids=[99999],  # no such id
            )


class TestSubsetFilter:
    """``subset="a" | "b"`` returns one of two disjoint, stratified halves.

    Each subset is 220 prompts (5/category × 44 categories), and the two
    halves together reconstruct the full 440-prompt benchmark. The split
    is by ``question_id`` parity.
    """

    # -- the SUBSET_*_QUESTION_IDS constants themselves -----------------

    def test_subset_constants_sizes(self) -> None:
        assert len(SUBSET_A_QUESTION_IDS) == 220
        assert len(SUBSET_B_QUESTION_IDS) == 220

    def test_subset_constants_are_disjoint(self) -> None:
        assert SUBSET_A_QUESTION_IDS.isdisjoint(SUBSET_B_QUESTION_IDS)

    def test_subset_constants_union_covers_full_benchmark(self) -> None:
        assert SUBSET_A_QUESTION_IDS | SUBSET_B_QUESTION_IDS == frozenset(range(1, 441))

    def test_subset_a_is_odd_q_ids(self) -> None:
        assert SUBSET_A_QUESTION_IDS == frozenset(range(1, 441, 2))

    def test_subset_b_is_even_q_ids(self) -> None:
        assert SUBSET_B_QUESTION_IDS == frozenset(range(2, 441, 2))

    # -- _filter_rows behavior on the real dataset ----------------------

    def test_subset_a_returns_220_rows(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="a",
        )
        assert len(out) == 220

    def test_subset_b_returns_220_rows(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="b",
        )
        assert len(out) == 220

    def test_subset_a_and_b_are_disjoint(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out_a = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="a",
        )
        out_b = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="b",
        )
        ids_a = {int(r["question_id"]) for r in out_a}
        ids_b = {int(r["question_id"]) for r in out_b}
        assert ids_a.isdisjoint(ids_b)

    def test_subset_a_and_b_union_covers_full(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out_a = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="a",
        )
        out_b = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=None,
            question_ids=None,
            subset="b",
        )
        ids = {int(r["question_id"]) for r in (*out_a, *out_b)}
        assert ids == set(range(1, 441))

    def test_each_category_is_evenly_split(self, question_jsonl_path: str) -> None:
        # Stratification invariant: each subset has exactly 5/10 prompts
        # for every one of the 44 categories.
        rows = _load_dataset(question_jsonl_path)
        for label in ("a", "b"):
            out = _filter_rows(
                rows,
                categories=None,
                prompts_per_category=None,
                question_ids=None,
                subset=label,  # type: ignore[arg-type]
            )
            per_cat: dict[int, int] = {}
            for r in out:
                per_cat[int(r["category"])] = per_cat.get(int(r["category"]), 0) + 1
            assert per_cat == {c: 5 for c in range(1, 45)}, (
                f"subset={label!r} not evenly stratified: {per_cat}"
            )

    def test_subset_composes_with_categories(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=[1, 2, 3],
            prompts_per_category=None,
            question_ids=None,
            subset="a",
        )
        assert len(out) == 15  # 3 cats × 5 prompts per cat (subset a)
        assert all(int(r["category"]) in {1, 2, 3} for r in out)
        assert all(int(r["question_id"]) % 2 == 1 for r in out)

    def test_subset_composes_with_prompts_per_category(
        self, question_jsonl_path: str
    ) -> None:
        rows = _load_dataset(question_jsonl_path)
        out = _filter_rows(
            rows,
            categories=None,
            prompts_per_category=2,
            question_ids=None,
            subset="b",
        )
        assert len(out) == 88  # 44 cats × 2 (truncated from 5 in subset b)
        assert all(int(r["question_id"]) % 2 == 0 for r in out)

    def test_invalid_subset_value_raises(self, question_jsonl_path: str) -> None:
        rows = _load_dataset(question_jsonl_path)
        with pytest.raises(ValueError, match="subset must be 'a', 'b', or None"):
            _filter_rows(
                rows,
                categories=None,
                prompts_per_category=None,
                question_ids=None,
                subset="c",  # type: ignore[arg-type]
            )


class TestSubsetFactory:
    """``sorry_bench_claim(subset=...)`` produces a balanced subset claim."""

    def test_subset_a_claim_has_220_tasks(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            subset="a",
        )
        tasks = list(claim)
        assert len(tasks) == 220

    def test_subset_b_claim_has_220_tasks(self, question_jsonl_path: str) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            subset="b",
        )
        tasks = list(claim)
        assert len(tasks) == 220

    def test_subsets_have_disjoint_question_ids(self, question_jsonl_path: str) -> None:
        claim_a = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            subset="a",
        )
        claim_b = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            subset="b",
        )
        ids_a = {t.question_id for t in claim_a}
        ids_b = {t.question_id for t in claim_b}
        assert ids_a.isdisjoint(ids_b)
        assert ids_a | ids_b == set(range(1, 441))

    def test_subset_a_stratified_across_categories(
        self, question_jsonl_path: str
    ) -> None:
        claim = sorry_bench_claim(
            target_model_id="gpt-4o-mini",
            dataset_path=question_jsonl_path,
            judge=RefusalRegexJudge(),
            subset="a",
        )
        per_cat: dict[int, int] = {}
        for task in claim:
            per_cat[task.category_id] = per_cat.get(task.category_id, 0) + 1
        assert per_cat == {c: 5 for c in range(1, 45)}


class TestPinnedConstants:
    def test_dataset_repo_id_is_canonical(self) -> None:
        assert DATASET_REPO_ID == "sorry-bench/sorry-bench-202503"

    def test_dataset_filename_is_question_jsonl(self) -> None:
        assert DATASET_FILENAME == "question.jsonl"

    def test_dataset_revision_is_pinned_to_known_commit(self) -> None:
        # Pinned to the verified commit on the gated dataset.
        assert DATASET_REVISION == "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9"


class TestRowSchemaInvariant:
    """Invariant: per-category aggregation requires (q_id - 1) // 10 + 1 == category."""

    def test_question_id_to_category_relation_holds(
        self, question_jsonl_path: str
    ) -> None:
        # The aggregation formula `(q_id - 1) // 10 + 1 == category` must hold
        # for all 440 rows; if it fails, per-category aggregation breaks.
        rows = _load_dataset(question_jsonl_path)
        mismatches = [
            (int(r["question_id"]), int(r["category"]))
            for r in rows
            if (int(r["question_id"]) - 1) // 10 + 1 != int(r["category"])
        ]
        assert mismatches == []

    def test_first_row_matches_appendix_e_example(
        self, question_jsonl_path: str
    ) -> None:
        # Paper Appendix E page 21, Category 1 example.
        with open(question_jsonl_path, encoding="utf-8") as fp:
            first = json.loads(fp.readline())
        assert first["question_id"] == 1
        assert first["category"] == "1"
        assert "disabled colleague" in first["turns"][0]


# ---------------------------------------------------------------------------
# HF lazy-load path (mocked so we don't hit network)
# ---------------------------------------------------------------------------


class TestHfLazyLoad:
    def test_uses_hf_hub_download_when_no_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        question_jsonl_path: str,
    ) -> None:
        # When ``path is None``, the loader calls ``hf_hub_download`` and passes
        # the configured revision; the returned path is read like a local file.
        captured: dict[str, object] = {}

        def fake_download(
            *,
            repo_id: str,
            filename: str,
            repo_type: str,
            revision: str,
        ) -> str:
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            captured["repo_type"] = repo_type
            captured["revision"] = revision
            return question_jsonl_path

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", fake_download, raising=True
        )
        rows = _load_dataset(None)
        assert len(rows) == 440
        assert captured["repo_id"] == DATASET_REPO_ID
        assert captured["filename"] == DATASET_FILENAME
        assert captured["repo_type"] == "dataset"
        assert captured["revision"] == DATASET_REVISION

    def test_raises_runtimeerror_when_hf_download_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(**_kw: object) -> str:
            raise OSError("simulated network failure")

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", fake_download, raising=True
        )
        with pytest.raises(RuntimeError, match="Failed to download"):
            _load_dataset(None)


# ---------------------------------------------------------------------------
# Defensive row handling
# ---------------------------------------------------------------------------


class TestMalformedRows:
    """The factory must skip malformed rows and out-of-range categories."""

    def test_skips_row_missing_required_field(
        self,
        tmp_path: Path,
        question_jsonl_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Build a tiny JSONL: one malformed row (no ``turns``) + one good row.
        with open(question_jsonl_path, encoding="utf-8") as fp:
            good = json.loads(fp.readline())
        bad = {"question_id": 999, "category": "1"}  # missing 'turns'
        path = tmp_path / "mixed.jsonl"
        path.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")
        with caplog.at_level("WARNING"):
            claim = sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=str(path),
                judge=RefusalRegexJudge(),
            )
        assert len(list(claim)) == 1
        assert any("malformed row" in r.message.lower() for r in caplog.records)

    def test_skips_row_with_out_of_range_category(
        self,
        tmp_path: Path,
        question_jsonl_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with open(question_jsonl_path, encoding="utf-8") as fp:
            good = json.loads(fp.readline())
        bad = {"question_id": 9, "category": "99", "turns": ["x"]}  # 99 > 44
        path = tmp_path / "out_of_range.jsonl"
        path.write_text(json.dumps(bad) + "\n" + json.dumps(good) + "\n")
        with caplog.at_level("WARNING"):
            claim = sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=str(path),
                judge=RefusalRegexJudge(),
            )
        assert len(list(claim)) == 1
        assert any("out-of-range category" in r.message for r in caplog.records)

    def test_skips_blank_lines_in_jsonl(
        self, tmp_path: Path, question_jsonl_path: str
    ) -> None:
        with open(question_jsonl_path, encoding="utf-8") as fp:
            good = json.loads(fp.readline())
        path = tmp_path / "with_blanks.jsonl"
        # Two blank lines surrounding a valid row.
        path.write_text("\n" + json.dumps(good) + "\n\n")
        rows = _load_dataset(str(path))
        assert len(rows) == 1


class TestNonBaseDatasetWarning:
    """L-8: warn loudly if the user accidentally points at a mutated JSONL."""

    def test_warns_when_prompt_style_is_not_base(
        self,
        tmp_path: Path,
        question_jsonl_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Build a JSONL where the first row simulates a mutated prompt.
        with open(question_jsonl_path, encoding="utf-8") as fp:
            good = json.loads(fp.readline())
        mutated = dict(good)
        mutated["prompt_style"] = "caesar"
        path = tmp_path / "fake_mutated.jsonl"
        path.write_text(json.dumps(mutated) + "\n")

        with caplog.at_level("WARNING"):
            sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=str(path),
                judge=RefusalRegexJudge(),
            )
        assert any(
            "non-base prompt_style" in r.message and "caesar=1" in r.message
            for r in caplog.records
        )

    def test_no_warning_when_all_rows_are_base(
        self,
        question_jsonl_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The canonical question.jsonl has prompt_style=='base' on every row.
        with caplog.at_level("WARNING"):
            sorry_bench_claim(
                target_model_id="gpt-4o-mini",
                dataset_path=question_jsonl_path,
                judge=RefusalRegexJudge(),
                prompts_per_category=1,
            )
        assert not any("non-base prompt_style" in r.message for r in caplog.records)
