"""Loader tests for the vendored SafeClawBench Semantic Core dataset."""

from __future__ import annotations

from safeclawbench_claim.loader import (
    AGENT_SYSTEM_PROMPT,
    ATTACK_FAMILIES,
    CASE_KEYS,
    SafeClawCase,
    family_label,
    load_cases,
)


def test_load_cases_count_and_type():
    cases = load_cases()
    assert len(cases) == 600
    assert all(isinstance(c, SafeClawCase) for c in cases)


def test_every_case_has_required_fields():
    for case in load_cases():
        for key in CASE_KEYS:
            assert getattr(case, key), f"{case.task_id} missing {key}"


def test_from_row_rejects_missing_keys():
    import pytest

    with pytest.raises(ValueError, match="missing keys"):
        SafeClawCase.from_row({"task_id": "X_1"})


def test_families_are_canonical():
    fams = {c.attack_type for c in load_cases()}
    assert fams == set(ATTACK_FAMILIES)


def test_family_label_maps_known_and_passes_unknown():
    assert family_label("DPI") == "Direct Prompt Injection"
    assert family_label("ZZZ") == "ZZZ"


def test_agent_system_prompt_lists_six_tool_categories():
    for cat in ("File system", "Web", "Email", "Code execution", "Database", "Memory"):
        assert cat in AGENT_SYSTEM_PROMPT
