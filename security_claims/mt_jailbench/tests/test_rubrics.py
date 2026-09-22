"""Rubric loader tests.

Content rule: never print rubric bodies. These tests compute booleans/lengths
from the loaded templates and assert on those, so a failure never surfaces the
rubric text in pytest's assertion introspection.
"""

from __future__ import annotations

from mt_jailbench_claim import rubrics


def test_three_default_rubrics_load_nonempty() -> None:
    lengths = {name: len(rubrics.rubric(name)) for name in ("XTEAMING", "COA", "FITD")}
    assert all(n > 0 for n in lengths.values())


def test_rubrics_have_behavior_and_response_placeholders() -> None:
    ok = {}
    for name in ("XTEAMING", "COA", "FITD"):
        tpl = rubrics.rubric(name)
        ok[name] = ("{behavior}" in tpl) and ("{response}" in tpl)
    assert all(ok.values()), f"missing placeholders in: {[k for k, v in ok.items() if not v]}"


def test_rubrics_format_with_synthetic_values() -> None:
    # .format must succeed with the two documented placeholders (any literal
    # braces in the rubric are escaped upstream). Use synthetic neutral values.
    for name in ("XTEAMING", "COA", "FITD"):
        filled = rubrics.rubric(name).format(behavior="BEHAVIOR_X", response="RESPONSE_Y")
        assert "BEHAVIOR_X" in filled and "RESPONSE_Y" in filled


def test_max_judge_retry_is_three() -> None:
    assert rubrics.max_judge_retry() == 3


def test_unknown_rubric_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        rubrics.rubric("NOT_A_JUDGE")
