"""HarmBenchTask construction + lifecycle tests."""

from __future__ import annotations

import pytest

from secclaim_harmbench.task import HarmBenchTask

from .conftest import StubChatbotTarget, fake_row


def test_task_construction_exposes_metadata(stub_judge_yes) -> None:
    row = fake_row(
        behavior_id="bid",
        functional="contextual",
        semantic="cybercrime_intrusion",
        split="val",
        context="some context",  # required for contextual rows
        tags="context",
    )
    task = HarmBenchTask(row=row, judge=stub_judge_yes)
    assert task.behavior_id == "bid"
    assert task.functional_category == "contextual"
    assert task.semantic_category == "cybercrime_intrusion"
    assert task.split == "val"


def test_task_goal_is_rendered_test_case(stub_judge_yes) -> None:
    row = fake_row(behavior="MY BEHAVIOR")
    task = HarmBenchTask(row=row, judge=stub_judge_yes)
    assert task.goal.description == "MY BEHAVIOR"


def test_task_goal_for_contextual_concatenates(stub_judge_yes) -> None:
    row = fake_row(
        behavior="BEHAVIOR",
        functional="contextual",
        tags="context",
        context="CONTEXT",
    )
    task = HarmBenchTask(row=row, judge=stub_judge_yes)
    assert task.goal.description == "CONTEXT\n\n---\n\nBEHAVIOR"


def test_task_missing_required_column_raises(stub_judge_yes) -> None:
    bad = {"BehaviorID": "x"}  # missing Behavior, etc.
    with pytest.raises(ValueError, match="row missing required column"):
        HarmBenchTask(row=bad, judge=stub_judge_yes)


def test_contextual_row_with_empty_context_raises(stub_judge_yes) -> None:
    """A FunctionalCategory='contextual' row MUST carry a non-empty
    ContextString. Otherwise evaluate() would silently dispatch to the
    standard judge prompt, producing wrong verdicts."""
    bad = fake_row(functional="contextual", context="", tags="context")
    with pytest.raises(ValueError, match="empty ContextString"):
        HarmBenchTask(row=bad, judge=stub_judge_yes)


def test_contextual_row_with_whitespace_context_raises(stub_judge_yes) -> None:
    """Whitespace-only ContextString is treated as empty."""
    bad = fake_row(functional="contextual", context="   \n\t  ", tags="context")
    with pytest.raises(ValueError, match="empty ContextString"):
        HarmBenchTask(row=bad, judge=stub_judge_yes)


def test_standard_row_with_empty_context_is_fine(stub_judge_yes) -> None:
    """Standard rows are allowed to have empty ContextString."""
    row = fake_row(functional="standard", context="")
    task = HarmBenchTask(row=row, judge=stub_judge_yes)  # no exception
    assert task.functional_category == "standard"


@pytest.mark.asyncio
async def test_configure_target_sets_empty_system_prompt(
    stub_judge_yes, stub_target: StubChatbotTarget,
) -> None:
    """HarmBench's canonical setup: no system prompt added to conversation."""
    task = HarmBenchTask(row=fake_row(), judge=stub_judge_yes)
    await task.configure_target(stub_target)  # type: ignore[arg-type]
    assert ("system_prompt", "") in stub_target.config_calls
