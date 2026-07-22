"""Tests for the Layer-1 factories.

Verifies the top, per-suite, and per-category factories return the
right Task counts and that filter kwargs narrow correctly.
"""

from __future__ import annotations

import pytest

from agentdojo_claim import (
    agentdojo_layer1_banking_claim,
    agentdojo_layer1_category_claim,
    agentdojo_layer1_claim,
    agentdojo_layer1_slack_claim,
    agentdojo_layer1_suite_claim,
    agentdojo_layer1_travel_claim,
    agentdojo_layer1_workspace_claim,
)
from agentdojo_claim.layer1_task import AgentDojoPairedTask


def _tasks(claim) -> list[AgentDojoPairedTask]:
    return list(claim)


def test_top_factory_default_is_27_pairs() -> None:
    tasks = _tasks(agentdojo_layer1_claim())
    assert len(tasks) == 27
    assert all(isinstance(t, AgentDojoPairedTask) for t in tasks)


def test_per_suite_factories_match_canonical_distribution() -> None:
    assert len(_tasks(agentdojo_layer1_banking_claim())) == 9
    assert len(_tasks(agentdojo_layer1_workspace_claim())) == 6
    assert len(_tasks(agentdojo_layer1_slack_claim())) == 5
    assert len(_tasks(agentdojo_layer1_travel_claim())) == 7


def test_per_suite_factories_only_emit_matching_suite() -> None:
    for suite_factory, suite in (
        (agentdojo_layer1_banking_claim, "banking"),
        (agentdojo_layer1_workspace_claim, "workspace"),
        (agentdojo_layer1_slack_claim, "slack"),
        (agentdojo_layer1_travel_claim, "travel"),
    ):
        for task in suite_factory():
            assert task.suite == suite


def test_per_category_factory_filters_correctly() -> None:
    """Per-category claim contains only tasks of that category."""
    claim = agentdojo_layer1_category_claim("exfil_via_memo")
    tasks = _tasks(claim)
    # exfil_via_memo covers banking IT 0,1,2,3,8 = 5 tasks
    assert len(tasks) == 5
    for t in tasks:
        assert t.category == "exfil_via_memo"


def test_pairs_kwarg_takes_precedence_and_supports_extension() -> None:
    """pairs= argument bypasses suite/category filters."""
    custom = [
        ("banking", "user_task_0", "injection_task_7"),
        ("workspace", "user_task_14", "injection_task_0"),
    ]
    tasks = _tasks(agentdojo_layer1_claim(pairs=custom))
    assert len(tasks) == 2
    assert {t.injection_task_id for t in tasks} == {
        "injection_task_7", "injection_task_0",
    }


def test_unknown_suite_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown suite"):
        agentdojo_layer1_claim(suites=["nonsense"])


def test_unknown_category_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown category"):
        agentdojo_layer1_claim(categories=["nonsense"])


def test_claim_is_re_iterable() -> None:
    """SecurityClaim must support multiple iterations (per framework convention)."""
    claim = agentdojo_layer1_banking_claim()
    first = list(claim)
    second = list(claim)
    assert len(first) == len(second) == 9


def test_every_task_has_a_goal_and_user_task_id() -> None:
    for task in agentdojo_layer1_claim():
        assert task.goal.description  # non-empty injection GOAL
        assert task.user_task_id.startswith("user_task_")
        assert task.injection_task_id.startswith("injection_task_")
