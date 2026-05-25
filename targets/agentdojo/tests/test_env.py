"""Tests for the composite environment.

Construction works from upstream-loaded sub-envs, deep-copy isolates
each sub-attribute, and the sibling ``Calendar``/``Inbox`` instances
between workspace and travel do not alias.
"""

from __future__ import annotations

from agentdojo.task_suite.load_suites import get_suite

from agentdojo_target import BENCHMARK_VERSION
from agentdojo_target.env import CompositeEnvironment


def _load_default_subenvs() -> tuple[object, object, object, object]:
    """Load each suite's default (no-injection) environment via AgentDojo upstream."""
    banking = get_suite(BENCHMARK_VERSION, "banking").load_and_inject_default_environment({})
    workspace = get_suite(BENCHMARK_VERSION, "workspace").load_and_inject_default_environment({})
    slack = get_suite(BENCHMARK_VERSION, "slack").load_and_inject_default_environment({})
    travel = get_suite(BENCHMARK_VERSION, "travel").load_and_inject_default_environment({})
    return banking, workspace, slack, travel


def test_composite_construction() -> None:
    """All four sub-envs slot into a single composite root."""
    banking, workspace, slack, travel = _load_default_subenvs()
    env = CompositeEnvironment(
        banking=banking, workspace=workspace, slack=slack, travel=travel,
    )
    assert env.banking is banking
    assert env.workspace is workspace
    assert env.slack is slack
    assert env.travel is travel


def test_deep_copy_isolates_subenvs() -> None:
    """``model_copy(deep=True)`` produces an independent snapshot."""
    banking, workspace, slack, travel = _load_default_subenvs()
    env = CompositeEnvironment(
        banking=banking, workspace=workspace, slack=slack, travel=travel,
    )
    snapshot = env.model_copy(deep=True)

    # Mutate the original; the snapshot must not change.
    env.banking.bank_account.balance = 0.0
    assert snapshot.banking.bank_account.balance != 0.0


def test_workspace_and_travel_calendars_are_distinct() -> None:
    """Workspace and travel both hold a Calendar; the instances do not alias."""
    banking, workspace, slack, travel = _load_default_subenvs()
    env = CompositeEnvironment(
        banking=banking, workspace=workspace, slack=slack, travel=travel,
    )
    assert env.workspace.calendar is not env.travel.calendar
    # Mutating one must not affect the other.
    env.workspace.calendar.events.clear()
    assert env.travel.calendar.events  # still populated from the travel seed


def test_workspace_and_travel_inboxes_are_distinct() -> None:
    """Workspace and travel both hold an Inbox; the instances do not alias."""
    banking, workspace, slack, travel = _load_default_subenvs()
    env = CompositeEnvironment(
        banking=banking, workspace=workspace, slack=slack, travel=travel,
    )
    assert env.workspace.inbox is not env.travel.inbox


def test_composite_is_taskenvironment_subclass() -> None:
    """CompositeEnvironment subclasses agentdojo's TaskEnvironment so it
    plugs into FunctionsRuntime.run_function(env, ...) unchanged."""
    from agentdojo.functions_runtime import TaskEnvironment

    assert issubclass(CompositeEnvironment, TaskEnvironment)
