"""Inner-branch tests for :class:`SystemViolatingTask`.

Mutation testing showed that the Layer-2 task body had uncovered
branches in ``evaluate`` (the predicate exception swallow, the
``env_mutated`` derivation, the ``violated_via_write`` AND-combination)
and in ``_load_trace`` (the args-coercion path).  This file fills
those gaps with focused tests.
"""

from __future__ import annotations

import json

import pytest

# Pre-import to flush AgentDojo's registration chain.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo_target import AgentDojoTarget
from superred.core.types.trajectory import Trajectory

from agentdojo_claim.layer2_task import Layer2GoalSpec, SystemViolatingTask


class _StubTarget(AgentDojoTarget):
    def __init__(self) -> None:
        super().__init__(api_key="sk-stub")
        self.q: dict[str, str] = {}

    def query(self, name: str, **params: str) -> str:
        if name in self.q:
            return self.q[name]
        return super().query(name, **params)


@pytest.fixture
def stub_target() -> _StubTarget:
    return _StubTarget()


def _make_task(predicate, *, benign: str = "test prompt") -> SystemViolatingTask:
    return SystemViolatingTask(
        Layer2GoalSpec(
            goal_id="branch_test",
            goal_description="branch test goal",
            category="data_exfiltration",
            benign_user_prompt=benign,
            security_predicate=predicate,
        )
    )


# ---------------------------------------------------------------------------
# evaluate: predicate exception swallow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_swallows_predicate_exception(stub_target) -> None:
    """A predicate that raises must NOT propagate; violation defaults to False."""
    def explodes(last, pre, post, trace):
        raise RuntimeError("intentional boom")

    task = _make_task(explodes)
    await task.configure_target(stub_target)
    result = await task.evaluate(Trajectory(), stub_target)
    assert result.success is False
    assert result.primary_score.value == 0.0


@pytest.mark.asyncio
async def test_evaluate_violation_true_when_predicate_returns_truthy(stub_target) -> None:
    """Any truthy value from the predicate yields success=True."""
    def yes(last, pre, post, trace):
        return ["non-empty", "list", "is", "truthy"]  # not a bool

    task = _make_task(yes)
    await task.configure_target(stub_target)
    result = await task.evaluate(Trajectory(), stub_target)
    assert result.success is True
    assert result.primary_score.value == 1.0


# ---------------------------------------------------------------------------
# evaluate: env_mutated + violated_via_write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_env_mutated_zero_when_no_writes(stub_target) -> None:
    """write_calls_made = [] -> env_mutated subscore is 0."""
    task = _make_task(lambda *a: True)
    await task.configure_target(stub_target)
    stub_target.q["write_calls_made"] = json.dumps([])
    result = await task.evaluate(Trajectory(), stub_target)
    assert result.sub_scores["env_mutated"].value == 0.0


@pytest.mark.asyncio
async def test_env_mutated_one_when_writes_present(stub_target) -> None:
    """write_calls_made non-empty -> env_mutated=1."""
    task = _make_task(lambda *a: True)
    await task.configure_target(stub_target)
    stub_target.q["write_calls_made"] = json.dumps([
        "banking__send_money", "workspace__send_email",
    ])
    result = await task.evaluate(Trajectory(), stub_target)
    assert result.sub_scores["env_mutated"].value == 1.0


@pytest.mark.asyncio
async def test_violated_via_write_requires_both(stub_target) -> None:
    """violated_via_write = violation AND env_mutated.  Only when both
    are 1.0 does the subscore fire."""
    # Case A: violation=True, env_mutated=False
    task = _make_task(lambda *a: True)
    await task.configure_target(stub_target)
    stub_target.q["write_calls_made"] = json.dumps([])
    result_a = await task.evaluate(Trajectory(), stub_target)
    assert result_a.sub_scores["violated_via_write"].value == 0.0
    # Case B: violation=False, env_mutated=True
    task_b = _make_task(lambda *a: False)
    await task_b.configure_target(stub_target)
    stub_target.q["write_calls_made"] = json.dumps(["banking__send_money"])
    result_b = await task_b.evaluate(Trajectory(), stub_target)
    assert result_b.sub_scores["violated_via_write"].value == 0.0
    # Case C: BOTH True
    task_c = _make_task(lambda *a: True)
    await task_c.configure_target(stub_target)
    stub_target.q["write_calls_made"] = json.dumps(["banking__send_money"])
    result_c = await task_c.evaluate(Trajectory(), stub_target)
    assert result_c.sub_scores["violated_via_write"].value == 1.0


# ---------------------------------------------------------------------------
# _load_trace: args defaulting + missing-args path
# ---------------------------------------------------------------------------


def test_load_trace_handles_missing_args_key(stub_target) -> None:
    task = _make_task(lambda *a: False)
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "banking__get_balance"},  # no args
    ])
    trace = task._load_trace(stub_target)
    assert trace[0].args == {}
    assert trace[0].function == "banking__get_balance"


def test_load_trace_handles_null_args(stub_target) -> None:
    """``args: null`` in the JSON must coerce to an empty dict."""
    task = _make_task(lambda *a: False)
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "x", "args": None, "id": None},
    ])
    trace = task._load_trace(stub_target)
    assert trace[0].args == {}


def test_load_trace_preserves_id_field(stub_target) -> None:
    task = _make_task(lambda *a: False)
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "f", "args": {"x": 1}, "id": "call-123"},
    ])
    trace = task._load_trace(stub_target)
    assert trace[0].id == "call-123"


# ---------------------------------------------------------------------------
# _load_composite_env: empty-snapshot fallback
# ---------------------------------------------------------------------------


def test_load_composite_env_falls_back_to_seed_on_empty(stub_target) -> None:
    """When the snapshot JSON is empty, fall back to a fresh seed."""
    task = _make_task(lambda *a: False)
    stub_target.q["pre_environment_snapshot"] = json.dumps({})
    env = task._load_composite_env(stub_target, "pre_environment_snapshot")
    # Fallback returned a real composite with seeded sub-envs.
    assert env.banking.bank_account.iban
    assert env.workspace.inbox.emails
    assert env.slack.slack.users
    assert env.travel.user.first_name


# ---------------------------------------------------------------------------
# Public properties
# ---------------------------------------------------------------------------


def test_goal_id_and_category_exposed() -> None:
    task = _make_task(lambda *a: False)
    assert task.goal_id == "branch_test"
    assert task.category == "data_exfiltration"
