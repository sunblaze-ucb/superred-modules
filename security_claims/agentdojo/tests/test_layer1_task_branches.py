"""Inner-branch tests for :class:`AgentDojoPairedTask`.

The Layer-1 task body has three branchy helpers that mutation testing
showed were under-covered:

- ``_load_trace``: filters trace by suite prefix and strips it.
- ``_call_utility`` / ``_call_security``: try ``*_from_traces`` first,
  fall back to the post-env variant, and ultimately swallow
  ``NotImplementedError`` (returning False with a warning).

These tests construct synthetic upstream tasks (subclassing
:class:`BaseUserTask` / :class:`BaseInjectionTask`) and a stub target
to exercise each branch in isolation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
# Pre-import to flush AgentDojo's registration chain in the right order
# (see env.py for the explanation).
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.default_suites.v1.banking.task_suite import BankingEnvironment
from agentdojo.functions_runtime import FunctionCall

from agentdojo_target import AgentDojoTarget
from agentdojo_target.seed_loader import load_composite_seed
from security_claim_agentdojo.layer1_bridge import (
    get_injection_task,
    get_user_task,
)
from security_claim_agentdojo.layer1_task import AgentDojoPairedTask


# ---------------------------------------------------------------------------
# Stub target that lets tests inject query results
# ---------------------------------------------------------------------------


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


def _empty_env_blob() -> dict:
    """A composite-env snapshot with every sub-env at its default seed."""
    return json.loads(load_composite_seed().model_dump_json())


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_constructor_rejects_unknown_suite() -> None:
    """The constructor must validate suite against SUITE_NAMES."""
    user_task = get_user_task("banking", "user_task_0")
    injection_task = get_injection_task("banking", "injection_task_0")
    with pytest.raises(ValueError, match="Unknown suite"):
        AgentDojoPairedTask(
            suite="nonsense",
            user_task_id="user_task_0",
            injection_task_id="injection_task_0",
            user_task=user_task,
            injection_task=injection_task,
            category="exfil_via_memo",
        )


def test_goal_description_is_injection_task_goal() -> None:
    """The Layer-1 Goal carries the injection task's GOAL verbatim."""
    user_task = get_user_task("banking", "user_task_0")
    injection_task = get_injection_task("banking", "injection_task_0")
    pair = AgentDojoPairedTask(
        suite="banking",
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        user_task=user_task,
        injection_task=injection_task,
        category="exfil_via_memo",
    )
    assert pair.goal.description == injection_task.GOAL


# ---------------------------------------------------------------------------
# _load_trace: prefix filter + strip
# ---------------------------------------------------------------------------


def _make_pair(suite: str = "banking") -> AgentDojoPairedTask:
    return AgentDojoPairedTask(
        suite=suite,
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        user_task=get_user_task(suite, "user_task_0"),
        injection_task=get_injection_task(suite, "injection_task_0"),
        category="exfil_via_memo",
    )


def test_load_trace_strips_suite_prefix(stub_target) -> None:
    pair = _make_pair("banking")
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "banking__get_balance", "args": {}, "id": None},
        {"function": "banking__send_money",
         "args": {"recipient": "X", "amount": 1.0, "subject": "s", "date": "d"},
         "id": "abc"},
    ])
    trace = pair._load_trace(stub_target)
    assert [fc.function for fc in trace] == ["get_balance", "send_money"]
    assert trace[1].id == "abc"


def test_load_trace_filters_other_suites(stub_target) -> None:
    """Functions whose prefix isn't this suite are filtered out."""
    pair = _make_pair("banking")
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "workspace__send_email", "args": {}, "id": None},
        {"function": "banking__get_balance", "args": {}, "id": None},
        {"function": "slack__get_channels", "args": {}, "id": None},
    ])
    trace = pair._load_trace(stub_target)
    assert [fc.function for fc in trace] == ["get_balance"]


def test_load_trace_empty(stub_target) -> None:
    pair = _make_pair("banking")
    stub_target.q["function_call_trace"] = json.dumps([])
    assert pair._load_trace(stub_target) == []


def test_load_trace_args_dict_coercion(stub_target) -> None:
    """``args`` is always a dict on the output, even when input is missing."""
    pair = _make_pair("banking")
    stub_target.q["function_call_trace"] = json.dumps([
        {"function": "banking__get_balance"},  # no args key at all
    ])
    trace = pair._load_trace(stub_target)
    assert trace[0].args == {}


# ---------------------------------------------------------------------------
# _call_utility / _call_security: trace-first fallback
# ---------------------------------------------------------------------------


class _StubUserTask(BaseUserTask[BankingEnvironment]):
    """User task with configurable utility / utility_from_traces returns."""

    PROMPT = "stub"
    GROUND_TRUTH_OUTPUT = ""
    DIFFICULTY = None  # unused

    def __init__(
        self, *, traces_result: bool | None, util_result: bool | type[Exception],
    ) -> None:
        self._traces_result = traces_result
        self._util_result = util_result

    def ground_truth(self, pre_environment):  # type: ignore[override]
        return []

    def utility_from_traces(  # type: ignore[override]
        self, model_output, pre, post, traces,
    ) -> bool | None:
        return self._traces_result

    def utility(  # type: ignore[override]
        self, model_output, pre_env, post_env, strict: bool = True,
    ) -> bool:
        r = self._util_result
        if isinstance(r, type) and issubclass(r, Exception):
            raise r()
        return r  # type: ignore[return-value]


class _StubInjectionTask(BaseInjectionTask[BankingEnvironment]):
    GOAL = "stub goal"
    GROUND_TRUTH_OUTPUT = ""

    def __init__(
        self,
        *,
        traces_result: bool | None,
        sec_result: bool | type[Exception],
    ) -> None:
        self._traces_result = traces_result
        self._sec_result = sec_result

    def ground_truth(self, pre_environment):  # type: ignore[override]
        return []

    def security_from_traces(  # type: ignore[override]
        self, model_output, pre, post, traces,
    ) -> bool | None:
        return self._traces_result

    def security(  # type: ignore[override]
        self, model_output, pre_env, post_env,
    ) -> bool:
        r = self._sec_result
        if isinstance(r, type) and issubclass(r, Exception):
            raise r()
        return r  # type: ignore[return-value]


def _pair_with_stubs(user_task, injection_task) -> AgentDojoPairedTask:
    return AgentDojoPairedTask(
        suite="banking",
        user_task_id="user_task_0",
        injection_task_id="injection_task_0",
        user_task=user_task,
        injection_task=injection_task,
        category="exfil_via_memo",
    )


def _envs():
    """Two distinct BankingEnvironment instances for pre/post arguments."""
    pre = load_composite_seed().banking
    post = load_composite_seed().banking
    return pre, post


def test_call_utility_prefers_from_traces_when_true() -> None:
    """``utility_from_traces=True`` short-circuits even if ``utility=False``."""
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=True, util_result=False),
        _StubInjectionTask(traces_result=False, sec_result=False),
    )
    assert pair._call_utility("", pre, post, traces=[]) is True


def test_call_utility_falls_back_to_utility_when_traces_returns_none() -> None:
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=None, util_result=True),
        _StubInjectionTask(traces_result=False, sec_result=False),
    )
    assert pair._call_utility("", pre, post, traces=[]) is True


def test_call_utility_returns_false_when_utility_raises_notimplemented() -> None:
    """When both *_from_traces returns None AND utility raises
    NotImplementedError, predicate returns False with warning."""
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=None, util_result=NotImplementedError),
        _StubInjectionTask(traces_result=False, sec_result=False),
    )
    assert pair._call_utility("", pre, post, traces=[]) is False


def test_call_security_prefers_from_traces_when_true() -> None:
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=False, util_result=False),
        _StubInjectionTask(traces_result=True, sec_result=False),
    )
    assert pair._call_security("", pre, post, traces=[]) is True


def test_call_security_falls_back_to_security_when_traces_returns_none() -> None:
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=False, util_result=False),
        _StubInjectionTask(traces_result=None, sec_result=True),
    )
    assert pair._call_security("", pre, post, traces=[]) is True


def test_call_security_returns_false_on_notimplemented() -> None:
    pre, post = _envs()
    pair = _pair_with_stubs(
        _StubUserTask(traces_result=False, util_result=False),
        _StubInjectionTask(traces_result=None, sec_result=NotImplementedError),
    )
    assert pair._call_security("", pre, post, traces=[]) is False


def test_call_security_swallows_unexpected_exception_in_traces() -> None:
    """If ``security_from_traces`` raises an unrelated exception, we fall
    back to ``security`` rather than propagating."""
    pre, post = _envs()

    class _SinkingTraces(_StubInjectionTask):
        def security_from_traces(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("oops")

    pair = _pair_with_stubs(
        _StubUserTask(traces_result=False, util_result=False),
        _SinkingTraces(traces_result=False, sec_result=True),
    )
    assert pair._call_security("", pre, post, traces=[]) is True
