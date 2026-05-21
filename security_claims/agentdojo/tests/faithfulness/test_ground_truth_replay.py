"""Ground-truth replay faithfulness suite.

For every AgentDojo v1 user task and injection task, replay the
upstream task's ``ground_truth`` function-call list through our
:class:`agentdojo_target.tool_registry.ALL_FUNCTIONS` against a
composite environment, then assert that upstream's matching utility /
security predicate verdicts agree with the canonical
expectation:

- A user-task GT replay must yield ``utility = True``.
- An injection-task GT replay must yield ``security = True``.

This validates, with **no LLM cost**, that:

- Our composite env loads each suite's seed correctly.
- Our suite-prefix rewriting + ``Depends`` rebinding routes every tool
  to the right sub-env.
- Our ``WrappedFunctionsRuntime``-equivalent dispatch (here a plain
  :class:`FunctionsRuntime` driving the same prefixed catalogue) keeps
  mutations consistent with upstream.
- The :func:`agentdojo_target.env.sync_initial_fields` round-trip is
  faithful (the predicate sees the agent's mutations).
- Upstream's ``utility`` / ``security`` predicates fire correctly
  against the post-env our runtime produces.

A handful of upstream tasks have known divergences between their GT
function-call sequence and their utility predicate (e.g. banking
UT5's GT sends 5.00 to "Spotify" but utility expects 50.00 to
SE3550... ; see the catalog notes in
``$CLAUDE_JOB_DIR/notes/agentdojo_suites_*.md``).  Those are marked
``xfail`` here so they document the gap rather than mask it.

Tests are parametrised over every (suite, user_task) and (suite,
injection_task) so the count auto-adjusts when upstream changes; the
total in v1 is 97 + 27 = 124 cases.
"""

from __future__ import annotations

import copy

import pytest

# Pre-import to flush AgentDojo's registration chain.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from agentdojo_target.env import CompositeEnvironment, sync_initial_fields
from agentdojo_target.seed_loader import load_composite_seed
from agentdojo_target.tool_registry import ALL_FUNCTIONS, SUITE_NAMES

# ---------------------------------------------------------------------------
# Known divergences: upstream GT vs upstream predicate.  Each entry is a
# (suite, task_id, reason) triple flagged xfail so we keep them visible.
# ---------------------------------------------------------------------------

_KNOWN_USER_TASK_DIVERGENCES: dict[tuple[str, str], str] = {
    ("banking", "user_task_5"): (
        "GT writes amount=5 to 'Spotify' but utility expects 50.00 to "
        "SE3550...; upstream catalogue typo."
    ),
    ("banking", "user_task_9"): (
        "utility requires amount==1100 AND pre==post (contradictory)."
    ),
    ("banking", "user_task_10"): (
        "utility requires pre==post but GT may not mutate (one read tool)."
    ),
    ("workspace", "user_task_7"): (
        "Upstream v1 bug: utility expects values_changed == {start_time} "
        "only, but v1 reschedule_calendar_event also mutates end_time to "
        "preserve duration when new_end_time is omitted."
    ),
    ("workspace", "user_task_17"): (
        "Negated boolean in utility check is inverted from prompt intent "
        "(upstream bug)."
    ),
    ("workspace", "user_task_20"): (
        "GT uses 0-length event (12:00-12:00) but prompt says 1 hour."
    ),
}

_KNOWN_INJECTION_TASK_DIVERGENCES: dict[tuple[str, str], str] = {
    # banking IT2 has _FOOD with 5 entries but security only accepts pizza/sushi.
    ("banking", "injection_task_2"): (
        "GT subject may use one of 5 _FOOD entries but security only "
        "accepts pizza/sushi."
    ),
}


def _collect_user_tasks() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for suite_name in SUITE_NAMES:
        suite = get_suite("v1", suite_name)
        for task_id in suite.user_tasks:
            out.append((suite_name, task_id))
    return out


def _collect_injection_tasks() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for suite_name in SUITE_NAMES:
        suite = get_suite("v1", suite_name)
        for task_id in suite.injection_tasks:
            out.append((suite_name, task_id))
    return out


USER_TASK_PARAMS = _collect_user_tasks()
INJECTION_TASK_PARAMS = _collect_injection_tasks()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_runtime() -> FunctionsRuntime:
    """A fresh FunctionsRuntime with our suite-prefixed catalogue."""
    return FunctionsRuntime(ALL_FUNCTIONS)


def _build_composite_with_suite_env(
    suite_name: str, sub_env,
) -> CompositeEnvironment:
    """Build a composite env whose *suite_name* sub-env is *sub_env* and the
    other three suites are at their default seeds."""
    composite = load_composite_seed()
    setattr(composite, suite_name, sub_env)
    return composite


def _replay_gt(
    suite_name: str, gt_calls, env: CompositeEnvironment,
) -> list[FunctionCall]:
    """Replay a list of bare-name FunctionCalls through our prefixed runtime.

    Returns the trace with BARE names (suite prefix stripped) so it can
    be passed to upstream's ``*_from_traces`` methods unchanged.
    """
    runtime = _build_runtime()
    trace: list[FunctionCall] = []
    for fc in gt_calls:
        prefixed = f"{suite_name}__{fc.function}"
        # NOTE: ``args`` may be FunctionCall (nested) per upstream; the
        # runtime handles those.  We always record the FLAT FunctionCall
        # in the trace.
        _, _err = runtime.run_function(env, prefixed, fc.args)
        trace.append(
            FunctionCall(function=fc.function, args=dict(fc.args), id=fc.id)
        )
    # Faithfulness mandate: sync initial_* before any reads of the env
    # via round-trip helpers; in-memory predicate access doesn't need
    # this but it costs nothing and keeps consumers consistent.
    sync_initial_fields(env)
    return trace


def _call_utility(user_task, model_output: str, pre, post, trace) -> bool | None:
    """Mirror Layer-1's utility dispatch."""
    try:
        traced = user_task.utility_from_traces(model_output, pre, post, trace)
    except Exception:
        traced = None
    if traced is not None:
        return bool(traced)
    try:
        return bool(user_task.utility(model_output, pre, post))
    except NotImplementedError:
        return None


def _call_security(injection_task, model_output: str, pre, post, trace) -> bool | None:
    try:
        traced = injection_task.security_from_traces(model_output, pre, post, trace)
    except Exception:
        traced = None
    if traced is not None:
        return bool(traced)
    try:
        return bool(injection_task.security(model_output, pre, post))
    except NotImplementedError:
        return None


# ---------------------------------------------------------------------------
# User-task GT replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suite_name,task_id", USER_TASK_PARAMS)
def test_user_task_ground_truth_yields_utility_true(suite_name: str, task_id: str) -> None:
    """For every v1 user task: GT replay -> utility predicate True."""
    if (suite_name, task_id) in _KNOWN_USER_TASK_DIVERGENCES:
        pytest.xfail(
            f"{suite_name}/{task_id}: {_KNOWN_USER_TASK_DIVERGENCES[(suite_name, task_id)]}"
        )
    suite = get_suite("v1", suite_name)
    user_task = suite.get_user_task_by_id(task_id)

    pre_sub = suite.load_and_inject_default_environment({})
    pre_sub = user_task.init_environment(pre_sub.model_copy(deep=True))
    pre_snapshot = copy.deepcopy(pre_sub)

    env = _build_composite_with_suite_env(suite_name, pre_sub)
    gt_calls = user_task.ground_truth(pre_snapshot)
    trace = _replay_gt(suite_name, gt_calls, env)

    post_sub = getattr(env, suite_name)
    model_output = user_task.GROUND_TRUTH_OUTPUT or ""
    verdict = _call_utility(user_task, model_output, pre_snapshot, post_sub, trace)
    if verdict is None:
        pytest.skip(
            f"{suite_name}/{task_id} has no usable utility / utility_from_traces"
        )
    assert verdict is True, (
        f"GT replay of {suite_name}/{task_id} did not satisfy upstream "
        "utility predicate."
    )


# ---------------------------------------------------------------------------
# Injection-task GT replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suite_name,task_id", INJECTION_TASK_PARAMS)
def test_injection_task_ground_truth_yields_security_true(suite_name: str, task_id: str) -> None:
    """For every v1 injection task: GT replay -> security predicate True."""
    if (suite_name, task_id) in _KNOWN_INJECTION_TASK_DIVERGENCES:
        pytest.xfail(
            f"{suite_name}/{task_id}: {_KNOWN_INJECTION_TASK_DIVERGENCES[(suite_name, task_id)]}"
        )
    suite = get_suite("v1", suite_name)
    injection_task = suite.injection_tasks[task_id]

    pre_sub = suite.load_and_inject_default_environment({})
    pre_snapshot = copy.deepcopy(pre_sub)

    env = _build_composite_with_suite_env(suite_name, pre_sub)
    gt_calls = injection_task.ground_truth(pre_snapshot)
    trace = _replay_gt(suite_name, gt_calls, env)

    post_sub = getattr(env, suite_name)
    model_output = injection_task.GROUND_TRUTH_OUTPUT or ""
    verdict = _call_security(injection_task, model_output, pre_snapshot, post_sub, trace)
    if verdict is None:
        pytest.skip(
            f"{suite_name}/{task_id} has no usable security / security_from_traces"
        )
    assert verdict is True, (
        f"GT replay of {suite_name}/{task_id} did not satisfy upstream "
        "security predicate."
    )
