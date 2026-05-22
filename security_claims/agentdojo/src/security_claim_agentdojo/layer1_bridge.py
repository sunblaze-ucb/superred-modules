"""Bridge helpers between superred and AgentDojo upstream tasks.

Exposes:

- :func:`get_user_task`         -> :class:`BaseUserTask` instance by id.
- :func:`get_injection_task`    -> :class:`BaseInjectionTask` instance by id.
- :func:`get_environment_type`  -> per-suite ``TaskEnvironment`` subclass.
- :func:`compute_init_env_overlay` -> serialised post-``init_environment``
  state for the ``seed_yaml_override__{suite}`` config slot.

Wraps AgentDojo's import-time decorator-driven task registration via
``agentdojo.task_suite.load_suites.get_suite``; pinned to v1.2.2 (the
latest released benchmark version).  Version is set in
:data:`_BENCHMARK_VERSION` and must stay in sync with the matching
constant in :mod:`agentdojo_target.seed_loader`.
"""

from __future__ import annotations

from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
# Pre-import to flush the AgentDojo registration chain in the right order
# (matches the workaround documented in agentdojo_target.env).
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.functions_runtime import TaskEnvironment
from agentdojo.task_suite.load_suites import get_suite

_BENCHMARK_VERSION: str = "v1.2.2"
"""Latest released AgentDojo benchmark version.  See
``agentdojo.task_suite.load_suites._V1_2_2_SUITES`` for the per-suite
version mapping (banking + workspace at (1,2,2), travel + slack at
(1,2,0))."""


def _suite(suite_name: str):
    """Return the v1.2.2 TaskSuite for *suite_name*.

    Local helper around :func:`agentdojo.task_suite.load_suites.get_suite`
    so callers don't have to repeat the version string.
    """
    return get_suite(_BENCHMARK_VERSION, suite_name)


def get_user_task(suite: str, user_task_id: str) -> BaseUserTask:
    """Look up a v1.2.2 user task instance by id (e.g. ``user_task_3``)."""
    return _suite(suite).get_user_task_by_id(user_task_id)


def get_injection_task(suite: str, injection_task_id: str) -> BaseInjectionTask:
    """Look up a v1.2.2 injection task instance by id."""
    s = _suite(suite)
    if injection_task_id not in s.injection_tasks:
        raise KeyError(
            f"Suite {suite!r} has no injection task {injection_task_id!r}"
        )
    return s.injection_tasks[injection_task_id]


def get_environment_type(suite: str) -> type[TaskEnvironment]:
    """Return the per-suite ``TaskEnvironment`` subclass."""
    return _suite(suite).environment_type


def compute_init_env_overlay(suite: str, user_task: BaseUserTask) -> str:
    """Run ``user_task.init_environment`` and serialise the result.

    Returns the post-mutation sub-env as a JSON string suitable for the
    target's ``seed_yaml_override__{suite}`` config slot.  If the user
    task makes no env mutation, returns ``""`` so the target's overlay
    merger short-circuits.

    The baseline is AgentDojo's default-injected environment for the
    suite (i.e. the upstream ``load_and_inject_default_environment({})``
    output), so the overlay encodes only what the user-task expects on
    top of that baseline.
    """
    s = _suite(suite)
    baseline = s.load_and_inject_default_environment({})
    mutated = user_task.init_environment(baseline.model_copy(deep=True))
    # Pydantic equality is value-based; identity is preserved when the
    # default identity init_environment is in effect.
    if mutated == baseline:
        return ""
    return mutated.model_dump_json()


__all__ = [
    "get_user_task",
    "get_injection_task",
    "get_environment_type",
    "compute_init_env_overlay",
]
