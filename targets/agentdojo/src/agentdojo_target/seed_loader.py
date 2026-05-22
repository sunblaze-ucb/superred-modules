"""Seed loader: build the canonical no-attacker :class:`CompositeEnvironment`.

Calls :meth:`TaskSuite.load_and_inject_default_environment({})` on each of
the four v1 suites so the upstream ``injection_vectors.yaml`` defaults
are substituted into the YAML before parsing.  The result is the
legitimate baseline state that the wrapped runtime will surface as
``ControllablePostCallEvent.answer``.

Also exposes a per-suite YAML-overlay merger that Tasks use via the
``seed_yaml_override__{suite}`` config slots to replay AgentDojo's
per-task ``init_environment`` mutations.
"""

from __future__ import annotations

import json
from typing import Any

import yaml
from agentdojo.task_suite.load_suites import get_suite

from agentdojo_target.env import CompositeEnvironment

_SUITE_NAMES: tuple[str, str, str, str] = ("banking", "workspace", "slack", "travel")
_BENCHMARK_VERSION: str = "v1.2.2"
"""Latest released AgentDojo benchmark version.  Inherits all fixes
through the v1.2 series: workspace IT3 / IT6-IT13 added or revised,
workspace UT0/UT17/UT18 fixed (UT17 De Morgan time-check repaired),
workspace UT16 (read-flag flip), banking UT6 (iPhone-subject lambda
broadened), slack UT2 and UT11 (route through ``*_from_traces``),
travel IT2.  Per-suite version mapping lives in
``agentdojo.task_suite.load_suites._V1_2_2_SUITES``."""


def load_composite_seed() -> CompositeEnvironment:
    """Build a fresh :class:`CompositeEnvironment` from the v1 default environments.

    Each sub-env is constructed by calling AgentDojo's
    ``load_and_inject_default_environment({})``, which interpolates the
    ``injection_vectors.yaml`` defaults into the YAML text before parsing.
    The legitimate (un-attacked) baseline returned here is what the
    wrapped runtime serves as the legitimate read value for the
    per-tool Controllables.

    Returns:
        A freshly-constructed :class:`CompositeEnvironment`.  Callers
        should ``model_copy(deep=True)`` if they need an isolated
        snapshot.
    """
    banking = get_suite(_BENCHMARK_VERSION, "banking").load_and_inject_default_environment({})
    workspace = get_suite(_BENCHMARK_VERSION, "workspace").load_and_inject_default_environment({})
    slack = get_suite(_BENCHMARK_VERSION, "slack").load_and_inject_default_environment({})
    travel = get_suite(_BENCHMARK_VERSION, "travel").load_and_inject_default_environment({})
    return CompositeEnvironment(
        banking=banking,
        workspace=workspace,
        slack=slack,
        travel=travel,
    )


def merge_yaml_overlay(
    env: CompositeEnvironment,
    suite_name: str,
    overlay_text: str,
) -> CompositeEnvironment:
    """Apply a JSON or YAML overlay to a sub-env, returning a new composite.

    The overlay is parsed as JSON if it starts with ``{`` or ``[`` (after
    stripping whitespace); otherwise YAML.  The parsed dict is
    recursively merged into the named sub-env's existing model dump,
    then re-validated.  The original *env* is not modified.

    Empty or whitespace-only overlays are no-ops.

    Args:
        env: The composite to overlay onto.
        suite_name: One of ``banking``, ``workspace``, ``slack``,
            ``travel``.
        overlay_text: JSON or YAML text encoding a partial sub-env dict.

    Returns:
        A new :class:`CompositeEnvironment` with the overlay applied to
        the named sub-env.  Other sub-envs are deep-copied unchanged.

    Raises:
        ValueError: If *suite_name* is not a recognised suite, the
            overlay fails to parse, or the merged result fails pydantic
            validation against the sub-env class.
    """
    if suite_name not in _SUITE_NAMES:
        raise ValueError(
            f"Unknown suite {suite_name!r}; expected one of {_SUITE_NAMES}"
        )
    stripped = overlay_text.strip()
    if not stripped:
        return env.model_copy(deep=True)

    parsed: Any
    if stripped[0] in "{[":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Overlay for suite {suite_name!r} could not be parsed as JSON: {exc}"
            ) from exc
    else:
        try:
            parsed = yaml.safe_load(stripped)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Overlay for suite {suite_name!r} could not be parsed as YAML: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Overlay for suite {suite_name!r} must encode a dict; got {type(parsed).__name__}"
        )

    sub = getattr(env, suite_name)
    base_dump = sub.model_dump()
    merged = _deep_merge(base_dump, parsed)
    new_sub = type(sub).model_validate(merged)

    return env.model_copy(update={suite_name: new_sub}, deep=True)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict where *overlay* recursively replaces values in *base*.

    Dict values are merged recursively.  Every other value type is
    overwritten by *overlay*.  Lists, in particular, are NOT merged
    element-wise; the overlay list replaces the base list entirely.
    This matches the principle of least surprise for partial-overlay
    overrides like ``{"bank_account": {"balance": 0}}``.
    """
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["load_composite_seed", "merge_yaml_overlay"]
