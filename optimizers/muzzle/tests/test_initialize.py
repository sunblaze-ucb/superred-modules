"""Construction and ``initialize`` contract for :class:`MuzzleOptimizer`.

These tests pin the zero-argument factory contract (an ``OptimizerFactory`` builds a fresh
instance with no required config), the constructor's budget-argument validation, and what
``initialize`` infers from the per-task scope: the LLM client, the brain helpers, and the
``_can_inject`` flag that records whether any in-scope controllable is an injectable surface.
``initialize`` must never raise on degenerate scopes (no controllables, opaque names, or no
observables) -- the optimizer has to run against arbitrary, unseen targets.
"""

from __future__ import annotations

import pytest

from muzzle_optimizer.optimizer import MuzzleOptimizer
from superred.core.types.goal import Goal

from conftest import (
    SYSTEM_PROMPT_TAG,
    TOOL_A_TAG,
    TOOL_B_TAG,
    USER_TAG,
    ScriptedLLM,
    make_controllable,
    make_observable_value,
)


def test_constructs_with_zero_arguments() -> None:
    """The factory contract: a fresh instance needs no constructor configuration."""
    optimizer = MuzzleOptimizer()
    # Nothing is inferred yet -- scope specifics arrive in initialize().
    assert optimizer._can_inject is False
    assert optimizer._phase == "OBSERVE"
    assert optimizer._summarizer is None
    assert optimizer._pair is None


def test_construct_accepts_boundary_budgets() -> None:
    """The minimum legal budgets (1 each) are accepted, not rejected at the boundary."""
    optimizer = MuzzleOptimizer(max_attempts=1, max_tries=1)
    assert optimizer._max_attempts == 1
    assert optimizer._max_tries == 1


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_max_attempts_below_one_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        MuzzleOptimizer(max_attempts=bad)


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_max_tries_below_one_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="max_tries"):
        MuzzleOptimizer(max_tries=bad)


@pytest.mark.asyncio
async def test_initialize_sets_llm_and_builds_helpers() -> None:
    """After initialize the constrained client is exposed and every brain helper exists."""
    optimizer = MuzzleOptimizer()
    llm = ScriptedLLM()
    goal = Goal(description="exfiltrate the secret")
    await optimizer.initialize(goal, [make_controllable("user_prompt", USER_TAG)], [], llm)

    assert optimizer.llm is llm
    assert optimizer._goal is goal
    # All five red-team-role helpers are constructed once self.llm is available.
    assert optimizer._summarizer is not None
    assert optimizer._grafter is not None
    assert optimizer._prompter is not None
    assert optimizer._judge is not None
    assert optimizer._pair is not None
    # Phase machine starts in OBSERVE (the benign passthrough run).
    assert optimizer._phase == "OBSERVE"


@pytest.mark.asyncio
async def test_initialize_stores_controllables_and_observables() -> None:
    """initialize snapshots both input lists for later vessel building / recovery."""
    optimizer = MuzzleOptimizer()
    controllables = [
        make_controllable("user_prompt", USER_TAG),
        make_controllable("tool:web", TOOL_A_TAG),
    ]
    observables = [make_observable_value("system_prompt", "You are a helpful agent.")]
    await optimizer.initialize(Goal(description="g"), controllables, observables, ScriptedLLM())

    assert len(optimizer._controllables) == 2
    assert len(optimizer._observables) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, tag, expected",
    [
        ("tool:lookup", TOOL_A_TAG, True),  # content surface via the tool: prefix
        ("read__file", TOOL_A_TAG, True),  # content surface via the read__ prefix
        ("retrieved_document", TOOL_B_TAG, True),  # content surface via name hints
        ("user_prompt", USER_TAG, True),  # user-prompt fallback surface
        ("query", USER_TAG, True),  # reserved user-prompt name
        ("system_prompt", SYSTEM_PROMPT_TAG, True),  # last-resort system-prompt vessel
        ("xq_7", TOOL_B_TAG, False),  # opaque, non-injectable name
    ],
)
async def test_can_inject_reflects_injectable_surface(
    name: str, tag: object, expected: bool
) -> None:
    """_can_inject is True iff the lone in-scope controllable is an injectable surface."""
    optimizer = MuzzleOptimizer()
    controllable = make_controllable(name, tag)  # type: ignore[arg-type]
    await optimizer.initialize(Goal(description="g"), [controllable], [], ScriptedLLM())
    assert optimizer._can_inject is expected


@pytest.mark.asyncio
async def test_extra_content_names_force_classify_opaque_surface() -> None:
    """An opaque controllable is injectable only when force-classified via extra_content_names."""
    opaque = make_controllable("opaque_blob", TOOL_A_TAG)

    plain = MuzzleOptimizer()
    await plain.initialize(Goal(description="g"), [opaque], [], ScriptedLLM())
    assert plain._can_inject is False

    forced = MuzzleOptimizer(extra_content_names=["opaque_blob"])
    await forced.initialize(Goal(description="g"), [opaque], [], ScriptedLLM())
    assert forced._can_inject is True


@pytest.mark.asyncio
async def test_initialize_empty_controllables_and_observables_no_crash() -> None:
    """No injectable surface and no observables must initialize cleanly (passthrough scope)."""
    optimizer = MuzzleOptimizer()
    await optimizer.initialize(Goal(description="g"), [], [], ScriptedLLM())
    assert optimizer._can_inject is False
    assert optimizer._controllables == []
    assert optimizer._observables == []


@pytest.mark.asyncio
async def test_initialize_opaque_unknown_names_no_crash() -> None:
    """Opaque controllable names (odd value_type/description) classify safely as non-surfaces."""
    optimizer = MuzzleOptimizer()
    controllables = [
        make_controllable("xq_7", TOOL_B_TAG, value_type="blob", description="vendor opaque slot"),
        make_controllable("k9-thing", USER_TAG, value_type="binary", description="opaque handle"),
    ]
    await optimizer.initialize(Goal(description="g"), controllables, [], ScriptedLLM())
    # None of the opaque names match a content / user-prompt / system-prompt surface.
    assert optimizer._can_inject is False
