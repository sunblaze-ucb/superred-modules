"""Unit tests for ``muzzle_optimizer.vessels``.

Covers the two surface-classification predicates (``is_content_surface`` /
``is_user_prompt_surface``) and the grafter-guided ranking in ``build_vessels``:
content surfaces first (ranked by grafter confidence, descending), then a single
user-prompt fallback, then a single system-prompt fallback.
"""

from __future__ import annotations

import os
import sys

import pytest

from muzzle_optimizer.vessels import (
    SYSTEM_PROMPT_NAME,
    Vessel,
    build_vessels,
    is_content_surface,
    is_user_prompt_surface,
)

# Ensure the shared harness (tests/conftest.py) is importable as the top-level
# ``conftest`` module regardless of pytest's import mode, then pull in helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import (  # noqa: E402
    SYSTEM_PROMPT_TAG,
    TOOL_A_TAG,
    TOOL_B_TAG,
    USER_TAG,
    make_controllable,
)

# --- is_content_surface -----------------------------------------------------


@pytest.mark.parametrize("name", ["tool:web", "read__inbox", "tool_call:x"])
def test_is_content_surface_true_for_tool_and_read_prefixes(name: str) -> None:
    """The ``read__`` / ``tool:`` / ``tool_call:`` prefixes mark content surfaces."""
    assert is_content_surface(make_controllable(name=name, tag=TOOL_A_TAG)) is True


@pytest.mark.parametrize(
    ("description", "value_type"),
    [("retrieved web page", "text"), ("", "html")],
)
def test_is_content_surface_true_for_content_metadata(description: str, value_type: str) -> None:
    """An opaque name is a content surface when its description/value_type hints at content."""
    ctrl = make_controllable(
        name="payload", tag=USER_TAG, value_type=value_type, description=description
    )
    assert is_content_surface(ctrl) is True


def test_is_content_surface_false_for_user_prompt() -> None:
    """The user prompt is not a content surface."""
    assert is_content_surface(make_controllable(name="user_prompt", tag=USER_TAG)) is False


@pytest.mark.parametrize("name", ["response", "assistant"])
def test_is_content_surface_false_for_bare_response_names(name: str) -> None:
    """A bare model-response name with no content metadata is not a content surface."""
    ctrl = make_controllable(name=name, tag=USER_TAG, description="", value_type="text")
    assert is_content_surface(ctrl) is False


def test_is_content_surface_extra_names_force_classifies_opaque_name() -> None:
    """An otherwise-uninformative name is forced to content only when in ``extra_names``."""
    ctrl = make_controllable(name="xyzzy", tag=USER_TAG)
    assert is_content_surface(ctrl) is False
    assert is_content_surface(ctrl, extra_names=frozenset({"xyzzy"})) is True


@pytest.mark.parametrize("category", ["content-injection", "environment-write"])
def test_is_content_surface_llm_role_marks_opaque_name_as_content(category: str) -> None:
    """An LLM ``content-injection`` / ``environment-write`` role marks an opaque surface."""
    ctrl = make_controllable(name="atlas_field_42", tag=USER_TAG)
    assert is_content_surface(ctrl) is False  # backstop alone: not content
    assert is_content_surface(ctrl, roles={"atlas_field_42": category}) is True


def test_is_content_surface_roles_none_preserves_backstop() -> None:
    """``roles=None`` (the default) leaves the name/keyword backstop untouched."""
    opaque = make_controllable(name="atlas_field_42", tag=USER_TAG)
    tool = make_controllable(name="tool:web", tag=TOOL_A_TAG)
    assert is_content_surface(opaque, roles=None) is False
    assert is_content_surface(tool, roles=None) is True
    # An unrelated role does not force a match either.
    assert is_content_surface(opaque, roles={"atlas_field_42": "user-prompt"}) is False


# --- is_user_prompt_surface -------------------------------------------------


@pytest.mark.parametrize("name", ["user_prompt", "user_message", "query"])
def test_is_user_prompt_surface_true_for_reserved_names(name: str) -> None:
    """Reserved user-prompt names classify as user-prompt surfaces."""
    assert is_user_prompt_surface(make_controllable(name=name, tag=USER_TAG)) is True


def test_is_user_prompt_surface_true_for_user_task_description() -> None:
    """An opaque name is a user-prompt surface when described as a user message/task."""
    ctrl = make_controllable(
        name="channel_0", tag=USER_TAG, description="the user provided this task message"
    )
    assert is_user_prompt_surface(ctrl) is True


def test_is_user_prompt_surface_false_for_content_name() -> None:
    """A content surface is not mistaken for the user prompt."""
    assert is_user_prompt_surface(make_controllable(name="tool:web", tag=TOOL_A_TAG)) is False


def test_is_user_prompt_surface_llm_role_marks_opaque_name() -> None:
    """An LLM ``user-prompt`` role marks an opaque surface the backstop would miss."""
    ctrl = make_controllable(name="channel_0", tag=USER_TAG)
    assert is_user_prompt_surface(ctrl) is False  # backstop alone: not a prompt
    assert is_user_prompt_surface(ctrl, roles={"channel_0": "user-prompt"}) is True


def test_build_vessels_threads_roles_into_content_classification() -> None:
    """An LLM-classified content surface becomes a content vessel via ``build_vessels``."""
    opaque = make_controllable(name="atlas_field_42", tag=USER_TAG)
    vessels = build_vessels([opaque], roles={"atlas_field_42": "environment-write"})
    assert [v.controllable.name for v in vessels] == ["atlas_field_42"]
    assert vessels[0].kind == "content"
    # Without the role it is neither content nor a prompt -> no vessel.
    assert build_vessels([opaque]) == []


def test_system_prompt_name_constant() -> None:
    """The last-resort system-prompt vessel keys off the canonical controllable name."""
    assert SYSTEM_PROMPT_NAME == "system_prompt"


# --- build_vessels ----------------------------------------------------------


def test_build_vessels_ranks_content_by_confidence_then_prompt_fallbacks() -> None:
    """Grafter-named content ranks by confidence DESC, then user-prompt, then system-prompt."""
    tool_a = make_controllable(name="tool:a", tag=TOOL_A_TAG)
    tool_b = make_controllable(name="tool:b", tag=TOOL_B_TAG)
    user = make_controllable(name="user_prompt", tag=USER_TAG)
    system = make_controllable(name="system_prompt", tag=SYSTEM_PROMPT_TAG)

    vessels = build_vessels(
        [tool_a, tool_b, user, system],
        grafter_candidates=[
            {"element": "the tool:a input field", "confidence": 0.3},
            {"element": "the tool:b output box", "confidence": 0.9},
        ],
    )

    assert all(isinstance(vessel, Vessel) for vessel in vessels)
    assert [vessel.controllable.name for vessel in vessels] == [
        "tool:b",
        "tool:a",
        "user_prompt",
        "system_prompt",
    ]
    assert [vessel.kind for vessel in vessels] == [
        "content",
        "content",
        "user_prompt",
        "system_prompt",
    ]
    assert [vessel.confidence for vessel in vessels] == [0.9, 0.3, 0.0, 0.0]
    # The grafter free-text propagates onto the winning content vessel.
    assert vessels[0].element == "the tool:b output box"


def test_build_vessels_appends_unranked_content_at_zero_confidence() -> None:
    """An in-scope content surface the grafter never named is still included at 0.0."""
    tool_a = make_controllable(name="tool:a", tag=TOOL_A_TAG)
    tool_b = make_controllable(name="tool:b", tag=TOOL_B_TAG)

    vessels = build_vessels(
        [tool_a, tool_b],
        grafter_candidates=[{"element": "the tool:a area", "confidence": 0.7}],
    )

    assert [vessel.controllable.name for vessel in vessels] == ["tool:a", "tool:b"]
    assert [vessel.kind for vessel in vessels] == ["content", "content"]
    assert [vessel.confidence for vessel in vessels] == [0.7, 0.0]


def test_build_vessels_drops_candidate_naming_no_in_scope_surface() -> None:
    """A candidate that names no in-scope content surface contributes nothing."""
    tool_a = make_controllable(name="tool:a", tag=TOOL_A_TAG)

    vessels = build_vessels(
        [tool_a],
        grafter_candidates=[
            {"element": "the tool:a area", "confidence": 0.5},
            {"element": "the totally_absent area", "confidence": 0.99},
        ],
    )

    assert len(vessels) == 1
    assert vessels[0].controllable.name == "tool:a"
    assert vessels[0].confidence == 0.5
    assert all(vessel.confidence != 0.99 for vessel in vessels)


def test_build_vessels_dedupes_controllable_keeping_highest_confidence() -> None:
    """Two candidates naming the same surface collapse to one vessel at the higher confidence."""
    tool_a = make_controllable(name="tool:a", tag=TOOL_A_TAG)

    vessels = build_vessels(
        [tool_a],
        grafter_candidates=[
            {"element": "first tool:a anchor", "confidence": 0.3},
            {"element": "second tool:a anchor", "confidence": 0.9},
        ],
    )

    assert len(vessels) == 1
    assert vessels[0].controllable is tool_a
    assert vessels[0].confidence == 0.9


def test_build_vessels_empty_controllables_returns_empty() -> None:
    """No controllables yields no vessels, even with grafter candidates present."""
    assert build_vessels([]) == []
    assert build_vessels([], grafter_candidates=[{"element": "x", "confidence": 0.5}]) == []


def test_build_vessels_single_fallback_of_each_prompt_kind() -> None:
    """At most one user-prompt and one system-prompt fallback are emitted."""
    user_a = make_controllable(name="user_prompt", tag=USER_TAG)
    user_b = make_controllable(name="query", tag=USER_TAG)
    system = make_controllable(name="system_prompt", tag=SYSTEM_PROMPT_TAG)

    vessels = build_vessels([user_a, user_b, system])

    kinds = [vessel.kind for vessel in vessels]
    assert kinds.count("user_prompt") == 1
    assert kinds.count("system_prompt") == 1
    # User-prompt fallback precedes the system-prompt fallback.
    assert kinds == ["user_prompt", "system_prompt"]


def test_build_vessels_opaque_names_and_bad_candidates_never_raise() -> None:
    """Garbage names, non-dict candidates, and unparseable confidences must not raise."""
    controllables = [
        make_controllable(name="###", tag=USER_TAG),
        make_controllable(name="λ-element", tag=USER_TAG),
        make_controllable(name="tool:weird", tag=TOOL_A_TAG, description="opaque"),
    ]
    candidates = [
        "not-a-dict",
        42,
        None,
        {"element": "tool:weird anchor", "confidence": "abc"},
    ]

    vessels = build_vessels(controllables, grafter_candidates=candidates)  # type: ignore[arg-type]

    assert isinstance(vessels, list)
    # The only content surface survives; the unparseable confidence falls back to 0.0.
    weird = next(vessel for vessel in vessels if vessel.controllable.name == "tool:weird")
    assert weird.kind == "content"
    assert weird.confidence == 0.0
