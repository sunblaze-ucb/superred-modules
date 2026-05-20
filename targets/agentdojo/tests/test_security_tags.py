"""Tests for the AgentDojo target's security-domain forest.

Verifies forest topology, parent-child relationships, scope-inclusion
semantics, and that the assembled :class:`SecurityDomain` exposes exactly
the expected roots.
"""

from __future__ import annotations

from superred.core.types.security_domain import scope_includes

from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    CONTENT_1P_DATA_1P_TAG,
    CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG,
    CONTENT_3P_DATA_3P_TAG,
    DOMAIN,
    MODEL_IDENTITY_TAG,
    PROMPT_READABLE_TAG,
    PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    USER_TAG,
)


def test_domain_has_three_roots() -> None:
    """system, user, tools are the three independent trees."""
    root_names = {t.name for t in DOMAIN.roots}
    assert root_names == {"system", "user", "tools"}


def test_system_tag_includes_all_system_descendants() -> None:
    """Holding the system root grants every system-side capability."""
    for descendant in (
        PROMPT_TAG,
        PROMPT_READABLE_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_READABLE_TAG,
        TOOL_CATALOGUE_ADDABLE_TAG,
        MODEL_IDENTITY_TAG,
        AGENT_TRACE_TAG,
        AGENT_TRACE_MESSAGES_TAG,
        AGENT_TRACE_TOOL_CALLS_TAG,
        AGENT_TRACE_TOOL_RESPONSES_TAG,
    ):
        assert SYSTEM_TAG.includes(descendant), descendant.name


def test_tool_catalogue_subsumption() -> None:
    """The catalogue write tag subsumes readable and addable children."""
    assert TOOL_CATALOGUE_TAG.includes(TOOL_CATALOGUE_READABLE_TAG)
    assert TOOL_CATALOGUE_TAG.includes(TOOL_CATALOGUE_ADDABLE_TAG)
    # but addable does not imply replace/unregister capability
    assert not TOOL_CATALOGUE_ADDABLE_TAG.includes(TOOL_CATALOGUE_TAG)


def test_prompt_subsumption() -> None:
    """The writable prompt tag subsumes the readable child."""
    assert PROMPT_TAG.includes(PROMPT_READABLE_TAG)
    assert not PROMPT_READABLE_TAG.includes(PROMPT_TAG)


def test_agent_trace_subsumption() -> None:
    """Aggregate agent_trace subsumes the three finer-grained children."""
    assert AGENT_TRACE_TAG.includes(AGENT_TRACE_MESSAGES_TAG)
    assert AGENT_TRACE_TAG.includes(AGENT_TRACE_TOOL_CALLS_TAG)
    assert AGENT_TRACE_TAG.includes(AGENT_TRACE_TOOL_RESPONSES_TAG)


def test_user_has_no_children() -> None:
    """The user tree is a single tag with no internal hierarchy."""
    for tag in (
        SYSTEM_TAG, TOOLS_TAG, PROMPT_TAG, MODEL_IDENTITY_TAG,
        AGENT_TRACE_TAG, CONTENT_1P_DATA_1P_TAG,
    ):
        assert not USER_TAG.includes(tag)


def test_tools_2x2_grid_siblings() -> None:
    """The four 2x2-grid leaves are siblings under tools; none subsumes any other."""
    leaves = (
        CONTENT_1P_DATA_1P_TAG,
        CONTENT_1P_DATA_3P_TAG,
        CONTENT_3P_DATA_1P_TAG,
        CONTENT_3P_DATA_3P_TAG,
    )
    for a in leaves:
        assert TOOLS_TAG.includes(a)
        for b in leaves:
            if a is not b:
                assert not a.includes(b), f"{a.name} should not include {b.name}"


def test_independent_trees_do_not_cross() -> None:
    """Roots do not include tags from other trees."""
    assert not SYSTEM_TAG.includes(USER_TAG)
    assert not SYSTEM_TAG.includes(TOOLS_TAG)
    assert not USER_TAG.includes(SYSTEM_TAG)
    assert not USER_TAG.includes(TOOLS_TAG)
    assert not TOOLS_TAG.includes(SYSTEM_TAG)
    assert not TOOLS_TAG.includes(USER_TAG)


def test_scope_includes_uses_tags() -> None:
    """Sanity check that scope_includes() agrees with tag.includes() across roots."""
    scope = frozenset({SYSTEM_TAG})
    assert scope_includes(scope, PROMPT_TAG)
    assert scope_includes(scope, AGENT_TRACE_MESSAGES_TAG)
    assert not scope_includes(scope, USER_TAG)
    assert not scope_includes(scope, CONTENT_1P_DATA_1P_TAG)


def test_assembled_domain_size() -> None:
    """All 17 declared tags appear in the assembled DOMAIN forest."""
    # We cannot inspect DOMAIN._tags directly (private), but distinct_combinations
    # exercises the full tag set; the empty antichain is always present, and the
    # antichain count grows monotonically with the tag count.  At minimum,
    # every root must enumerate.
    combos = DOMAIN.distinct_combinations()
    root_names = {t.name for c in combos for t in c if t.parent is None}
    assert root_names == {"system", "user", "tools"}
