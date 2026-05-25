"""Tests for the Controllable enumeration."""

from __future__ import annotations

from agentdojo_target.controllables import (
    CONTROLLABLES,
    READ_CTRLS,
    READ_QUADRANT_MAP,
    SYSTEM_PROMPT_CTRL,
    TOOL_CATALOG_CTRLS,
    USER_PROMPT_CTRL,
)
from agentdojo_target.security_tags import (
    CONTENT_1P_DATA_1P_TAG,
    CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG,
    CONTENT_3P_DATA_3P_TAG,
    PROMPT_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from agentdojo_target.tool_registry import READ_FUNCTION_NAMES


def test_system_prompt_ctrl_is_on_prompt_tag() -> None:
    assert SYSTEM_PROMPT_CTRL.security_domain is PROMPT_TAG


def test_user_prompt_ctrl_is_on_user_tag() -> None:
    assert USER_PROMPT_CTRL.security_domain is USER_TAG


def test_four_tool_catalog_ctrls() -> None:
    assert len(TOOL_CATALOG_CTRLS) == 4
    names = [c.name for c in TOOL_CATALOG_CTRLS]
    assert names == [
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_unregister",
        "tool_catalog_rewrite_doc",
    ]


def test_tool_catalog_register_is_weakest_capability() -> None:
    """register is on the narrower 'addable' tag; replace/unregister/rewrite
    are on the broader 'tool_catalogue' tag.  The broader tag implies the
    weaker one (subsumption tested separately)."""
    register, replace, unregister, rewrite = TOOL_CATALOG_CTRLS
    assert register.security_domain is TOOL_CATALOGUE_ADDABLE_TAG
    assert replace.security_domain is TOOL_CATALOGUE_TAG
    assert unregister.security_domain is TOOL_CATALOGUE_TAG
    assert rewrite.security_domain is TOOL_CATALOGUE_TAG


def test_one_read_ctrl_per_read_tool() -> None:
    """Every read tool has exactly one matching Controllable."""
    assert set(READ_CTRLS) == set(READ_FUNCTION_NAMES)
    assert len(READ_CTRLS) == 47


def test_every_read_ctrl_is_on_a_tools_leaf() -> None:
    """All per-read Controllables sit on one of the 2x2-grid leaves."""
    leaves = {
        CONTENT_1P_DATA_1P_TAG,
        CONTENT_1P_DATA_3P_TAG,
        CONTENT_3P_DATA_1P_TAG,
        CONTENT_3P_DATA_3P_TAG,
    }
    for ctrl in READ_CTRLS.values():
        assert ctrl.security_domain in leaves
        # Sanity: the tools root subsumes every leaf.
        assert TOOLS_TAG.includes(ctrl.security_domain)


def test_aggregate_controllables_no_duplicates() -> None:
    """The CONTROLLABLES list has unique names."""
    names = [c.name for c in CONTROLLABLES]
    assert len(names) == len(set(names))


def test_aggregate_controllables_size() -> None:
    """1 system_prompt + 1 user_prompt + 4 catalog + 47 reads = 53."""
    assert len(CONTROLLABLES) == 53


def test_read_quadrant_map_total_by_quadrant() -> None:
    """Sanity-check the per-quadrant distribution against the rationale
    laid out in the controllables module docstring.

    Most reads land in 3p/3p (most external content); a smaller cluster
    in 1p/3p (user PII held by 3p providers); one 3p/1p (banking
    read_file returns vendor-authored files in the user's filesystem);
    one 1p/1p (workspace.get_current_day system clock).
    """
    counts = {
        CONTENT_1P_DATA_1P_TAG: 0,
        CONTENT_1P_DATA_3P_TAG: 0,
        CONTENT_3P_DATA_1P_TAG: 0,
        CONTENT_3P_DATA_3P_TAG: 0,
    }
    for tag in READ_QUADRANT_MAP.values():
        counts[tag] += 1
    assert counts[CONTENT_1P_DATA_1P_TAG] >= 1
    assert counts[CONTENT_3P_DATA_1P_TAG] >= 1
    assert counts[CONTENT_1P_DATA_3P_TAG] >= 10
    assert counts[CONTENT_3P_DATA_3P_TAG] >= 25
    assert sum(counts.values()) == len(READ_CTRLS)
