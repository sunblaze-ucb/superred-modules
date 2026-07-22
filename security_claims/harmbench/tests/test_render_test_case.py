"""Tests for render_test_case (canonical separator handling)."""

from __future__ import annotations

import pytest

from harmbench_claim.task import render_test_case


def test_standard_returns_raw_behavior() -> None:
    row = {
        "Behavior": "Make a thing",
        "ContextString": "",
        "BehaviorID": "stub", "FunctionalCategory": "standard",
        "SemanticCategory": "harmful", "Tags": "",
    }
    assert render_test_case(row) == "Make a thing"


def test_contextual_concatenates_with_canonical_separator() -> None:
    row = {
        "Behavior": "B",
        "ContextString": "C",
        "BehaviorID": "stub", "FunctionalCategory": "contextual",
        "SemanticCategory": "harmful", "Tags": "context",
    }
    assert render_test_case(row) == "C\n\n---\n\n B".replace(" B", "B")


def test_contextual_separator_is_exact() -> None:
    """The separator must be exactly two newlines, three dashes, two newlines.
    This matches the canonical form across HarmBench's DirectRequest, PAIR,
    TAP, and GPTFuzz baselines."""
    row = {
        "Behavior": "BEHAV",
        "ContextString": "CTX",
        "BehaviorID": "stub", "FunctionalCategory": "contextual",
        "SemanticCategory": "harmful", "Tags": "context",
    }
    assert render_test_case(row) == "CTX\n\n---\n\nBEHAV"


def test_empty_context_treated_as_standard() -> None:
    row = {
        "Behavior": "Just the behavior",
        "ContextString": "",  # explicit empty (standard rows have this)
        "BehaviorID": "stub", "FunctionalCategory": "standard",
        "SemanticCategory": "harmful", "Tags": "",
    }
    assert render_test_case(row) == "Just the behavior"


def test_missing_context_field_treated_as_standard() -> None:
    """If somehow a row lacks ContextString entirely, fall back to raw."""
    row = {
        "Behavior": "Just the behavior",
        "BehaviorID": "stub", "FunctionalCategory": "standard",
        "SemanticCategory": "harmful", "Tags": "",
    }
    assert render_test_case(row) == "Just the behavior"


def test_multiline_context_preserved() -> None:
    """ContextString often contains newlines (e.g. the DDoS code example).
    Multi-line contexts must be passed through verbatim."""
    multiline = "line1\nline2\nline3"
    row = {
        "Behavior": "Obfuscate this",
        "ContextString": multiline,
        "BehaviorID": "stub", "FunctionalCategory": "contextual",
        "SemanticCategory": "harmful", "Tags": "context",
    }
    expected = f"{multiline}\n\n---\n\nObfuscate this"
    assert render_test_case(row) == expected


@pytest.mark.parametrize("ctx", ["short", "with spaces", "with\nnewlines"])
def test_separator_appears_exactly_once(ctx: str) -> None:
    row = {
        "Behavior": "B",
        "ContextString": ctx,
        "BehaviorID": "stub", "FunctionalCategory": "contextual",
        "SemanticCategory": "harmful", "Tags": "context",
    }
    rendered = render_test_case(row)
    # Find the separator string.
    sep = "\n\n---\n\n"
    assert rendered.count(sep) == 1
