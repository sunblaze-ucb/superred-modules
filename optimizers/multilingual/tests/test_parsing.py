"""Tests for the multilingual reply parsers and compliance prompt."""

from __future__ import annotations

from multilingual_optimizer.compliance import build_compliance_check_prompt
from multilingual_optimizer.parsing import (
    parse_enhanced_attack,
    parse_is_translation,
    parse_non_compliant,
    rebuild_dump,
)


def test_parse_enhanced_attack() -> None:
    assert parse_enhanced_attack('{"translation_logic":"L","input":"T"}') == ("L", "T")
    assert parse_enhanced_attack('```json\n{"input":"T"}\n```') == ("", "T")


def test_enhanced_attack_requires_nonempty_input() -> None:
    assert parse_enhanced_attack('{"input":"   "}') is None
    assert parse_enhanced_attack('{"translation_logic":"x"}') is None
    assert parse_enhanced_attack("garbage") is None


def test_parse_is_translation() -> None:
    assert parse_is_translation('{"is_translation":true}') is True
    assert parse_is_translation('{"is_translation":false}') is False
    assert parse_is_translation('{"is_translation":"true"}') is True
    assert parse_is_translation("nope") is None


def test_parse_non_compliant() -> None:
    assert parse_non_compliant('{"non_compliant":true}') is True
    assert parse_non_compliant('{"non_compliant":false}') is False
    assert parse_non_compliant("nope") is None


def test_rebuild_dump_matches_upstream_shape() -> None:
    assert rebuild_dump("L", "T") == {"translation_logic": "L", "input": "T"}


def test_compliance_prompt_embeds_the_text_and_asks_for_non_compliant() -> None:
    p = build_compliance_check_prompt("ANALYZE_ME")
    assert "ANALYZE_ME" in p
    assert "non_compliant" in p
    assert "Hard Refusal" in p


def test_json_wrapped_in_prose_is_still_read() -> None:
    """Upstream constrains replies with pydantic; this module does not.

    A chatty reply is therefore a realistic shape here, and rejecting it costs
    a retry and eventually degrades the attack to the untranslated goal.
    """
    raw = 'Sure! Here is the JSON you asked for: {"translation_logic": "bn", "input": "TRANSLATED"} Hope that helps.'
    assert parse_enhanced_attack(raw) == ("bn", "TRANSLATED")


def test_prose_without_any_json_object_is_still_rejected() -> None:
    assert parse_enhanced_attack("I cannot help with that request.") is None
