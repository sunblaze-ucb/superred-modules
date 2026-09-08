"""Tests for the math-problem reply parsers and compliance prompt."""

from __future__ import annotations

from mathprompt_optimizer.compliance import build_compliance_check_prompt
from mathprompt_optimizer.parsing import (
    parse_enhanced_attack,
    parse_is_math_problem,
    parse_non_compliant,
    rebuild_dump,
)


def test_parse_enhanced_attack() -> None:
    assert parse_enhanced_attack('{"math_strategy":"S","input":"E"}') == ("S", "E")
    assert parse_enhanced_attack('```json\n{"input":"E"}\n```') == ("", "E")


def test_enhanced_attack_requires_nonempty_input() -> None:
    assert parse_enhanced_attack('{"input":"  "}') is None
    assert parse_enhanced_attack("garbage") is None


def test_parse_is_math_problem() -> None:
    assert parse_is_math_problem('{"is_math_problem":true}') is True
    assert parse_is_math_problem('{"is_math_problem":false}') is False
    assert parse_is_math_problem("nope") is None


def test_parse_non_compliant() -> None:
    assert parse_non_compliant('{"non_compliant":true}') is True
    assert parse_non_compliant('{"non_compliant":false}') is False


def test_rebuild_dump_matches_upstream_shape() -> None:
    assert rebuild_dump("S", "E") == {"math_strategy": "S", "input": "E"}


def test_compliance_prompt_is_shared_and_intact() -> None:
    p = build_compliance_check_prompt("X")
    assert "non_compliant" in p and "Hard Refusal" in p
