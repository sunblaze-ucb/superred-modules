"""Unit tests for the pure claim/request helpers (no LLM, synthetic strings)."""

from __future__ import annotations

from anecdoctor_optimizer import claims


def test_default_example_claims_are_present_and_stringy() -> None:
    assert isinstance(claims.DEFAULT_EXAMPLE_CLAIMS, tuple)
    assert claims.DEFAULT_EXAMPLE_CLAIMS  # non-empty
    assert all(isinstance(c, str) and c.strip() for c in claims.DEFAULT_EXAMPLE_CLAIMS)


def test_build_evaluation_data_folds_goal_first() -> None:
    data = claims.build_evaluation_data(
        goal="GOAL_CLAIM",
        example_claims=["EX_A", "EX_B"],
        include_goal_as_claim=True,
    )
    assert data == ["GOAL_CLAIM", "EX_A", "EX_B"]


def test_build_evaluation_data_can_exclude_goal() -> None:
    data = claims.build_evaluation_data(
        goal="GOAL_CLAIM",
        example_claims=["EX_A", "EX_B"],
        include_goal_as_claim=False,
    )
    assert data == ["EX_A", "EX_B"]


def test_build_evaluation_data_skips_blank_entries() -> None:
    data = claims.build_evaluation_data(
        goal="   ",
        example_claims=["EX_A", "  ", ""],
        include_goal_as_claim=True,
    )
    assert data == ["EX_A"]


def test_format_few_shot_examples_has_header_and_lines() -> None:
    block = claims.format_few_shot_examples(["EX_A", "EX_B"])
    assert block == "### examples\nEX_A\nEX_B"


def test_assemble_framed_request_joins_framing_and_body() -> None:
    out = claims.assemble_framed_request(framing="FRAMING\n", body="BODY")
    assert out == "FRAMING\n\nBODY"
    # framing precedes the body
    assert out.index("FRAMING") < out.index("BODY")
