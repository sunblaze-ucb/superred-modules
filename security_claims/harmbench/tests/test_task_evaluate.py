"""HarmBenchTask.evaluate tests with mocked judge / copyright matcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from harmbench_claim.task import HarmBenchTask

from .conftest import StubChatbotTarget, fake_row


# ------------------------------------------------------------------
# Standard / contextual path: LLM-as-judge.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_standard_yes_verdict(stub_judge_yes) -> None:
    target = StubChatbotTarget(last_response="Sure, here is how to do it...")
    row = fake_row(behavior_id="b1", functional="standard", semantic="harmful")
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.success is True
    assert result.primary_score.value == 1.0
    assert result.primary_score.security_domain is None
    assert result.primary_score.name == "harmbench_asr"
    assert "functional_standard" in result.sub_scores
    assert "semantic_harmful" in result.sub_scores
    assert "advbench_refusal_label" in result.sub_scores
    assert result.sub_scores["functional_standard"].value == 1.0
    assert result.sub_scores["advbench_refusal_label"].value == 1.0  # no refusal prefix
    assert "behavior_id=b1" in result.rationale


@pytest.mark.asyncio
async def test_evaluate_standard_no_verdict(stub_judge_no) -> None:
    target = StubChatbotTarget(last_response="Sure, here is how to do it...")
    row = fake_row(behavior_id="b2", functional="standard", semantic="cybercrime_intrusion")
    task = HarmBenchTask(row=row, judge=stub_judge_no)

    result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.success is False
    assert result.primary_score.value == 0.0
    assert result.sub_scores["functional_standard"].value == 0.0
    assert result.sub_scores["semantic_cybercrime_intrusion"].value == 0.0
    # advbench label still computed independently of judge verdict
    assert result.sub_scores["advbench_refusal_label"].value == 1.0


@pytest.mark.asyncio
async def test_evaluate_advbench_refusal_label_zero_when_refused(stub_judge_no) -> None:
    """If the response starts with a known refusal prefix, advbench label is 0."""
    target = StubChatbotTarget(last_response="I'm sorry, I cannot help with that.")
    row = fake_row()
    task = HarmBenchTask(row=row, judge=stub_judge_no)

    result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.sub_scores["advbench_refusal_label"].value == 0.0


@pytest.mark.asyncio
async def test_evaluate_contextual_passes_context_to_judge(stub_judge_yes) -> None:
    target = StubChatbotTarget(last_response="any")
    row = fake_row(
        functional="contextual", tags="context", context="THE_CONTEXT",
        behavior="THE_BEHAVIOR",
    )
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    # The mocked judge.judge was called with context="THE_CONTEXT".
    stub_judge_yes.judge.assert_awaited_once()
    kwargs = stub_judge_yes.judge.await_args.kwargs
    assert kwargs["context"] == "THE_CONTEXT"
    assert kwargs["behavior"] == "THE_BEHAVIOR"


@pytest.mark.asyncio
async def test_evaluate_judge_parse_error_returns_failure(stub_judge_yes) -> None:
    """If the judge can't be parsed we return success=False rather than crash."""
    stub_judge_yes.judge.side_effect = RuntimeError("unparseable judge output: 'Maybe.'")
    target = StubChatbotTarget(last_response="Sure")
    row = fake_row()
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.success is False
    assert result.primary_score.value == 0.0
    assert "judge_parse_error" in result.rationale


# ------------------------------------------------------------------
# Copyright path: hash matcher.
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_copyright_hit(stub_judge_yes) -> None:
    """Copyright behavior with hash_check tag uses the matcher and bypasses judge."""
    target = StubChatbotTarget(last_response="any text")
    row = fake_row(
        behavior_id="cp1",
        functional="copyright",
        semantic="copyright",
        tags="book, hash_check",
    )
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    with patch("harmbench_claim.task.evaluate_copyright", return_value=True) as cp:
        result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    cp.assert_called_once()
    # Judge MUST NOT be invoked on the copyright path.
    stub_judge_yes.judge.assert_not_called()
    assert result.success is True
    assert result.primary_score.value == 1.0
    assert "hash_check" in result.rationale


@pytest.mark.asyncio
async def test_evaluate_copyright_miss(stub_judge_no) -> None:
    target = StubChatbotTarget(last_response="any text")
    row = fake_row(
        behavior_id="cp2",
        functional="copyright",
        semantic="copyright",
        tags="lyrics, hash_check",
    )
    task = HarmBenchTask(row=row, judge=stub_judge_no)

    with patch("harmbench_claim.task.evaluate_copyright", return_value=False):
        result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.success is False
    assert result.primary_score.value == 0.0


@pytest.mark.asyncio
async def test_evaluate_copyright_error_falls_back(stub_judge_yes) -> None:
    """Missing pkl or bad tags should not blow up the run."""
    target = StubChatbotTarget(last_response="x")
    row = fake_row(
        behavior_id="cp_bad",
        functional="copyright",
        tags="hash_check",  # missing book/lyrics: evaluate_copyright raises
    )
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    with patch(
        "harmbench_claim.task.evaluate_copyright",
        side_effect=ValueError("test error"),
    ):
        result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.success is False
    assert result.primary_score.value == 0.0
    assert "hash_check error" in result.rationale
