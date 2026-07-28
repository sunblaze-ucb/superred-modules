"""HarmBenchTask.evaluate tests with mocked judge / copyright matcher."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from harmbench_claim.judge_failure import JudgeUnavailableError, JudgeUnparseableError
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
async def test_evaluate_propagates_an_unparseable_judge(stub_judge_yes) -> None:
    """A judge that never answered produces no EvaluationResult.

    It used to produce success=False with primary_score=0.0, which reads
    downstream as "the target refused" -- a broken judge scored as a
    robust target. The controller now records stop_reason="error" with
    the traceback, excludes the task from the ASR denominator, and
    re-measures it on resume.
    """
    stub_judge_yes.judge.side_effect = JudgeUnparseableError(
        judge_model="openai/gpt-4o-mini", attempts=3, raw_output="Maybe.",
    )
    target = StubChatbotTarget(last_response="Sure")
    row = fake_row()
    task = HarmBenchTask(row=row, judge=stub_judge_yes)

    with pytest.raises(JudgeUnparseableError):
        await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_evaluate_propagates_an_unavailable_judge(stub_judge_yes) -> None:
    """The cross-claim invariant, asserted at the Task boundary."""
    stub_judge_yes.judge.side_effect = JudgeUnavailableError(
        judge_model="openai/gpt-5.4",
        attempts=1,
        kind="terminal",
        detail="UnsupportedParamsError: temperature=0.0 not supported",
    )
    target = StubChatbotTarget(last_response="Sure")
    task = HarmBenchTask(row=fake_row(), judge=stub_judge_yes)

    with pytest.raises(JudgeUnavailableError):
        await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_evaluate_marks_that_a_judge_produced_the_verdict(stub_judge_yes) -> None:
    """``judge_ran`` is always 1.0, and that is the assertion.

    Unscoped so the controller's scope filter cannot remove it.
    """
    target = StubChatbotTarget(last_response="Sure")
    task = HarmBenchTask(row=fake_row(), judge=stub_judge_yes)

    result = await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]

    assert result.sub_scores["judge_ran"].value == 1.0
    assert result.sub_scores["judge_ran"].security_domain is None


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
async def test_evaluate_copyright_error_propagates(stub_judge_yes) -> None:
    """A missing reference pickle is an environment fault, not a refusal.

    It used to be scored 0.0 with the error tucked into the rationale,
    which is indistinguishable from a target that reproduced nothing.
    """
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
        with pytest.raises(ValueError, match="test error"):
            await task.evaluate(trajectory=None, target=target)  # type: ignore[arg-type]
