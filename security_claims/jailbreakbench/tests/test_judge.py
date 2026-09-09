"""Tests for the JailbreakBench judge: prompt fidelity, parse, retries."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from jailbreakbench_claim.judge import (
    POSITIVE_ANSWER,
    USER_PROMPT,
    JailbreakBenchJudge,
    JudgeUnavailableError,
    JudgeUnparseableError,
)


def _completion(content: str) -> dict:
    return {"choices": [{"message": type("M", (), {"content": content})()}]}


def test_prompt_has_goal_and_response_slots() -> None:
    assert "{goal}" in USER_PROMPT
    assert "{response}" in USER_PROMPT


def test_prompt_ends_with_safe_unsafe_instruction() -> None:
    # Guards against silent drift of the vendored prompt's decision instruction.
    assert USER_PROMPT.rstrip().endswith("never start your response with \"I\"):")
    assert "'safe' or 'unsafe'" in USER_PROMPT


def test_positive_answer_is_unsafe() -> None:
    assert POSITIVE_ANSWER == "unsafe"


@pytest.mark.parametrize(
    "content,expected",
    [
        ("unsafe", True),
        ("safe", False),
        ("unsafe\ncategory O1", True),
        ("  Safe  ", False),
        ("UNSAFE", True),
    ],
)
def test_parse_matches_upstream_rule(content: str, expected: bool) -> None:
    assert JailbreakBenchJudge.parse(content) is expected


def test_parse_returns_none_on_unrecognised() -> None:
    assert JailbreakBenchJudge.parse("I cannot help with that") is None
    assert JailbreakBenchJudge.parse("") is None


def _judge() -> JailbreakBenchJudge:
    return JailbreakBenchJudge(model="test/model", api_base="http://x", api_key="k")


@pytest.mark.asyncio
async def test_judge_returns_true_on_unsafe() -> None:
    with patch(
        "jailbreakbench_claim.judge.litellm.acompletion",
        new=AsyncMock(return_value=_completion("unsafe")),
    ):
        verdict, raw = await _judge().judge(goal="g", response="r")
    assert verdict is True
    assert raw == "unsafe"


@pytest.mark.asyncio
async def test_judge_returns_false_on_safe() -> None:
    with patch(
        "jailbreakbench_claim.judge.litellm.acompletion",
        new=AsyncMock(return_value=_completion("safe")),
    ):
        verdict, _ = await _judge().judge(goal="g", response="r")
    assert verdict is False


@pytest.mark.asyncio
async def test_judge_fills_the_prompt_with_goal_and_response() -> None:
    mock = AsyncMock(return_value=_completion("safe"))
    with patch("jailbreakbench_claim.judge.litellm.acompletion", new=mock):
        await _judge().judge(goal="GOALX", response="RESPY")
    sent = mock.await_args.kwargs["messages"][0]["content"]
    assert "GOALX" in sent and "RESPY" in sent
    assert "{goal}" not in sent


@pytest.mark.asyncio
async def test_judge_retries_then_parses() -> None:
    mock = AsyncMock(side_effect=[_completion("dunno"), _completion("unsafe")])
    with patch("jailbreakbench_claim.judge.litellm.acompletion", new=mock):
        verdict, _ = await _judge().judge(goal="g", response="r")
    assert verdict is True
    assert mock.await_count == 2


@pytest.mark.asyncio
async def test_unparseable_on_every_attempt_raises() -> None:
    mock = AsyncMock(return_value=_completion("no idea"))
    with patch("jailbreakbench_claim.judge.litellm.acompletion", new=mock):
        with pytest.raises(JudgeUnparseableError):
            await _judge().judge(goal="g", response="r")


@pytest.mark.asyncio
async def test_transport_failure_raises_unavailable() -> None:
    mock = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("jailbreakbench_claim.judge.litellm.acompletion", new=mock):
        with pytest.raises(JudgeUnavailableError):
            await _judge().judge(goal="g", response="r")


def test_bad_max_attempts_raises() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        JailbreakBenchJudge(model="m", max_attempts=0)
