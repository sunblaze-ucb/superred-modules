"""grading_bridge: score mean, error transparency, refusal loop (mocked)."""

from __future__ import annotations

import types

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

from agentharm_claim import grading_bridge as gb


@pytest.mark.asyncio
async def test_grade_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fn(md, msgs):  # noqa: ANN001
        return {"a": True, "b": False, "c": True}

    monkeypatch.setattr(gb, "harmful_grading_functions", types.SimpleNamespace(grade_x=fake_fn))
    score, sd, err = await gb.grade({"grading_function": "grade_x"}, [], semantic_judge=None)
    assert score == pytest.approx(2 / 3)
    assert sd == {"a": True, "b": False, "c": True}
    assert err is None


@pytest.mark.asyncio
async def test_grade_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(md, msgs):  # noqa: ANN001
        raise RuntimeError("kaboom")

    monkeypatch.setattr(gb, "harmful_grading_functions", types.SimpleNamespace(grade_x=boom))
    score, sd, err = await gb.grade({"grading_function": "grade_x"}, [], semantic_judge=None)
    assert score == 0.0 and sd == {}
    assert err is not None and "RuntimeError" in err


@pytest.mark.asyncio
async def test_check_refusal_first_wins() -> None:
    class _Judge:
        async def check_refusal(self, text: str):  # noqa: ANN202
            return ("refuse" in text, "expl")

    msgs = [ChatMessageAssistant(content="ok"), ChatMessageAssistant(content="i refuse")]
    refusal, err = await gb.check_refusal(msgs, _Judge())  # type: ignore[arg-type]
    assert refusal is True and err is None


@pytest.mark.asyncio
async def test_check_refusal_skips_empty_and_non_assistant() -> None:
    class _Judge:
        async def check_refusal(self, text: str):  # noqa: ANN202
            return (False, "")

    msgs = [ChatMessageUser(content="hi"), ChatMessageAssistant(content="")]
    refusal, err = await gb.check_refusal(msgs, _Judge())  # type: ignore[arg-type]
    assert refusal is False and err is None


def test_build_judges_offline() -> None:
    # construction must not require network
    assert gb.build_judge_model("openai/gpt-4o-2024-08-06", api_base="b", api_key="k") is not None
    assert gb.build_refusal_judge("openai/gpt-4o-2024-08-06", api_base="b", api_key="k") is not None
