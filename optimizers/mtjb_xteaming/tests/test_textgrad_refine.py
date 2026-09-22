"""Env-gated test for the TextGrad refine path.

Skipped whenever ``textgrad`` is not installed (the shared venv does not ship
it), so in the offline battery this appears as a skip, not a failure. When
``textgrad`` is present, it checks that the refine step routes through the given
LLMClient and returns a refined prompt.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textgrad")

from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from mtjb_xteaming_optimizer.textgrad_refine import refine_prompt  # noqa: E402


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


def test_refine_prompt_returns_a_string() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=_resp("REFINED SUGGESTION"))
    result = asyncio.run(
        refine_prompt(
            llm=llm,
            attacker_prompt="ATTACK",
            target_response="RESPONSE",
            score=2,
            reason="too vague",
            behavior="GOAL_TEXT",
        )
    )
    assert isinstance(result, str)
    assert llm.complete.await_count >= 1
