"""evaluate() must emit neither the prompt nor the response text.

The Aegis prompts are unsafe content; the target's response may be too. Neither
may leak into the stored ``EvaluationResult.rationale`` or into logs. The
response is embedded only in the message *sent* to the judge model (unavoidable
for grading), which is not what this test inspects.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from chatbot_target import ChatbotTarget

from aegis_claim.judge import AegisSafetyJudge, RefusalHeuristicJudge
from aegis_claim.task import AegisTask
from tests.conftest import fake_llm

# Distinctive sentinels that would be trivial to spot if they leaked.
_PROMPT_SENTINEL = "PROMPT_SENTINEL_QF83KZ_do_something_forbidden"
_RESPONSE_SENTINEL = "RESPONSE_SENTINEL_QF83KZ_here_is_the_forbidden_content"


def _target(last_response: object) -> MagicMock:
    target = MagicMock(spec=ChatbotTarget)
    target.query.return_value = last_response
    return target


def _task(judge: object) -> AegisTask:
    return AegisTask(
        prompt=_PROMPT_SENTINEL,
        risk_category="Violence",
        row_id="row-sentinel",
        judge=judge,  # type: ignore[arg-type]
    )


async def test_rationale_never_contains_prompt_or_response_llm_judge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    judge = AegisSafetyJudge(fake_llm('{"pass": false}'), model_name="openai/gpt-4o")
    with caplog.at_level(logging.DEBUG):
        result = await _task(judge).evaluate(MagicMock(), _target(_RESPONSE_SENTINEL))

    assert result.primary_score.value == 1.0  # sanity: it did judge
    _assert_clean(result.rationale, caplog)


async def test_rationale_never_contains_prompt_or_response_heuristic_judge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        result = await _task(RefusalHeuristicJudge()).evaluate(
            MagicMock(), _target(_RESPONSE_SENTINEL)
        )
    _assert_clean(result.rationale, caplog)


async def test_empty_response_short_circuit_rationale_is_clean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        result = await _task(RefusalHeuristicJudge()).evaluate(MagicMock(), _target(""))
    assert result.sub_scores["judge_ran"].value == 0.0
    _assert_clean(result.rationale, caplog)


def _assert_clean(rationale: str, caplog: pytest.LogCaptureFixture) -> None:
    assert _PROMPT_SENTINEL not in rationale
    assert _RESPONSE_SENTINEL not in rationale
    for record in caplog.records:
        msg = record.getMessage()
        assert _PROMPT_SENTINEL not in msg
        assert _RESPONSE_SENTINEL not in msg
    # sub-score names carry the category label only, never content.
    # (nothing to assert beyond the rationale/log scan above)
