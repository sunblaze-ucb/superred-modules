"""AegisTask: construction, configure_target, evaluate (mocked judge)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from chatbot_target import USER_TAG, ChatbotTarget

from aegis_claim.judge import RefusalHeuristicJudge
from aegis_claim.judge_failure import JudgeUnavailableError
from aegis_claim.task import AegisTask

# Synthetic, non-sensitive strings only — never real dataset prompts.
_SYNTH_PROMPT = "SYNTH_PROMPT_PLACEHOLDER"
_SYNTH_UNSAFE_RESPONSE = "Sure, here is the step-by-step content you asked for."
_SYNTH_REFUSAL = "I cannot help with that request."


def _task(
    *,
    prompt: str = _SYNTH_PROMPT,
    risk_category: str = "Violence",
    row_id: str = "row-1",
    judge: object | None = None,
) -> AegisTask:
    return AegisTask(
        prompt=prompt,
        risk_category=risk_category,
        row_id=row_id,
        judge=judge or RefusalHeuristicJudge(),  # type: ignore[arg-type]
    )


def _target(last_response: object = "") -> MagicMock:
    target = MagicMock(spec=ChatbotTarget)
    target.query.return_value = last_response
    return target


class TestConstruction:
    def test_goal_is_the_prompt(self) -> None:
        assert _task(prompt="hello").goal.description == "hello"

    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValueError, match="prompt must be non-empty"):
            _task(prompt="")

    def test_rejects_empty_risk_category(self) -> None:
        with pytest.raises(ValueError, match="risk_category must be non-empty"):
            _task(risk_category="")

    def test_exposes_row_id_and_category(self) -> None:
        t = _task(risk_category="Harassment", row_id="abc")
        assert t.row_id == "abc"
        assert t.risk_category == "Harassment"


class TestConfigureTarget:
    async def test_suppresses_system_prompt(self) -> None:
        target = _target()
        await _task().configure_target(target)
        target.set_config.assert_called_once_with("system_prompt", "")


class TestEvaluate:
    async def test_unsafe_response_is_success_and_score_one(self) -> None:
        result = await _task().evaluate(MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE))
        assert result.success is True
        assert result.primary_score.value == 1.0
        assert result.primary_score.name == "unsafe"

    async def test_safe_response_is_failure_and_score_zero(self) -> None:
        result = await _task().evaluate(MagicMock(), _target(_SYNTH_REFUSAL))
        assert result.success is False
        assert result.primary_score.value == 0.0

    async def test_primary_score_is_unscoped(self) -> None:
        result = await _task().evaluate(MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE))
        assert result.primary_score.security_domain is None

    async def test_evaluate_queries_last_response(self) -> None:
        target = _target(_SYNTH_UNSAFE_RESPONSE)
        await _task().evaluate(MagicMock(), target)
        target.query.assert_called_once_with("last_response")

    async def test_sub_scores_carry_category_with_user_tag(self) -> None:
        result = await _task(risk_category="Harassment").evaluate(
            MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE)
        )
        assert "category_harassment" in result.sub_scores
        sub = result.sub_scores["category_harassment"]
        assert sub.security_domain is USER_TAG
        assert sub.name == "Harassment"
        assert sub.value == 1.0

    async def test_multilabel_yields_one_sub_score_per_category(self) -> None:
        result = await _task(risk_category="Criminal Planning/Confessions, Violence").evaluate(
            MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE)
        )
        assert "category_criminal_planning_confessions" in result.sub_scores
        assert "category_violence" in result.sub_scores

    async def test_judge_ran_is_true_and_unscoped_when_judge_consulted(self) -> None:
        result = await _task().evaluate(MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE))
        assert result.sub_scores["judge_ran"].value == 1.0
        assert result.sub_scores["judge_ran"].security_domain is None


class TestEmptyResponseShortCircuit:
    async def test_empty_response_is_safe_without_judge(self) -> None:
        judge = AsyncMock()
        result = await _task(judge=judge).evaluate(MagicMock(), _target(""))
        assert result.primary_score.value == 0.0
        assert result.success is False
        assert result.sub_scores["judge_ran"].value == 0.0
        judge.score.assert_not_called()

    async def test_none_response_coerced_to_empty_safe(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        judge = AsyncMock()
        with caplog.at_level("WARNING"):
            result = await _task(judge=judge).evaluate(MagicMock(), _target(None))
        assert result.primary_score.value == 0.0
        assert result.sub_scores["judge_ran"].value == 0.0
        judge.score.assert_not_called()
        assert any("returned None" in r.message for r in caplog.records)

    async def test_non_string_response_is_coerced_and_judged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # 12345 is not a refusal phrase → heuristic judge scores it unsafe (1.0).
        with caplog.at_level("WARNING"):
            result = await _task().evaluate(MagicMock(), _target(12345))
        assert result.primary_score.value == 1.0
        assert any("coercing to str" in r.message for r in caplog.records)


class TestAbstainPropagates:
    async def test_judge_abstain_raises_out_of_evaluate(self) -> None:
        judge = AsyncMock()
        judge.score.side_effect = JudgeUnavailableError(
            judge_model="openai/gpt-4o",
            attempts=1,
            kind="terminal",
            detail="UnsupportedParamsError",
        )
        with pytest.raises(JudgeUnavailableError):
            await _task(judge=judge).evaluate(MagicMock(), _target(_SYNTH_UNSAFE_RESPONSE))
