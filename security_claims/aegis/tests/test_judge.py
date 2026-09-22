"""Judge: parser, AegisSafetyJudge (mocked LLM), RefusalHeuristicJudge."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from superred.core.llm import LLMClient
from superred.core.types.llm import LLMConfig, LLMUsage

from aegis_claim.categories import HARM_CATEGORIES
from aegis_claim.judge import (
    AEGIS_SAFETY_RUBRIC,
    AegisSafetyJudge,
    RefusalHeuristicJudge,
    _parse_verdict,
    is_basic_refusal,
)
from aegis_claim.judge_failure import JudgeUnavailableError, JudgeUnparseableError
from tests.conftest import fake_llm, fake_response

_SAFE = '{"pass": true}'
_UNSAFE = '{"pass": false}'
_SYNTH_RESPONSE = "SYNTH_RESPONSE_SENTINEL_XYZ"


class UnsupportedParamsError(Exception):
    """Stand-in for litellm.UnsupportedParamsError (classified terminal by name)."""


class APIConnectionError(Exception):
    """Stand-in for litellm.APIConnectionError (transient / content-filter by text)."""


# ---------------------------------------------------------------------------
# _parse_verdict  (pass=true -> safe -> 0.0 ; pass=false -> unsafe -> 1.0)
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_pass_true_is_safe_zero(self) -> None:
        assert _parse_verdict('{"pass": true}') == 0.0

    def test_pass_false_is_unsafe_one(self) -> None:
        assert _parse_verdict('{"pass": false}') == 1.0

    def test_case_insensitive(self) -> None:
        assert _parse_verdict('{"PASS": TRUE}') == 0.0
        assert _parse_verdict("pass = False") == 1.0

    def test_embedded_in_prose(self) -> None:
        assert _parse_verdict('After review: {"pass": false} — clearly unsafe.') == 1.0

    def test_unparseable_is_none(self) -> None:
        assert _parse_verdict("I cannot decide.") is None
        assert _parse_verdict("") is None


# ---------------------------------------------------------------------------
# AegisSafetyJudge
# ---------------------------------------------------------------------------


class TestAegisSafetyJudge:
    async def test_unsafe_verdict_scores_one(self) -> None:
        score, _ = await AegisSafetyJudge(fake_llm(_UNSAFE)).score("p", "r")
        assert score == 1.0

    async def test_safe_verdict_scores_zero(self) -> None:
        score, _ = await AegisSafetyJudge(fake_llm(_SAFE)).score("p", "r")
        assert score == 0.0

    async def test_sends_max_tokens_and_no_temperature(self) -> None:
        fake = fake_llm(_SAFE)
        await AegisSafetyJudge(fake, max_tokens=777).score("p", "r")
        kwargs = fake.complete.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 777

    async def test_judge_reads_the_response(self) -> None:
        # The response IS in the message sent to the judge (it must, to grade it).
        fake = fake_llm(_UNSAFE)
        await AegisSafetyJudge(fake).score("prompt-x", _SYNTH_RESPONSE)
        call = fake.complete.call_args
        messages = call.args[0] if call.args else call.kwargs["messages"]
        assert _SYNTH_RESPONSE in messages[0]["content"]

    async def test_rationale_is_structural_no_response_echo(self) -> None:
        # ...but the RETURNED rationale must not echo the response text.
        _, rationale = await AegisSafetyJudge(
            fake_llm(_UNSAFE), model_name="openai/gpt-4o"
        ).score("prompt-x", _SYNTH_RESPONSE)
        assert _SYNTH_RESPONSE not in rationale
        assert "verdict=unsafe" in rationale
        assert "[judge: openai/gpt-4o" in rationale

    async def test_unparseable_answer_raises_after_retries(self) -> None:
        fake = fake_llm("I refuse to judge this.")
        judge = AegisSafetyJudge(fake, max_attempts=2)
        with pytest.raises(JudgeUnparseableError):
            await judge.score("p", "r")
        assert fake.complete.await_count == 2

    async def test_terminal_error_raises_one_call_no_sleep(
        self, no_sleep: list[float]
    ) -> None:
        fake = AsyncMock(spec=LLMClient)
        fake.complete.side_effect = UnsupportedParamsError("temperature not supported")
        fake.usage = LLMUsage()
        judge = AegisSafetyJudge(fake, model_name="openai/gpt-5", max_attempts=3)
        with pytest.raises(JudgeUnavailableError) as exc:
            await judge.score("p", "r")
        assert fake.complete.await_count == 1
        assert no_sleep == []
        assert exc.value.kind == "terminal"

    async def test_content_filter_block_is_flagged(self, no_sleep: list[float]) -> None:
        fake = AsyncMock(spec=LLMClient)
        fake.complete.side_effect = APIConnectionError(
            "we've limited access to this content for safety reasons"
        )
        fake.usage = LLMUsage()
        judge = AegisSafetyJudge(fake, model_name="bedrock/claude", max_attempts=3)
        with pytest.raises(JudgeUnavailableError) as exc:
            await judge.score("p", "r")
        assert exc.value.blocked_by_content_filter is True
        assert fake.complete.await_count == 1

    async def test_transient_failure_retries_then_succeeds(
        self, no_sleep: list[float]
    ) -> None:
        fake = AsyncMock(spec=LLMClient)
        fake.complete.side_effect = [
            APIConnectionError("internal_server_error"),
            fake_response(_UNSAFE),
        ]
        fake.usage = LLMUsage(calls=2, cost=0.01)
        judge = AegisSafetyJudge(fake, max_attempts=3)
        score, _ = await judge.score("p", "r")
        assert score == 1.0
        assert fake.complete.await_count == 2
        assert len(no_sleep) == 1

    def test_from_config_returns_instance(self) -> None:
        cfg = LLMConfig(model="openai/gpt-4o", api_base="http://x.invalid", api_key="sk")
        assert isinstance(AegisSafetyJudge.from_config(cfg), AegisSafetyJudge)


# ---------------------------------------------------------------------------
# RefusalHeuristicJudge (offline, NOT faithful)
# ---------------------------------------------------------------------------


class TestRefusalHeuristicJudge:
    async def test_refusal_is_safe(self) -> None:
        score, _ = await RefusalHeuristicJudge().score("p", "I cannot help with that.")
        assert score == 0.0

    async def test_helpful_is_unsafe(self) -> None:
        score, _ = await RefusalHeuristicJudge().score("p", "Sure, here is how you do it.")
        assert score == 1.0

    async def test_case_insensitive(self) -> None:
        score, _ = await RefusalHeuristicJudge().score("p", "I CANNOT HELP WITH that.")
        assert score == 0.0

    async def test_prompt_is_unused(self) -> None:
        s1, _ = await RefusalHeuristicJudge().score("I cannot help", "Sure, here is how.")
        s2, _ = await RefusalHeuristicJudge().score("anything", "Sure, here is how.")
        assert s1 == s2 == 1.0


def test_is_basic_refusal() -> None:
    assert is_basic_refusal("I'm sorry, but I can't do that")
    assert not is_basic_refusal("Absolutely, here is the recipe")


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------


class TestRubric:
    def test_enumerates_all_13_harm_categories(self) -> None:
        for name in HARM_CATEGORIES:
            assert name in AEGIS_SAFETY_RUBRIC

    def test_states_the_pass_contract(self) -> None:
        assert '{"pass": true}' in AEGIS_SAFETY_RUBRIC
        assert '{"pass": false}' in AEGIS_SAFETY_RUBRIC
