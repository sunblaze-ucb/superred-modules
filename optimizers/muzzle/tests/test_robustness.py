"""Robustness coverage for default/fallback paths not hit by the happy-path suite.

Closes the gaps surfaced by the PR #77 review + coverage analysis: the default GPT
PAIR-judge scorer (the optimizer's ``pair_judge_method`` default is ``"gpt"``, which the
unit suite otherwise only exercised via the live smoke), the no-judge scorer, the brain
helpers' retry-then-fallback behavior, and the ``json_utils`` edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import mock_response

from muzzle_optimizer.grafter import Grafter
from muzzle_optimizer.json_utils import extract_json_object, load_system_prompt
from muzzle_optimizer.pair.evaluator import PairEvaluator
from muzzle_optimizer.prompter import Prompter
from muzzle_optimizer.summarizer import Summarizer


class FixedLLM:
    """A fake LLMClient that returns the same content for every call and counts calls."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], **_: Any) -> Any:
        self.calls += 1
        return mock_response(self._content)


# --- PAIR GPT judge (the optimizer default; previously only live-covered) ----
@pytest.mark.asyncio
async def test_pair_gpt_judge_parses_rating() -> None:
    ev = PairEvaluator(llm=FixedLLM("Rating: [[8]]"), judge_method="gpt")
    score = await ev.score_response(goal="g", target_str="t", prompt="p", response="r")
    assert score.score == 8
    assert score.success is False
    assert score.normalized_score == pytest.approx((8 - 1) / 9)


@pytest.mark.asyncio
async def test_pair_gpt_judge_rating_ten_is_success() -> None:
    ev = PairEvaluator(llm=FixedLLM("Rating: [[10]]"), judge_method="gpt")
    assert (await ev.score_response(goal="g", target_str="t", prompt="p", response="r")).success


@pytest.mark.asyncio
async def test_pair_gpt_judge_unparseable_defaults_to_one() -> None:
    ev = PairEvaluator(llm=FixedLLM("no bracketed rating here"), judge_method="gpt")
    assert (await ev.score_response(goal="g", target_str="t", prompt="p", response="r")).score == 1


@pytest.mark.asyncio
async def test_pair_gpt_judge_clamps_out_of_range() -> None:
    ev = PairEvaluator(llm=FixedLLM("Rating: [[99]]"), judge_method="gpt")
    assert (await ev.score_response(goal="g", target_str="t", prompt="p", response="r")).score == 10


@pytest.mark.asyncio
async def test_pair_no_judge_is_neutral() -> None:
    ev = PairEvaluator(llm=FixedLLM("ignored"), judge_method="no-judge")
    score = await ev.score_response(goal="g", target_str="t", prompt="p", response="r")
    assert score.score == 1
    assert score.success is False


def test_pair_evaluator_rejects_unknown_judge_method() -> None:
    with pytest.raises(ValueError):
        PairEvaluator(llm=FixedLLM("x"), judge_method="bogus")


# --- brain helper retry-then-fallback (malformed JSON exhausts retries) -------
@pytest.mark.asyncio
async def test_summarizer_falls_back_after_exhausting_retries() -> None:
    llm = FixedLLM("definitely not json")
    result = await Summarizer(llm, max_tries=3).summarize("a transcript")
    assert result == {}
    assert llm.calls == 3  # one attempt per retry, all failed


@pytest.mark.asyncio
async def test_grafter_falls_back_to_empty_dict() -> None:
    llm = FixedLLM("not json")
    assert await Grafter(llm, max_tries=2).graft({"overview": "x"}) == {}
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_prompter_falls_back_to_empty_string() -> None:
    llm = FixedLLM("not json")
    assert await Prompter(llm, max_tries=2).make_instruction({"overview": "x"}, "the goal") == ""
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_helpers_recover_on_a_later_retry() -> None:
    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages: list[dict[str, str]], **_: Any) -> Any:
            self.calls += 1
            return mock_response("junk" if self.calls == 1 else '{"prompt": "ok"}')

    llm = FlakyLLM()
    out = await Prompter(llm, max_tries=3).make_instruction({"x": 1}, "g")
    assert out == "ok"
    assert llm.calls == 2  # failed once, succeeded on the second attempt


# --- json_utils edge cases ---------------------------------------------------
def test_extract_json_object_edge_cases() -> None:
    assert extract_json_object("not json at all") is None
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('<think>reasoning</think>{"b": 2}') == {"b": 2}
    assert extract_json_object("[1, 2, 3]") is None  # a list is not a dict
    assert extract_json_object("") is None


def test_load_system_prompt_missing_file_falls_back() -> None:
    assert load_system_prompt("/no/such/file.yaml") == "Empty System Prompt"
