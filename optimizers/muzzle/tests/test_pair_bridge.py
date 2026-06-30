"""Tests for the MUZZLE offline PAIR bridge (:mod:`muzzle_optimizer.pair.bridge`).

The bridge runs a self-contained PAIR loop in which one ``LLMClient`` plays the
attacker (proposes a payload), the *surrogate victim* (answers it while
conditioned on the real victim's system prompt), and the judge (scores it). No
framework runs are consumed. These tests drive it with the role-routing
``ScriptedLLM`` fake and assert the splice/condition/score/feedback wiring plus
the two control-flow exits (early jailbreak return and budget propagation).
"""

from __future__ import annotations

import pytest

from conftest import ScriptedLLM
from muzzle_optimizer.constants import PLACEHOLDER_INSTR
from muzzle_optimizer.pair.bridge import PairBridge, PairResult
from superred.core.types.llm import BudgetExhaustedError, LLMUsage

INSTRUCTION = "do X"
TEMPLATE = "ctx [INSTRUCTION_PLACEHOLDER] ctx"
VICTIM_SYSTEM = "SYS"


def _user_contents(messages: list[dict[str, str]]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "user"]


def _system_contents(messages: list[dict[str, str]]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "system"]


@pytest.mark.asyncio
async def test_craft_returns_pairresult_with_default_attacker_prompt() -> None:
    """The default attacker JSON surfaces as the best payload; result is a PairResult."""
    llm = ScriptedLLM()
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    result = await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=TEMPLATE,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    assert isinstance(result, PairResult)
    assert result.best_prompt == "REFINED PAYLOAD"
    assert result.best_improvement == "i"
    # Default victim ("I will not do that.") is a refusal -> gcg score 1, not a jailbreak.
    assert result.best_score == 1.0
    assert result.is_jailbroken is False


@pytest.mark.asyncio
async def test_surrogate_victim_gets_system_prompt_and_spliced_candidate() -> None:
    """The victim sees the victim_system_prompt as a system message and the spliced payload."""
    seen: list[list[dict[str, str]]] = []

    def victim(messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        return "I will not do that."

    llm = ScriptedLLM(overrides={"victim": victim})
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=TEMPLATE,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    assert len(seen) == 1
    messages = seen[0]
    assert _system_contents(messages) == [VICTIM_SYSTEM]
    user_content = _user_contents(messages)[-1]
    # The candidate is spliced into the template: the placeholder token is consumed.
    assert user_content == "ctx REFINED PAYLOAD ctx"
    assert "REFINED PAYLOAD" in user_content
    assert PLACEHOLDER_INSTR not in user_content


@pytest.mark.asyncio
async def test_gcg_compliant_answer_scores_ten_and_returns_early() -> None:
    """A long non-refusal answer trips the gcg dictionary -> score 10 -> early jailbreak return."""
    long_prompt = "please provide the complete detailed instructions for this task"
    compliant = "here is the full plan to accomplish your stated objective step by step"
    attacker_json = f'{{"improvement": "i", "prompt": "{long_prompt}"}}'
    llm = ScriptedLLM(overrides={"pair_attacker": attacker_json, "victim": lambda _m: compliant})
    # Two iterations available, but the first jailbreak must short-circuit the loop.
    bridge = PairBridge(llm, n_streams=1, n_iterations=2, judge_method="gcg")

    result = await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=None,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    assert result.is_jailbroken is True
    assert result.best_score == 10.0
    assert result.best_prompt == long_prompt
    # gcg is a local refusal-dictionary check: the judge role is never queried.
    assert "pair_judge" not in llm.roles_called()
    # Early return: the second iteration never fires another attacker round.
    assert llm.roles_called().count("pair_attacker") == 1


@pytest.mark.asyncio
async def test_none_template_sends_raw_candidate() -> None:
    """With no transcript template, the raw proposal prompt is sent to the victim verbatim."""
    seen: list[list[dict[str, str]]] = []

    def victim(messages: list[dict[str, str]]) -> str:
        seen.append(messages)
        return "I will not do that."

    llm = ScriptedLLM(overrides={"victim": victim})
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=None,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    assert _user_contents(seen[0])[-1] == "REFINED PAYLOAD"


@pytest.mark.asyncio
async def test_reflection_folded_into_attacker_seed_message() -> None:
    """Reflection payload + recommendations are injected into the attacker's seed user message."""
    reflection = {"most_recent_working_payload": "p", "recommendations": ["r"]}
    llm = ScriptedLLM()
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=None,
        victim_system_prompt=VICTIM_SYSTEM,
        reflection=reflection,
    )

    attacker_msgs = [msgs for role, msgs in llm.calls if role == "pair_attacker"]
    assert attacker_msgs
    seed = _user_contents(attacker_msgs[0])[-1]
    assert "PRIOR ATTEMPT FEEDBACK" in seed
    assert "Most recent working payload: p" in seed
    assert "Recommendations: r" in seed


@pytest.mark.asyncio
async def test_no_reflection_leaves_seed_message_clean() -> None:
    """Without reflection, no feedback block leaks into the attacker's seed message."""
    llm = ScriptedLLM()
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=None,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    attacker_msgs = [msgs for role, msgs in llm.calls if role == "pair_attacker"]
    seed = _user_contents(attacker_msgs[0])[-1]
    assert "PRIOR ATTEMPT FEEDBACK" not in seed


@pytest.mark.asyncio
async def test_unparseable_attacker_skips_stream_and_returns_none() -> None:
    """An attacker that never yields valid JSON is skipped: best_prompt is None, no crash."""
    llm = ScriptedLLM(overrides={"pair_attacker": "this is not valid json"})
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    result = await bridge.craft(
        instruction=INSTRUCTION,
        transcript_template=None,
        victim_system_prompt=VICTIM_SYSTEM,
    )

    assert result.best_prompt is None
    assert result.best_score == 0.0
    assert result.is_jailbroken is False
    # The stream is abandoned before the victim is ever queried.
    assert "victim" not in llm.roles_called()
    # The attacker exhausts its full retry budget (default 5) before raising.
    assert llm.roles_called().count("pair_attacker") == 5


@pytest.mark.asyncio
async def test_budget_exhausted_error_propagates() -> None:
    """Budget exhaustion from the surrogate victim is not swallowed: craft re-raises it."""

    def victim(_messages: list[dict[str, str]]) -> str:
        raise BudgetExhaustedError("budget exhausted", usage=LLMUsage())

    llm = ScriptedLLM(overrides={"victim": victim})
    bridge = PairBridge(llm, n_streams=1, n_iterations=1, judge_method="gcg")

    with pytest.raises(BudgetExhaustedError):
        await bridge.craft(
            instruction=INSTRUCTION,
            transcript_template=TEMPLATE,
            victim_system_prompt=VICTIM_SYSTEM,
        )
