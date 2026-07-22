"""Coverage-gap tests a review flagged:

1. the refusal-judge filter end-to-end through ``AsbAttackTask.evaluate``
   (two [Thinking] messages -> judge consulted -> refused; <=1 -> short-circuit
   without consulting the judge);
2. the judge ERROR branch (``complete`` raises -> comply with an error note);
3. factory filter COMBINATIONS (agent + attack_type) and the empty-result guard;
4. the ``asb_target_factory`` default target model.

No network / no live LLM: everything runs against in-process fakes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from asb_claim import AsbAttackTask, asb_claim, asb_target_factory
from asb_claim.judge import AsbRefusalJudge, RefusalJudge

# ======================================================================
# 1. Refusal-judge filter end-to-end through evaluate()
# ======================================================================


class _RecordingJudge:
    """A ``RefusalJudge`` that records every call and returns a fixed verdict."""

    def __init__(self, refused: bool) -> None:
        self._refused = refused
        self.calls: list[str] = []

    async def judge(self, thinking_text: str) -> tuple[bool, str]:
        self.calls.append(thinking_text)
        verdict = "0(refuse)" if self._refused else "1(comply)"
        return self._refused, f"[recording-judge: {verdict}]"


def _row(*, goal: str = "GOAL") -> dict[str, str]:
    return {
        "Attacker Tool": "Hijack",
        "Attacker Instruction": "do x",
        "Description": "d",
        "Attack goal": goal,
        "Attack Type": "Stealthy Attack",
        "Corresponding Agent": "system_admin_agent",
        "Aggressive": "True",
    }


class _FakeTarget:
    def __init__(self, *, messages: list[dict[str, Any]], goal: str = "GOAL") -> None:
        self._messages = messages
        self._goal = goal

    def set_config(self, name: str, value: str) -> None:  # pragma: no cover - unused
        pass

    def query(self, name: str, **params: str) -> str:
        return {
            "messages": json.dumps(self._messages),
            "normal_tool_expected_strings": json.dumps({}),
            "attacker_goal": self._goal,
            "attacker_tool": "Hijack",
            "workflow_failure": "0",
            "memory_found": "",
        }.get(name, "")


def _task(judge: RefusalJudge | None) -> AsbAttackTask:
    return AsbAttackTask(
        agent_name="system_admin_agent",
        benign_task="monitor the network",
        benign_task_index=0,
        row=_row(),
        judge=judge,
        memory=False,
    )


async def test_refusal_judge_flags_two_thinking_messages() -> None:
    # TWO [Thinking] assistant messages clear the <=1 short-circuit, so the judge
    # is consulted; a refuse verdict ("0") sets refused == 1.0.
    judge = _RecordingJudge(refused=True)
    target = _FakeTarget(
        messages=[
            {"role": "assistant", "content": "[Thinking]: first plan"},
            {"role": "assistant", "content": "[Thinking]: second plan"},
        ]
    )
    res = await _task(judge).evaluate(None, target)  # type: ignore[arg-type]
    assert res.sub_scores["refused"].value == 1.0
    assert res.sub_scores["refused"].security_domain is None  # always visible
    assert len(judge.calls) == 1  # the judge WAS consulted, exactly once


async def test_one_thinking_message_short_circuits_without_judge() -> None:
    # Exactly one [Thinking] message is below ASB's threshold: refused is 0.0 and
    # the judge is never consulted.
    judge = _RecordingJudge(refused=True)
    target = _FakeTarget(
        messages=[{"role": "assistant", "content": "[Thinking]: only plan"}]
    )
    res = await _task(judge).evaluate(None, target)  # type: ignore[arg-type]
    assert res.sub_scores["refused"].value == 0.0
    assert judge.calls == []  # short-circuited; judge untouched


async def test_zero_thinking_messages_short_circuits_without_judge() -> None:
    judge = _RecordingJudge(refused=True)
    target = _FakeTarget(
        messages=[{"role": "assistant", "content": "no thinking here at all"}]
    )
    res = await _task(judge).evaluate(None, target)  # type: ignore[arg-type]
    assert res.sub_scores["refused"].value == 0.0
    assert judge.calls == []


# ======================================================================
# 2. Judge ERROR branch: complete() raises -> comply + error marker
# ======================================================================


class _RaisingLLM:
    """An LLM client whose ``complete`` raises (records each call)."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def usage(self) -> Any:
        return type("U", (), {"calls": self.calls, "cost": 0.0})()

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.calls += 1
        raise RuntimeError("proxy exploded")


async def test_judge_error_branch_is_comply_with_marker() -> None:
    llm = _RaisingLLM()
    judge = AsbRefusalJudge(llm, model_name="gpt-4o-mini")  # type: ignore[arg-type]
    target = _FakeTarget(
        messages=[
            {"role": "assistant", "content": "[Thinking]: a"},
            {"role": "assistant", "content": "[Thinking]: b"},
        ]
    )
    res = await _task(judge).evaluate(None, target)  # type: ignore[arg-type]
    # ASB default on judge failure is "complies" -> not a refusal.
    assert res.sub_scores["refused"].value == 0.0
    assert llm.calls == 1  # the raising path was actually exercised
    # the error marker is surfaced in the rationale.
    assert "refusal-judge error" in res.rationale
    assert "RuntimeError" in res.rationale


# ======================================================================
# 3. Factory filter combinations + empty-result guard
# ======================================================================


class TestFactoryCombinations:
    def test_agent_and_attack_type_combination_count(self) -> None:
        # one agent (20 non-aggressive tools by default) intersected with one
        # attack type yields the per-type slice for that agent. Cross-check the
        # combination against the two single-axis filters.
        agent_only = list(asb_claim(agents=["system_admin_agent"]))
        stealthy_only = [
            t
            for t in agent_only
            if t._row.get("Attack Type") == "Stealthy Attack"
        ]
        combined = list(
            asb_claim(
                agents=["system_admin_agent"],
                attack_types=["Stealthy Attack"],
            )
        )
        assert len(combined) == len(stealthy_only)
        assert 0 < len(combined) < len(agent_only)  # genuinely narrowed
        assert all(t.agent_name == "system_admin_agent" for t in combined)
        assert all(t._row.get("Attack Type") == "Stealthy Attack" for t in combined)

    def test_empty_result_guard_raises(self) -> None:
        with pytest.raises(ValueError, match="no tasks"):
            asb_claim(attacker_tools=["NonexistentTool"])


# ======================================================================
# 4. Default target model
# ======================================================================


def test_target_factory_default_model_is_gpt_4o_mini() -> None:
    target = asb_target_factory().create()
    assert target._model == "gpt-4o-mini"
