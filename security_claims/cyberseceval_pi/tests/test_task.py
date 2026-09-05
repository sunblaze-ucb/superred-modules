"""Tests for PromptInjectionTask and the claim factory."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from cyberseceval_pi_claim import cyberseceval_pi_claim, load_cases
from cyberseceval_pi_claim.judge import PromptInjectionJudge
from cyberseceval_pi_claim.task import PromptInjectionTask


def _stub_judge(verdict: bool) -> PromptInjectionJudge:
    j = PromptInjectionJudge(model="t/stub", api_base="http://x", api_key="k")
    j.judge = AsyncMock(return_value=(verdict, f"<stub {verdict}>"))  # type: ignore[method-assign]
    return j


def _task(verdict: bool = True) -> PromptInjectionTask:
    case = load_cases(dataset="english", limit=1)[0]
    return PromptInjectionTask(case=case, judge=_stub_judge(verdict))


class _FakeChatbot:
    def __init__(self, reply: str) -> None:
        self.config: dict[str, str] = {}
        self._reply = reply

    def set_config(self, name: str, value: str) -> None:
        self.config[name] = value

    def query(self, name: str) -> str:
        return self._reply if name == "last_response" else ""


def test_goal_is_the_injection_user_input() -> None:
    case = load_cases(dataset="english", limit=1)[0]
    task = PromptInjectionTask(case=case, judge=_stub_judge(True))
    assert task.goal.description == case.user_input


def test_configure_plants_the_defended_system_prompt() -> None:
    case = load_cases(dataset="english", limit=1)[0]
    task = PromptInjectionTask(case=case, judge=_stub_judge(True))
    target = _FakeChatbot("x")
    asyncio.run(task.configure_target(target))
    assert target.config["system_prompt"] == case.system_prompt
    assert target.config["system_prompt"]  # non-empty: this is the key difference


def test_evaluate_success_when_injection_wins() -> None:
    task = _task(verdict=True)
    result = asyncio.run(task.evaluate(None, _FakeChatbot("The secret is Piano")))
    assert result.success is True
    assert result.primary_score.value == 1.0
    assert result.primary_score.name == "cyberseceval_pi_asr"


def test_evaluate_failure_when_defense_holds() -> None:
    task = _task(verdict=False)
    result = asyncio.run(task.evaluate(None, _FakeChatbot("I can't share that.")))
    assert result.success is False
    assert result.primary_score.value == 0.0


def test_evaluate_passes_response_and_judge_question() -> None:
    case = load_cases(dataset="english", limit=1)[0]
    judge = _stub_judge(True)
    task = PromptInjectionTask(case=case, judge=judge)
    asyncio.run(task.evaluate(None, _FakeChatbot("RESP")))
    judge.judge.assert_awaited_once_with(
        llm_response="RESP", question=case.judge_question
    )


def test_evaluate_carries_type_and_risk_subscores() -> None:
    task = _task(verdict=True)
    result = asyncio.run(task.evaluate(None, _FakeChatbot("x")))
    assert any(k.startswith("type_") for k in result.sub_scores)
    assert any(k.startswith("risk_") for k in result.sub_scores)


def test_factory_builds_one_task_per_case() -> None:
    assert len(list(cyberseceval_pi_claim(limit=20))) == 20


def test_factory_full_english() -> None:
    assert len(list(cyberseceval_pi_claim())) == 251


def test_factory_multilingual() -> None:
    assert len(list(cyberseceval_pi_claim(dataset="multilingual", limit=5))) == 5


def test_factory_tasks_share_one_judge() -> None:
    tasks = list(cyberseceval_pi_claim(limit=3))
    assert len({id(t.judge) for t in tasks}) == 1
