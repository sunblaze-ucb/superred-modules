"""Tests for JailbreakBenchTask and the claim factory."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from jailbreakbench_claim import jailbreakbench_claim, load_behaviors
from jailbreakbench_claim.judge import JailbreakBenchJudge
from jailbreakbench_claim.task import JailbreakBenchTask


def _stub_judge(verdict: bool) -> JailbreakBenchJudge:
    j = JailbreakBenchJudge(model="test/stub", api_base="http://x", api_key="k")
    j.judge = AsyncMock(return_value=(verdict, f"<stub {verdict}>"))  # type: ignore[method-assign]
    return j


def _task(verdict: bool = True) -> JailbreakBenchTask:
    row = load_behaviors(limit=1)[0]
    return JailbreakBenchTask(row=row, judge=_stub_judge(verdict))


def test_goal_is_the_behaviour_goal() -> None:
    row = load_behaviors(limit=1)[0]
    task = JailbreakBenchTask(row=row, judge=_stub_judge(True))
    assert task.goal.description == row["Goal"]


def test_metadata_exposed() -> None:
    row = load_behaviors(limit=1)[0]
    task = JailbreakBenchTask(row=row, judge=_stub_judge(True))
    assert task.index == row["Index"]
    assert task.category == row["Category"]
    assert task.source == row["Source"]


class _FakeChatbot:
    """Minimal ChatbotTarget stand-in: records config, returns a canned reply."""

    def __init__(self, reply: str) -> None:
        self.config: dict[str, str] = {}
        self._reply = reply

    def set_config(self, name: str, value: str) -> None:
        self.config[name] = value

    def query(self, name: str) -> str:
        return self._reply if name == "last_response" else ""


def test_configure_clears_system_prompt() -> None:
    task = _task()
    target = _FakeChatbot("whatever")
    asyncio.run(task.configure_target(target))
    assert target.config["system_prompt"] == ""


def test_evaluate_success_when_judge_says_unsafe() -> None:
    task = _task(verdict=True)
    result = asyncio.run(task.evaluate(None, _FakeChatbot("here is how...")))
    assert result.success is True
    assert result.primary_score.value == 1.0
    assert result.primary_score.name == "jbb_asr"


def test_evaluate_failure_when_judge_says_safe() -> None:
    task = _task(verdict=False)
    result = asyncio.run(task.evaluate(None, _FakeChatbot("I can't help")))
    assert result.success is False
    assert result.primary_score.value == 0.0


def test_evaluate_passes_goal_and_response_to_judge() -> None:
    row = load_behaviors(limit=1)[0]
    judge = _stub_judge(True)
    task = JailbreakBenchTask(row=row, judge=judge)
    asyncio.run(task.evaluate(None, _FakeChatbot("RESP")))
    judge.judge.assert_awaited_once_with(goal=row["Goal"], response="RESP")


def test_evaluate_carries_category_subscore() -> None:
    row = load_behaviors(limit=1)[0]
    task = JailbreakBenchTask(row=row, judge=_stub_judge(True))
    result = asyncio.run(task.evaluate(None, _FakeChatbot("x")))
    assert f"category_{row['Category']}" in result.sub_scores


def test_factory_builds_one_task_per_behaviour() -> None:
    claim = jailbreakbench_claim(limit=5)
    assert len(list(claim)) == 5


def test_factory_all_hundred() -> None:
    assert len(list(jailbreakbench_claim())) == 100


def test_factory_filters_by_category() -> None:
    from jailbreakbench_claim import all_categories

    cat = all_categories()[0]
    tasks = list(jailbreakbench_claim(categories=[cat]))
    assert tasks
    assert all(t.category == cat for t in tasks)


def test_factory_tasks_share_one_judge() -> None:
    tasks = list(jailbreakbench_claim(limit=3, judge_model="m"))
    assert len({id(t.judge) for t in tasks}) == 1
