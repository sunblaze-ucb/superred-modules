"""Tests for SnowballTask and the claim factory."""

from __future__ import annotations

import asyncio

from snowball_claim import snowball_claim
from snowball_claim.loader import SnowballItem, load_items
from snowball_claim.task import SnowballTask


class _FakeChatbot:
    def __init__(self, reply: str) -> None:
        self.config: dict[str, str] = {}
        self._reply = reply

    def set_config(self, name: str, value: str) -> None:
        self.config[name] = value

    def query(self, name: str) -> str:
        return self._reply if name == "last_response" else ""


def _task(subset: str = "graph_connectivity", polarity: str = "yes") -> SnowballTask:
    return SnowballTask(
        item=SnowballItem(subset=subset, question="Is there a route?", polarity=polarity)
    )


def test_goal_is_the_question() -> None:
    task = _task()
    assert task.goal.description == "Is there a route?"


def test_configure_clears_system_prompt() -> None:
    task = _task()
    target = _FakeChatbot("x")
    asyncio.run(task.configure_target(target))
    assert target.config["system_prompt"] == ""


def test_evaluate_success_on_hallucinated_yes() -> None:
    task = _task(polarity="yes")
    result = asyncio.run(task.evaluate(None, _FakeChatbot("Yes, there is a route.")))
    assert result.success is True
    assert result.primary_score.value == 1.0
    assert result.primary_score.name == "snowball_asr"


def test_evaluate_failure_on_correct_refusal() -> None:
    task = _task(polarity="yes")
    result = asyncio.run(task.evaluate(None, _FakeChatbot("No, no such route exists.")))
    assert result.success is False
    assert result.primary_score.value == 0.0


def test_evaluate_respects_no_polarity() -> None:
    task = _task(subset="primes", polarity="no")
    hit = asyncio.run(task.evaluate(None, _FakeChatbot("No, 7411 is not prime.")))
    miss = asyncio.run(task.evaluate(None, _FakeChatbot("Yes, 7411 is prime.")))
    assert hit.success is True
    assert miss.success is False


def test_evaluate_carries_subset_subscore() -> None:
    task = _task(subset="senators", polarity="yes")
    result = asyncio.run(task.evaluate(None, _FakeChatbot("There was a senator.")))
    assert "subset_senators" in result.sub_scores


def test_factory_default_builds_300_tasks() -> None:
    assert len(list(snowball_claim())) == 300


def test_factory_subset_and_limit() -> None:
    tasks = list(snowball_claim(subsets=["primes"], limit=10))
    assert len(tasks) == 10
    assert all(t.subset == "primes" and t.polarity == "no" for t in tasks)


def test_factory_full_size() -> None:
    tasks = list(snowball_claim(subsets=["graph_connectivity"], limit=None))
    assert len(tasks) == 500


def test_factory_tasks_are_real_snowball_questions() -> None:
    tasks = list(snowball_claim(subsets=["primes"], limit=3))
    assert all("prime" in t.goal.description.lower() for t in tasks)
