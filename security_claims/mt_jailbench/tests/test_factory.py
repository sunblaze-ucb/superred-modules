"""Factory tests: the claim builds one task per standard behavior."""

from __future__ import annotations

from mt_jailbench_claim import mt_jailbench_claim
from mt_jailbench_claim.judge import DEFAULT_JUDGE_TYPES, DEFAULT_THRESHOLD
from mt_jailbench_claim.task import MTJailBenchTask


def test_full_claim_has_159_tasks() -> None:
    claim = mt_jailbench_claim(judge_api_base="x", judge_api_key="y")
    tasks = list(claim)
    assert len(tasks) == 159
    assert all(isinstance(t, MTJailBenchTask) for t in tasks)


def test_number_of_behaviors_caps() -> None:
    claim = mt_jailbench_claim(judge_api_base="x", judge_api_key="y", number_of_behaviors=10)
    assert len(list(claim)) == 10


def test_tasks_share_judge_config_and_defaults() -> None:
    claim = mt_jailbench_claim(judge_api_base="base", judge_api_key="key", judge_model="m")
    task = next(iter(claim))
    assert task._judge_config.model == "m"
    assert task._judge_config.api_base == "base"
    assert task._judge_types == DEFAULT_JUDGE_TYPES
    assert task._threshold == DEFAULT_THRESHOLD


def test_env_fallback_for_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_BASE", "http://env-base")
    monkeypatch.setenv("LITELLM_API_KEY", "env-key")
    task = next(iter(mt_jailbench_claim()))
    assert task._judge_config.api_base == "http://env-base"
    assert task._judge_config.api_key == "env-key"


def test_claim_reiterable() -> None:
    claim = mt_jailbench_claim(judge_api_base="x", judge_api_key="y", number_of_behaviors=3)
    assert len(list(claim)) == 3
    assert len(list(claim)) == 3  # tasks are stateless; iterate again
