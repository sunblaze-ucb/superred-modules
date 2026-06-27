"""Every claim task is 1:1 faithful to its upstream AgentHarm dataset sample.

Offline (uses the cached dataset; no LLM). Verifies that the claim does not
drop, add, or alter behaviors: each AgentHarmTask carries exactly the upstream
sample's prompt, target_functions, grading_function, and category, every task's
grading function exists in the upstream grading module, and every task's tools
resolve via the AgentHarm resolver.
"""

from __future__ import annotations

from inspect_evals.agentharm.benchmark import harmful_grading_functions

from security_claim_agentharm import agentharm_total_claim
from security_claim_agentharm.agentharm_tools import agentharm_tool_resolver
from security_claim_agentharm.dataset_loader import dataset_by_id, load_agentharm_dataset

_MODEL = "openai/gpt-4o-mini"


def _tasks() -> list:
    # Faithfulness to upstream is the direct-request mode (benign_cover=False): all
    # 176 harmful behaviours present 1:1. The default benign-cover mode deliberately
    # excludes base 22 (172), so it is not the right basis for a 1:1-with-upstream check.
    return list(agentharm_total_claim(target_model=_MODEL, benign_cover=False))


def test_tasks_are_one_to_one_with_upstream_dataset() -> None:
    by_id = dataset_by_id(load_agentharm_dataset())
    tasks = _tasks()
    assert len(tasks) == len(by_id) == 176
    assert {t.behavior_id for t in tasks} == set(by_id)
    for t in tasks:
        md = by_id[t.behavior_id].metadata
        assert t._prompt == md["prompt"]
        assert t._metadata["target_functions"] == md["target_functions"]
        assert t._metadata["grading_function"] == md["grading_function"]
        assert t.category == md["category"]


def test_every_task_grading_function_exists_upstream() -> None:
    for t in _tasks():
        name = t._metadata["grading_function"]
        fn = getattr(harmful_grading_functions, name, None)
        assert callable(fn), f"upstream grading function missing: {name}"


def test_every_task_tool_resolves() -> None:
    for t in _tasks():
        for name in t._metadata["target_functions"]:
            assert agentharm_tool_resolver(name) is not None
