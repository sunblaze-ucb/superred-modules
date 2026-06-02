"""End-to-end smoke (credentialed): the AgentHarm claim against the real target.

Asserts that the claim wires up correctly with `inspect-agent-target` and a real
model through the Controller: a behavior runs, its tools are exercised, the harm
score / refusal land in range, `success` matches the strict gate, and
`aggregate()` reproduces AgentHarm's canonical metric keys.

This is the assertive counterpart to the printed demo in ``run.py``.
Skipped without LITELLM_API_KEY+LITELLM_API_BASE (or OPENAI_API_KEY).
Run: LITELLM_API_KEY=... LITELLM_API_BASE=... pytest -m smoke tests/smoke
"""

from __future__ import annotations

import os

import pytest
from inspect_agent_target import (
    SYSTEM_TAG,
    TOOL_OUTPUT_TAG,
    USER_TAG,
)
from superred.core.controller import Controller
from superred.core.interfaces.optimizer import Optimizer
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)

from security_claim_agentharm import (
    agentharm_claim,
    agentharm_target_factory,
    aggregate,
)

pytestmark = pytest.mark.smoke


class _Passthrough(Optimizer):
    async def initialize(self, goal, controllables, observables, llm_client) -> None:  # noqa: ANN001
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, (ControllablePreCallEvent, ControllablePostCallEvent)):
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)


def _creds() -> tuple[str, str | None, str | None] | None:
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return (
            os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini"),
            os.environ["LITELLM_API_BASE"],
            os.environ["LITELLM_API_KEY"],
        )
    if "OPENAI_API_KEY" in os.environ:
        return (os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"), None, os.environ["OPENAI_API_KEY"])
    return None


@pytest.mark.asyncio
async def test_claim_runs_end_to_end_against_target() -> None:
    creds = _creds()
    if creds is None:
        pytest.skip("no LLM credentials")
    model, api_base, api_key = creds

    controller = Controller(
        optimizer_factory=_Passthrough,
        target_factory=agentharm_target_factory(
            target_model=model, api_base=api_base, api_key=api_key, concurrency=1
        ),
        security_claim=agentharm_claim(
            target_model=model, api_base=api_base, api_key=api_key, behavior_ids=["1-1"]
        ),
        scope=frozenset({USER_TAG, SYSTEM_TAG, TOOL_OUTPUT_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()
    tr = result.task_results[0]
    ev = tr.best_evaluation
    assert ev is not None

    score = float(ev.primary_score.value)
    refusal = float(ev.sub_scores["refusal"].value)
    assert tr.stop_reason == "done"
    assert 0.0 <= score <= 1.0
    assert refusal in (0.0, 1.0)
    # strict-gate success consistency
    assert bool(tr.success) == (refusal == 0.0 and score == 1.0)
    # grading actually ran (per-check sub-scores present) and no scoring errors
    assert any(k.startswith("check_") for k in ev.sub_scores)
    assert ev.sub_scores["grading_error"].value == 0.0
    assert ev.sub_scores["refusal_judge_error"].value == 0.0

    # aggregate reproduces AgentHarm's canonical metric keys (+ per-category)
    agg = aggregate([ev])
    for key in ("avg_score", "avg_full_score", "avg_refusals", "avg_score_non_refusals"):
        assert key in agg
    assert any(k.endswith("_avg_scores") for k in agg)
