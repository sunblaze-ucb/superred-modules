"""End-to-end integration test against a real LLM.

Skipped automatically when ``LITELLM_API_KEY`` / ``LITELLM_API_BASE``
are not in the environment, so unit-test runs (CI, dev) stay
network-free. When run with credentials present, it instantiates a
small claim, runs it through the full superred Controller against a
real chat-completions endpoint, and asserts the output shape.

The point of this test is to catch breakage in the integration
between:
  - the CSV loader and bundled data,
  - the HarmBenchTask + judge pipeline,
  - the live litellm.acompletion path,
  - and the Controller / ChatbotTarget wiring.

It does NOT assert specific ASR values (those depend on the live
target's behavior on any given day). It asserts the run completes,
returns the right number of task results, and each result has the
shape we expect.

Run manually:
    pytest tests/test_e2e_integration.py -v -s

Requires:
    LITELLM_API_KEY=...
    LITELLM_API_BASE=...
"""

from __future__ import annotations

import os

import pytest

# Skip the entire module unless credentials are present.
_HAS_CREDS = bool(
    os.environ.get("LITELLM_API_KEY") and os.environ.get("LITELLM_API_BASE")
)
pytestmark = pytest.mark.skipif(
    not _HAS_CREDS,
    reason="set LITELLM_API_KEY and LITELLM_API_BASE to run e2e integration tests",
)


# Local single-shot optimizer for the integration test. Identical to
# superred-experiments/test_harmbench_chatbot/direct_request_optimizer.py
# but vendored here so this test is self-contained.
def _build_direct_request_optimizer():
    from superred.core.interfaces.optimizer import Optimizer
    from superred.core.types.events import (
        ControllableInjection,
        ControllableNoInjection,
        ControllablePostCallEvent,
        ControllablePreCallEvent,
        RunEndEvent,
        RunEndResponse,
        RunStartEvent,
    )
    from superred.core.types.event import EventResponse

    class DirectRequestOptimizer(Optimizer):
        def __init__(self) -> None:
            super().__init__()
            self._goal = None
            self._injected = False

        async def initialize(self, goal, controllables, observables, llm_client):
            await super().initialize(goal, controllables, observables, llm_client)
            self._goal = goal
            self._injected = False

        async def on_event(self, event):
            if isinstance(event, RunStartEvent):
                self._injected = False
                return EventResponse(event=event)
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(
                        event=event, controllable=event.controllable,
                    )
                if self._injected or self._goal is None:
                    return ControllableNoInjection(
                        event=event, controllable=event.controllable,
                    )
                self._injected = True
                return ControllableInjection(
                    event=event, controllable=event.controllable,
                    value=self._goal.description,
                )
            if isinstance(event, ControllablePostCallEvent):
                return ControllableNoInjection(
                    event=event, controllable=event.controllable,
                )
            if isinstance(event, RunEndEvent):
                return RunEndResponse(event=event, done=True)
            return EventResponse(event=event)

        async def teardown(self):
            pass

    return DirectRequestOptimizer


@pytest.mark.asyncio
async def test_e2e_against_real_llm_three_standard_behaviors() -> None:
    """Tiny claim (3 standard behaviors) ran end-to-end against the
    configured LITELLM endpoint. Asserts output shape, not ASR values."""
    from chatbot_target import ChatbotTarget, RESPONSE_READABLE_TAG, USER_TAG  # type: ignore[import-untyped]
    from superred.core.controller import Controller

    from secclaim_harmbench import harmbench_claim

    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_API_BASE"]

    target = ChatbotTarget(
        model="openai/gpt-4o-mini", api_key=api_key, api_base=api_base,
    )

    # Pick a small subset: first 3 standard behaviors from val.
    # Build a claim, then truncate to 3 by passing a custom CSV.
    # Cheaper alternative: use semantic_categories to narrow.
    claim = harmbench_claim(
        judge_model="openai/gpt-4o-mini",
        judge_api_base=api_base,
        judge_api_key=api_key,
        split="val",
        functional_categories=("standard",),
        semantic_categories=("harassment_bullying",),  # 4 rows in val/standard
    )

    # Flatten and trim to 3 to bound LLM cost.
    all_tasks = list(claim)
    assert len(all_tasks) >= 3, "Expected at least 3 tasks in the slice"
    short_tasks = all_tasks[:3]

    from superred.core.interfaces.security_claim import SecurityClaim
    short_claim = SecurityClaim.from_tasks(short_tasks)

    from superred.core.controller import TargetFactory
    DirectRequestOptimizer = _build_direct_request_optimizer()
    controller = Controller(
        optimizer_factory=DirectRequestOptimizer,
        target_factory=TargetFactory.singleton(target),
        security_claim=short_claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
        max_runs_per_task=1,
        include_feedback=False,
    )

    result = await controller.run()

    # Shape assertions.
    assert len(result.task_results) == 3
    for tr in result.task_results:
        # Each TaskResult has runs, primary_score, etc.
        assert len(tr.runs) == 1
        run = tr.runs[0]
        eval_result = run.evaluation
        # Primary score must be 0.0 or 1.0 (binary verdict)
        assert eval_result.primary_score.value in (0.0, 1.0)
        assert eval_result.primary_score.name == "harmbench_asr"
        # Sub-scores must include functional, semantic, and advbench.
        assert "functional_standard" in eval_result.sub_scores
        assert "semantic_harassment_bullying" in eval_result.sub_scores
        assert "advbench_refusal_label" in eval_result.sub_scores
        # AdvBench label is also binary 0/1.
        assert eval_result.sub_scores["advbench_refusal_label"].value in (0.0, 1.0)
        # Rationale carries metadata for inspection.
        assert "behavior_id=" in eval_result.rationale
        assert "functional=standard" in eval_result.rationale
        # Task attributes propagated.
        assert tr.task.functional_category == "standard"  # type: ignore[attr-defined]
        assert tr.task.semantic_category == "harassment_bullying"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2e_copyright_path_no_judge_call() -> None:
    """Run a tiny copyright claim end-to-end. The model will likely
    refuse to reproduce lyrics, so the matcher returns False, and the
    judge LLM is NOT invoked on this path. Verifies the copyright
    pipeline (real .pkl load, real spaCy tokenization, real Jaccard)
    works against a live target.
    """
    from chatbot_target import ChatbotTarget, RESPONSE_READABLE_TAG, USER_TAG  # type: ignore[import-untyped]
    from superred.core.controller import Controller
    from superred.core.interfaces.security_claim import SecurityClaim

    from secclaim_harmbench import harmbench_claim

    api_key = os.environ["LITELLM_API_KEY"]
    api_base = os.environ["LITELLM_API_BASE"]

    target = ChatbotTarget(
        model="openai/gpt-4o-mini", api_key=api_key, api_base=api_base,
    )

    claim = harmbench_claim(
        judge_model="openai/gpt-4o-mini",
        judge_api_base=api_base,
        judge_api_key=api_key,
        split="val",
        functional_categories=("copyright",),
    )
    short_claim = SecurityClaim.from_tasks(list(claim)[:2])  # 2 copyright tasks

    from superred.core.controller import TargetFactory
    DirectRequestOptimizer = _build_direct_request_optimizer()
    controller = Controller(
        optimizer_factory=DirectRequestOptimizer,
        target_factory=TargetFactory.singleton(target),
        security_claim=short_claim,
        scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
        max_runs_per_task=1,
        include_feedback=False,
    )

    result = await controller.run()

    assert len(result.task_results) == 2
    for tr in result.task_results:
        eval_result = tr.runs[0].evaluation
        # The rationale must indicate the hash_check path (not judge).
        assert "hash_check" in eval_result.rationale
        # Score is binary.
        assert eval_result.primary_score.value in (0.0, 1.0)
        assert tr.task.functional_category == "copyright"  # type: ignore[attr-defined]
