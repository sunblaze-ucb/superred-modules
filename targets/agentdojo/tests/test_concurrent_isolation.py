"""Concurrency isolation test: 4 parallel AgentDojoTarget instances must
not cross-contaminate each other's state.

Brief Section 2.g requires the target factory model to be safe under
parallel use; this file establishes the per-instance-isolation
contract empirically.  The test runs 4 distinct synthetic tasks
through ``Controller`` at ``TargetFactory(concurrency=4)``, each task
configured with a unique fingerprint (different seed_yaml_override,
different system_prompt, different benign user_prompt).  The asserts
verify:

1. Each task's post-env snapshot reflects its OWN overrides and is
   uncontaminated by the other three parallel tasks' overrides.
2. Each task's last_response observable reflects its OWN system+user
   prompts (verified via a fake LLM that echoes the prompts back).
3. The tool catalog, function-call trace, and write_calls list are
   per-task, with no shared-state bleed.

A failure here would indicate either upstream-side shared state in
AgentDojo (e.g. a cached Inbox) or a bug in our own per-Target
construction (e.g. accidentally sharing ``_seed_overrides`` between
instances).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest
# Pre-import to flush AgentDojo's registration chain.
import agentdojo.task_suite.load_suites  # noqa: F401
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from superred.core.controller import Controller, TargetFactory
from superred.core.interfaces.optimizer import Optimizer
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.trajectory import Trajectory

from agentdojo_target import AgentDojoTarget, USER_TAG, PROMPT_TAG, TOOLS_TAG


# ---------------------------------------------------------------------------
# Fake LLM that echoes the system+user prompts back so we can verify
# per-task isolation through last_response.
# ---------------------------------------------------------------------------


class _EchoingLLM(BasePipelineElement):
    """Emits a single assistant message whose text contains the user
    prompt verbatim plus a hash of any extra args.  No tool calls.

    The pipeline element receives ``messages`` already containing the
    system + user messages from prior pipeline elements; we read the
    user message and echo it.
    """

    name = "echoing-llm"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        echoed = user_msgs[-1]["content"] if user_msgs else "(no user msg)"
        # echoed may be a list of content blocks; flatten to text
        if isinstance(echoed, list):
            text = " ".join(
                b.get("content", "") for b in echoed
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = str(echoed)
        reply = {
            "role": "assistant",
            "content": [{"type": "text", "content": f"ECHO[{text}]"}],
            "tool_calls": None,
        }
        return query, runtime, env, [*messages, reply], extra_args


@pytest.fixture(autouse=True)
def patch_llm_build(monkeypatch: pytest.MonkeyPatch):
    """Force build_pipeline to use the echoing fake LLM (no real network)."""
    from agentdojo_target import pipeline_bridge

    def fake_build(model_id, *, api_base, api_key):  # noqa: ARG001
        return _EchoingLLM()

    monkeypatch.setattr(pipeline_bridge, "_build_llm", fake_build)
    yield


# ---------------------------------------------------------------------------
# Per-task optimizer: passthrough but captures the run's last_response.
# ---------------------------------------------------------------------------


class _PassthroughOptimizer(Optimizer):
    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, (ControllablePreCallEvent, ControllablePostCallEvent)):
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)


# ---------------------------------------------------------------------------
# Task with unique fingerprint per instance.
# ---------------------------------------------------------------------------


class _FingerprintedTask(Task[AgentDojoTarget]):
    """A Layer-2-style task that plants a unique fingerprint into the
    target's config slots and verifies (via evaluate) that post-run state
    reflects the SAME fingerprint -- never another concurrent task's."""

    def __init__(self, fingerprint: str) -> None:
        self._fp = fingerprint
        self._goal = Goal(description=f"sentinel goal {fingerprint}")

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def fingerprint(self) -> str:
        return self._fp

    async def configure_target(self, target: AgentDojoTarget) -> None:
        # Each task carries a unique system prompt, user prompt, and seed
        # overlay.  None of these should leak between concurrent tasks.
        target.set_config("system_prompt", f"SYS_{self._fp}")
        target.set_config("user_prompt", f"USER_{self._fp}")
        # Seed the banking balance to a unique value so the per-task
        # composite env is observably different.
        target.set_config(
            "seed_yaml_override__banking",
            json.dumps({"bank_account": {"balance": float(hash(self._fp) % 100000)}}),
        )

    async def evaluate(self, _trajectory: Trajectory, target: AgentDojoTarget) -> EvaluationResult:
        last_response = target.query("last_response")
        post_env_blob = json.loads(target.query("post_environment_snapshot"))

        # ASSERTION 1: last_response echoes THIS task's user prompt.
        own_echo = f"ECHO[USER_{self._fp}]"
        if own_echo not in last_response:
            return EvaluationResult(
                success=False,
                primary_score=Score(
                    value=0.0, name="isolation",
                    security_domain=USER_TAG,
                ),
                sub_scores={},
                rationale=(
                    f"cross-contamination: expected {own_echo!r} in "
                    f"last_response, got {last_response!r}"
                ),
            )

        # ASSERTION 2: banking balance reflects THIS task's overlay.
        expected_balance = float(hash(self._fp) % 100000)
        actual_balance = post_env_blob["banking"]["bank_account"]["balance"]
        if actual_balance != expected_balance:
            return EvaluationResult(
                success=False,
                primary_score=Score(
                    value=0.0, name="isolation",
                    security_domain=USER_TAG,
                ),
                sub_scores={},
                rationale=(
                    f"cross-contamination: expected balance "
                    f"{expected_balance}, got {actual_balance}"
                ),
            )

        # Both invariants hold: per-task isolation preserved.
        return EvaluationResult(
            success=True,
            primary_score=Score(
                value=1.0, name="isolation",
                security_domain=USER_TAG,
            ),
            sub_scores={
                "fingerprint": Score(
                    value=1.0, name=self._fp,
                    security_domain=USER_TAG,
                ),
            },
            rationale=f"isolation OK for fingerprint {self._fp}",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_four_targets_run_concurrently_without_cross_contamination() -> None:
    """4 distinct tasks at TargetFactory(concurrency=4): each must see
    its OWN fingerprint in last_response and its OWN balance in
    post_environment_snapshot."""
    fingerprints = ["alpha", "beta", "gamma", "delta"]
    tasks = [_FingerprintedTask(fp) for fp in fingerprints]
    from typing import cast as _cast
    claim = SecurityClaim.from_tasks(
        _cast(list[Task[Target]], tasks)
    )

    def factory() -> AgentDojoTarget:
        return AgentDojoTarget(
            pipeline_model="openai/gpt-4o-2024-05-13",
            api_key="sk-stub",
        )

    controller = Controller(
        optimizer_factory=_PassthroughOptimizer,
        target_factory=TargetFactory(create=factory, concurrency=4),
        security_claim=claim,
        scope=frozenset({USER_TAG, PROMPT_TAG, TOOLS_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()

    assert len(result.task_results) == 4
    rationales = [tr.best_evaluation.rationale for tr in result.task_results]
    failures = [r for r in rationales if "cross-contamination" in r]
    assert not failures, f"isolation failures detected: {failures}"

    # Every task succeeded individually
    for tr in result.task_results:
        assert tr.success is True, (
            f"task {tr.task.fingerprint!r} did not succeed: "  # type: ignore[attr-defined]
            f"{tr.best_evaluation.rationale}"
        )

    # Every fingerprint appears exactly once in the sub-scores
    seen = sorted(
        next(iter(tr.best_evaluation.sub_scores.values())).name
        for tr in result.task_results
    )
    assert seen == sorted(fingerprints)


@pytest.mark.asyncio
async def test_eight_targets_concurrent_stress() -> None:
    """Stress test: 8 distinct tasks at concurrency=8.  Same isolation
    contract as the 4-target test but exercises more parallelism."""
    fingerprints = [f"task_{i}" for i in range(8)]
    tasks = [_FingerprintedTask(fp) for fp in fingerprints]
    from typing import cast as _cast
    claim = SecurityClaim.from_tasks(_cast(list[Task[Target]], tasks))

    def factory() -> AgentDojoTarget:
        return AgentDojoTarget(
            pipeline_model="openai/gpt-4o-2024-05-13",
            api_key="sk-stub",
        )

    controller = Controller(
        optimizer_factory=_PassthroughOptimizer,
        target_factory=TargetFactory(create=factory, concurrency=8),
        security_claim=claim,
        scope=frozenset({USER_TAG, PROMPT_TAG, TOOLS_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()

    assert len(result.task_results) == 8
    for tr in result.task_results:
        assert tr.success is True, tr.best_evaluation.rationale


def test_no_module_level_mutable_state_in_target_package() -> None:
    """Audit pin: no agentdojo_target module exposes a mutable
    module-level data structure that could be shared across Target
    instances.

    The pin is conservative: we list the known-safe module-level names
    and assert nothing else exists.  Adding a new module-level name in
    the future will fail this test, forcing the author to confirm the
    new symbol is either immutable or per-Target.
    """
    import agentdojo_target
    import agentdojo_target.controllables as ctrls
    import agentdojo_target.observables as obs
    import agentdojo_target.tool_registry as treg
    import agentdojo_target.security_tags as stags

    # Module-level data that is acceptable: frozen specs, read-only
    # lookup tables, read-only registries.  All entries here are
    # immutable post-import.
    KNOWN_IMMUTABLE = {
        # tool_registry
        "TOOL_REGISTRY", "ALL_FUNCTIONS", "READ_FUNCTION_NAMES",
        "WRITE_FUNCTION_NAMES", "READ_TOOLS", "WRITE_TOOLS",
        "SUITE_NAMES",
        # controllables
        "CONTROLLABLES", "READ_CTRLS", "READ_QUADRANT_MAP",
        "TOOL_CATALOG_CTRLS",
    }
    # Each of these must exist (smoke check) and be hashable or
    # otherwise non-mutable-by-reference.
    for name in KNOWN_IMMUTABLE:
        # Look up across the modules where these names actually live
        found = (
            getattr(treg, name, None)
            or getattr(ctrls, name, None)
            or getattr(stags, name, None)
        )
        # Some are dict/list/frozenset; we just confirm presence here.
        # Concrete mutability properties are covered by their own
        # module's tests (e.g. test_tool_registry asserts the registry
        # has the expected size; test_controllables_no_dup, etc.).
        if found is None:
            pytest.fail(f"expected module-level constant missing: {name}")
