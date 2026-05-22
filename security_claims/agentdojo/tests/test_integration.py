"""End-to-end integration test for the AgentDojo SecurityClaim port.

Wires:

- :class:`Controller` (superred)
- :class:`AgentDojoTarget` with a FAKE LLM spliced in via monkeypatch
- A passthrough optimizer that responds ``NoInjection`` to every event
- The Layer-1, Layer-2, and combined claims (one task each, narrow)

Validates the full lifecycle round-trips: phases 1-5 of Target.run fire
the right events, the task evaluate methods produce the expected
no-attacker baseline (security=False, utility depends on the agent),
and the Controller returns a populated :class:`ThreatModelResult`.

No real LLM is used; the FakeLLM returns a fixed assistant turn with
no tool calls so the pipeline terminates immediately.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from agentdojo_target import (
    AgentDojoTarget,
    CONTENT_3P_DATA_3P_TAG,
    PROMPT_TAG,
    TOOL_CATALOGUE_TAG,
    USER_TAG,
)
from superred.core.controller import Controller, TargetFactory
from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
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

from security_claim_agentdojo import (
    agentdojo_combined_claim,
    agentdojo_layer1_claim,
    agentdojo_layer2_claim,
)


# ---------------------------------------------------------------------------
# Fake LLM that returns a fixed assistant turn (no tool calls)
# ---------------------------------------------------------------------------


class _FakeLLM(BasePipelineElement):
    """A pipeline element that returns one assistant message and stops.

    The message has no tool_calls, so ToolsExecutionLoop terminates
    immediately.  Content is fixed text so model_output_from_messages
    extracts a stable value.
    """

    name = "fake-llm"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        new_messages = list(messages) + [
            {
                "role": "assistant",
                "content": [{"type": "text", "content": "I cannot help with that."}],
                "tool_calls": None,
            }
        ]
        return query, runtime, env, new_messages, extra_args


@pytest.fixture(autouse=True)
def patch_llm_build(monkeypatch: pytest.MonkeyPatch):
    """Patch :func:`pipeline_bridge._build_llm` to return the fake LLM."""
    from agentdojo_target import pipeline_bridge

    def fake_build(model_id: str, *, api_base, api_key):  # noqa: ARG001
        return _FakeLLM()

    monkeypatch.setattr(pipeline_bridge, "_build_llm", fake_build)
    yield


# ---------------------------------------------------------------------------
# Passthrough optimizer
# ---------------------------------------------------------------------------


class _PassthroughOptimizer(Optimizer):
    """Optimizer that declines every controllable event (no injections)."""

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        # Stash for inspection in tests if needed.
        self._goal = goal

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, (ControllablePreCallEvent, ControllablePostCallEvent)):
            return ControllableNoInjection(event=event, controllable=event.controllable)
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _wide_scope():
    """Scope that grants the optimizer every relevant capability."""
    return frozenset({
        USER_TAG, PROMPT_TAG, TOOL_CATALOGUE_TAG, CONTENT_3P_DATA_3P_TAG,
    })


def _factory():
    return AgentDojoTarget(
        pipeline_model="openai/gpt-4o-2024-05-13",
        api_key="sk-stub",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_layer1_single_task_end_to_end() -> None:
    """Run one Layer-1 paired task end-to-end through the Controller.

    With a passthrough optimizer the agent does nothing (FakeLLM emits
    no tool calls), so we expect security=False and the task to
    complete cleanly via ``done=True``.
    """
    claim = agentdojo_layer1_claim(
        pairs=[("banking", "user_task_1", "injection_task_0")]
    )
    controller = Controller(
        optimizer_factory=_PassthroughOptimizer,
        target_factory=TargetFactory(create=_factory, concurrency=1),
        security_claim=claim,
        scope=_wide_scope(),
        max_runs_per_task=1,
    )
    result = await controller.run()
    assert len(result.task_results) == 1
    tr = result.task_results[0]
    assert tr.stop_reason == "done"
    assert tr.success is False  # no attack with passthrough optimizer
    assert tr.best_score.value == 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_layer2_single_task_end_to_end() -> None:
    """Same but for a Layer-2 system-violating task."""
    claim = agentdojo_layer2_claim(
        goal_ids=["banking_unauth_transfer_to_attacker"]
    )
    controller = Controller(
        optimizer_factory=_PassthroughOptimizer,
        target_factory=TargetFactory(create=_factory, concurrency=1),
        security_claim=claim,
        scope=_wide_scope(),
        max_runs_per_task=1,
    )
    result = await controller.run()
    assert len(result.task_results) == 1
    tr = result.task_results[0]
    assert tr.stop_reason == "done"
    # Without any injection, the no-attack baseline must produce
    # success=False.
    assert tr.success is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_combined_claim_runs_to_completion() -> None:
    """Run a narrow combined claim (one Layer-1 task + one Layer-2 goal)."""
    # Pass narrow filters to keep the integration test fast.
    layer1_kwargs = {"pairs": [("banking", "user_task_0", "injection_task_7")]}
    layer2_kwargs = {"goal_ids": ["banking_unauth_transfer_to_attacker"]}
    claim = agentdojo_combined_claim(
        layer1_kwargs=layer1_kwargs,
        layer2_kwargs=layer2_kwargs,
    )
    controller = Controller(
        optimizer_factory=_PassthroughOptimizer,
        target_factory=TargetFactory(create=_factory, concurrency=1),
        security_claim=claim,
        scope=_wide_scope(),
        max_runs_per_task=1,
    )
    result = await controller.run()
    assert len(result.task_results) == 2
    for tr in result.task_results:
        assert tr.stop_reason == "done"
        assert tr.best_score.value == 0.0
