"""Opt-in Docker concurrency tests (``TargetFactory.concurrency > 1``).

Requires a reachable Docker daemon. Proves two managed Docker targets can run
in parallel without cross-talk when driven through the Superred Controller.

Run explicitly::

    pytest test_docker_concurrency.py -v
"""

from __future__ import annotations

import pytest

from openclaw_target import SYSTEM_TAG, openclaw_target_factory
from openclaw_target.target import OpenClawTarget, USER_MESSAGE_CTRL
from test_support import docker_daemon_ready, docker_image, echo_stub_upstream_for_host_proxy

from superred.core.controller import Controller
from superred.core.interfaces.optimizer import Optimizer
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue
from superred.core.types.trajectory import Trajectory


pytestmark = pytest.mark.skipif(
    not docker_daemon_ready(),
    reason="Docker daemon unavailable",
)


class _MarkerEchoOptimizer(Optimizer):
    """Send the task goal's marker string as the sole user message."""

    def __init__(self) -> None:
        super().__init__()
        self._marker: str = ""

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: object,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)  # type: ignore[arg-type]
        prefix = "Echo marker "
        desc = goal.description
        self._marker = desc[len(prefix):] if desc.startswith(prefix) else desc

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            if event.controllable is USER_MESSAGE_CTRL:
                return ControllableInjection(
                    event=event, controllable=event.controllable, value=self._marker,
                )
            return ControllableInjection(
                event=event, controllable=event.controllable, value="",
            )
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)


class _MarkerTask(Task[OpenClawTarget]):
    """Succeed when ``last_response`` echoes the task's unique marker."""

    def __init__(self, marker: str) -> None:
        self._marker = marker
        self._goal = Goal(description=f"Echo marker {marker}")

    @property
    def goal(self) -> Goal:
        return self._goal

    async def configure_target(self, target: OpenClawTarget) -> None:
        await target.warmup_static_observables()

    async def evaluate(
        self, trajectory: Trajectory, target: OpenClawTarget,
    ) -> EvaluationResult:
        response = target.query("last_response") or ""
        ok = self._marker in response
        return EvaluationResult(
            success=ok,
            primary_score=Score(value=1.0 if ok else 0.0, name="marker_echo"),
            rationale=response[:300] if ok else f"expected {self._marker!r} in {response!r}",
        )


@pytest.mark.asyncio
async def test_docker_controller_concurrency_two_parallel_tasks() -> None:
    """Two Docker-managed targets run concurrently via ``concurrency=2``."""
    marker_a = "CONCURRENT-MARKER-ALPHA-91"
    marker_b = "CONCURRENT-MARKER-BETA-92"
    image = docker_image()
    async with echo_stub_upstream_for_host_proxy() as stub_url:
        controller = Controller(
            optimizer_factory=lambda: _MarkerEchoOptimizer(),
            target_factory=openclaw_target_factory(
                managed=True,
                managed_runtime="docker",
                model_id="openai/gpt-4o-mini",
                provider_base_url=stub_url,
                provider_api_key="stub-key",
                managed_kwargs={"image": image},
                concurrency=2,
            ),
            security_claim=SecurityClaim.from_tasks([
                _MarkerTask(marker_a),
                _MarkerTask(marker_b),
            ]),
            scope=frozenset({SYSTEM_TAG}),
            max_runs_per_task=1,
            include_feedback=False,
        )

        result = await controller.run()

    assert len(result.task_results) == 2
    assert all(tr.success for tr in result.task_results), [
        (tr.task.goal.description, tr.runs[0].evaluation.rationale if tr.runs else "no runs")
        for tr in result.task_results
        if not tr.success
    ]
    assert marker_a in (result.task_results[0].runs[0].evaluation.rationale or "")
    assert marker_b in (result.task_results[1].runs[0].evaluation.rationale or "")
