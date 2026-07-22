"""End-to-end smoke run for the AgentHarm claim against a real model.

Runs a small AgentHarm claim through Controller + InspectAgentTarget +
PassthroughOptimizer (the direct-request baseline) and prints per-task
harm_score / refusal plus the aggregate() summary.  NOT a pytest test; a
runnable demonstration that the full stack works and reproduces sane numbers.

Credentials (one combination):
- litellm proxy: LITELLM_API_KEY + LITELLM_API_BASE
- direct OpenAI: OPENAI_API_KEY

Optional: LITELLM_MODEL / OPENAI_MODEL (target model; default openai/gpt-4o-mini),
AGENTHARM_BEHAVIOR_IDS (comma-separated; default "1-1").

Usage::

    LITELLM_API_KEY=... LITELLM_API_BASE=... python tests/smoke/run.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from inspect_agent_target import (
    SYSTEM_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from superred.core.controller import Controller
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

from agentharm_claim import (
    agentharm_claim,
    agentharm_target_factory,
    aggregate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agentharm-smoke")


class PassthroughOptimizer(Optimizer):
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


def _resolve_config() -> tuple[str, str | None, str | None]:
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return (
            os.environ.get("LITELLM_MODEL", "openai/gpt-4o-mini"),
            os.environ["LITELLM_API_BASE"],
            os.environ["LITELLM_API_KEY"],
        )
    if "OPENAI_API_KEY" in os.environ:
        return (
            os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"),
            None,
            os.environ["OPENAI_API_KEY"],
        )
    return ("openai/gpt-4o-mini", None, None)


async def main() -> int:
    model, api_base, api_key = _resolve_config()
    if not api_key:
        logger.error("No credentials: set OPENAI_API_KEY or LITELLM_API_KEY+LITELLM_API_BASE.")
        return 2

    behavior_ids = os.environ.get("AGENTHARM_BEHAVIOR_IDS", "1-1").split(",")
    claim = agentharm_claim(
        target_model=model, api_base=api_base, api_key=api_key, behavior_ids=behavior_ids
    )
    controller = Controller(
        optimizer_factory=PassthroughOptimizer,
        target_factory=agentharm_target_factory(
            target_model=model, api_base=api_base, api_key=api_key, concurrency=2
        ),
        security_claim=claim,
        scope=frozenset({USER_TAG, SYSTEM_TAG, TOOLS_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()

    evals = []
    for tr in result.task_results:
        ev = tr.best_evaluation
        evals.append(ev)
        logger.info(
            "behavior=%s stop=%s harm_score=%.3f refusal=%s success=%s",
            tr.task.behavior_id,
            tr.stop_reason,
            tr.best_score.value,
            ev.sub_scores["refusal"].value if ev else "n/a",
            tr.success,
        )
    logger.info("aggregate: %s", aggregate(evals))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
