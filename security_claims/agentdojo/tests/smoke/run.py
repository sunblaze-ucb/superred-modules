"""End-to-end smoke run against a real LLM.

Runs ONE Layer-1 paired task against ``openai/gpt-4o-2024-05-13`` (or
whatever ``LITELLM_MODEL`` / ``OPENAI_MODEL`` is set to) via a litellm
proxy or the OpenAI API.  A passthrough optimizer is used, so the
result is the no-attacker baseline (security=False).  The script is
NOT a pytest test; it's a runnable demonstration that the full
pipeline works against a real model.

Required env vars (any one of these two combinations):

- Direct OpenAI: ``OPENAI_API_KEY``.
- litellm proxy: ``LITELLM_API_KEY`` + ``LITELLM_API_BASE``.

Usage::

    LITELLM_API_KEY=... LITELLM_API_BASE=... \\
      python tests/smoke/run.py

Expected output: one ``TaskResult`` with ``stop_reason='done'`` and
``best_score=0`` (no injection, so no attack succeeded).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agentdojo_target import (
    AgentDojoTarget,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
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

from agentdojo_claim import agentdojo_layer1_claim

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke")


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


def _resolve_pipeline_config() -> tuple[str, str | None, str | None]:
    """Resolve (pipeline_model, api_base, api_key) from environment."""
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return (
            os.environ.get("LITELLM_MODEL", "openai/gpt-4o-2024-05-13"),
            os.environ["LITELLM_API_BASE"],
            os.environ["LITELLM_API_KEY"],
        )
    if "OPENAI_API_KEY" in os.environ:
        return (
            os.environ.get("OPENAI_MODEL", "openai/gpt-4o-2024-05-13"),
            None,
            os.environ["OPENAI_API_KEY"],
        )
    return ("openai/gpt-4o-2024-05-13", None, None)


async def main() -> int:
    model, api_base, api_key = _resolve_pipeline_config()
    if not api_key:
        logger.error(
            "No API credentials in environment.  Set either "
            "OPENAI_API_KEY or LITELLM_API_KEY+LITELLM_API_BASE."
        )
        return 2

    claim = agentdojo_layer1_claim(
        pairs=[("banking", "user_task_1", "injection_task_0")]
    )

    def factory() -> AgentDojoTarget:
        return AgentDojoTarget(
            pipeline_model=model,
            api_base=api_base,
            api_key=api_key,
        )

    controller = Controller(
        optimizer_factory=PassthroughOptimizer,
        target_factory=TargetFactory(create=factory, concurrency=1),
        security_claim=claim,
        scope=frozenset({
            USER_TAG, SYSTEM_PROMPT_TAG, TOOL_CATALOGUE_TAG, TOOLS_TAG,
        }),
        max_runs_per_task=1,
    )
    result = await controller.run()
    for tr in result.task_results:
        logger.info(
            "task %s stop=%s best=%.2f utility_sub=%.2f",
            tr.task.goal.description[:60].replace("\n", " "),
            tr.stop_reason,
            tr.best_score.value,
            tr.best_evaluation.sub_scores.get("utility").value
            if tr.best_evaluation.sub_scores.get("utility") else float("nan"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
