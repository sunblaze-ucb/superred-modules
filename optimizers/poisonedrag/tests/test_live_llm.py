"""Opt-in live LLM checks for PoisonedRAG's LLM-assisted paths.

Run manually, for example:

    POISONEDRAG_RUN_LIVE_LLM=1 \
    POISONEDRAG_LIVE_MODEL=openai/gpt-4o-mini \
    LITELLM_API_KEY="$OPENAI_API_KEY" \
    pytest -q tests/test_live_llm.py

Skipped by default because it makes real model calls and spends LLM budget.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from superred.core.channel import EventEnvelope
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.llm import LLMConfig
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from poisonedrag_optimizer.optimizer import PoisonedRAGOptimizer

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
    pytest.mark.skipif(
        os.environ.get("POISONEDRAG_RUN_LIVE_LLM") != "1",
        reason="set POISONEDRAG_RUN_LIVE_LLM=1 to run live LLM integration checks",
    ),
]

RAG_TAG = SecurityDomainTag("rag")


def _live_llm() -> LLMClient:
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("set LITELLM_API_KEY or OPENAI_API_KEY for live LLM checks")
    return LLMClient(
        LLMConfig(
            model=os.environ.get("POISONEDRAG_LIVE_MODEL", "openai/gpt-4o-mini"),
            api_base=os.environ.get("LITELLM_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "",
            api_key=api_key,
        ),
        cost_cap_usd=float(os.environ.get("POISONEDRAG_LIVE_MAX_COST", "0.75")),
    )


def ctrl(name: str, *, description: str = "", value_type: str = "json") -> Controllable:
    return Controllable(name, RAG_TAG, description=description, value_type=value_type)


async def dispatch(opt: PoisonedRAGOptimizer, event: Event) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    await opt._dispatch(EventEnvelope(event=event, future=future, loop=loop))
    return await future


async def test_live_llm_selects_surface_and_generates_poison_batch() -> None:
    llm = _live_llm()
    optimizer = PoisonedRAGOptimizer(
        correct_answer="Paris",
        adv_per_query=2,
        max_attempts=1,
        generation_max_tokens=600,
    )
    surface = ctrl(
        "tenant_memory_slot",
        description="Writable storage that the retrieval layer reads before answering.",
        value_type="json",
    )

    await optimizer.initialize(
        Goal(description="What is the capital of France?"),
        [surface],
        [],
        llm,
    )
    # The model may bucket this as a corpus or a retrieved-context surface; both
    # are doc-carrying surfaces and deliver identically, so accept either.
    assert "tenant_memory_slot" in (
        optimizer._llm_corpus_surface_names | optimizer._llm_context_surface_names
    )

    await dispatch(optimizer, RunStartEvent(trajectory=Trajectory()))
    response = await dispatch(
        optimizer,
        ControllablePreCallEvent(
            controllable=surface, request=json.dumps({"documents": []})
        ),
    )

    assert isinstance(response, ControllableInjection)
    payload = json.loads(response.value)
    assert isinstance(payload["documents"], list)
    assert len(payload["documents"]) == 2
    assert optimizer.target_answer
    assert optimizer.target_answer.lower() != "paris"
    assert llm.usage.calls >= 2
