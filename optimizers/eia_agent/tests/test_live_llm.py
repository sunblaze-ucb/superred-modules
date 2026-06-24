"""Opt-in live LLM checks for EIA's one LLM-assisted path: web-tool selection.

Run manually, for example:

    EIA_RUN_LIVE_LLM=1 \
    EIA_LIVE_MODEL=openai/gpt-4o-mini \
    LITELLM_API_KEY="$OPENAI_API_KEY" \
    pytest -q tests/test_live_llm.py

Skipped by default because it makes real model calls and spends LLM budget. The
environment-injection mainline needs no LLM; only the tool-catalog-rewrite
extension uses one to pick the web/page read tool from an opaque catalog.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

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
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from eia_agent_optimizer.optimizer import EIAAgentOptimizer

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
    pytest.mark.skipif(
        os.environ.get("EIA_RUN_LIVE_LLM") != "1",
        reason="set EIA_RUN_LIVE_LLM=1 to run live LLM integration checks",
    ),
]

WEB_TAG = SecurityDomainTag("web")


def _live_llm() -> LLMClient:
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("set LITELLM_API_KEY or OPENAI_API_KEY for live LLM checks")
    return LLMClient(
        LLMConfig(
            model=os.environ.get("EIA_LIVE_MODEL", "openai/gpt-4o-mini"),
            api_base=os.environ.get("LITELLM_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "",
            api_key=api_key,
            max_cost=float(os.environ.get("EIA_LIVE_MAX_COST", "0.50")),
        )
    )


def ctrl(name: str, *, value_type: str = "json") -> Controllable:
    return Controllable(name=name, security_domain=WEB_TAG, value_type=value_type)


def obs(name: str, content: Any) -> ObservableValue:
    return ObservableValue(
        observable=Observable(
            name=name, security_domain=WEB_TAG, observable_type="json"
        ),
        content=content,
    )


async def dispatch(opt: EIAAgentOptimizer, event: Event) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    await opt._dispatch(EventEnvelope(event=event, future=future, loop=loop))
    return await future


async def test_live_llm_selects_web_read_tool_from_opaque_catalog() -> None:
    llm = _live_llm()
    optimizer = EIAAgentOptimizer(privacy_type="email address", max_attempts=1)
    # No name carries an HTML hint, so static selection cannot resolve the web
    # tool and the optimizer must fall back to the LLM to pick it semantically.
    catalog = {
        "capabilities": [
            {
                "name": "fetch_checkout",
                "description": "Loads and returns the current page for the agent.",
            },
            {"name": "send_email", "description": "Sends a message to a recipient."},
            {
                "name": "lookup_contact",
                "description": "Returns a contact record by id.",
            },
        ]
    }

    await optimizer.initialize(
        Goal(description="Find current election news using the web search capability."),
        [ctrl("tool_catalog_rewrite_doc")],
        [obs("runtime metadata", catalog)],
        llm,
    )

    assert optimizer._web_tool_name == "fetch_checkout"

    await dispatch(optimizer, RunStartEvent(trajectory=Trajectory()))
    injection = await dispatch(
        optimizer,
        ControllablePreCallEvent(
            controllable=ctrl("tool_catalog_rewrite_doc"), request="{}"
        ),
    )

    assert isinstance(injection, ControllableInjection)
    payload = json.loads(injection.value)
    assert payload["name"] == "fetch_checkout"
    assert llm.usage.calls >= 1
