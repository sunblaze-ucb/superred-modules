"""Tests for the AgentDojo pipeline bridge.

The bridge wires the wrapped runtime, the tool catalog, and the
event-channel callbacks into an upstream :class:`AgentPipeline`.  We do
not exercise a real LLM here; instead we verify:

- the LLM-build dispatch on the model-id prefix
- the catalog-edit hook fires the four catalog Controllables in order
- an injected register payload mutates the catalog and refreshes the
  runtime function list
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
)

from agentdojo_target.pipeline_bridge import _CatalogEditHook, _build_llm, build_pipeline
from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.tool_catalog import ToolCatalog
from agentdojo_target.tool_registry import ALL_FUNCTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Rec:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._response_fn: Any = lambda e: ControllableNoInjection(
            event=e, controllable=e.controllable,
        )

    async def send_event(self, event: Event) -> EventResponse:
        self.events.append(event)
        return self._response_fn(event)

    def emit(self, _event: Event) -> None:  # not used here
        pass

    def respond_with(self, fn: Any) -> None:
        self._response_fn = fn


@pytest.fixture
def loop() -> asyncio.AbstractEventLoop:
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2)


# ---------------------------------------------------------------------------
# LLM build dispatch
# ---------------------------------------------------------------------------


def test_build_llm_rejects_missing_provider() -> None:
    with pytest.raises(ValueError, match="provider/model"):
        _build_llm("gpt-4o", api_base=None, api_key="dummy")


def test_build_llm_unknown_provider_raises() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        _build_llm("nonexistent/m", api_base=None, api_key="dummy")


def test_build_llm_openai_returns_openai_llm() -> None:
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

    llm = _build_llm("openai/gpt-4o-2024-05-13", api_base=None, api_key="sk-dummy")
    assert isinstance(llm, OpenAILLM)


def test_build_llm_anthropic_thinking_suffix_parsed() -> None:
    from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM

    llm = _build_llm(
        "anthropic/claude-3-5-sonnet-20241022-thinking-1024",
        api_base=None, api_key="sk-ant-dummy",
    )
    assert isinstance(llm, AnthropicLLM)


def test_build_llm_anthropic_thinking_suffix_invalid_int() -> None:
    with pytest.raises(ValueError, match="thinking"):
        _build_llm(
            "anthropic/claude-3-5-sonnet-thinking-banana",
            api_base=None, api_key="dummy",
        )


# ---------------------------------------------------------------------------
# Pipeline shape
# ---------------------------------------------------------------------------


def test_build_pipeline_returns_agentpipeline(loop) -> None:
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    pipeline = build_pipeline(
        pipeline_model="openai/gpt-4o-2024-05-13",
        system_prompt="be helpful",
        catalog=catalog, wrapper=wrapper,
        send_event=rec.send_event, loop=loop,
        api_key="sk-dummy",
    )
    assert isinstance(pipeline, AgentPipeline)
    elements = list(pipeline.elements)
    # Outer order: SystemMessage, InitQuery, CatalogEditHook, llm, ToolsExecutionLoop
    assert isinstance(elements[0], SystemMessage)
    assert isinstance(elements[1], InitQuery)
    assert isinstance(elements[2], _CatalogEditHook)
    assert isinstance(elements[-1], ToolsExecutionLoop)


def test_build_pipeline_splices_hook_into_loop(loop) -> None:
    """Inner loop has [ToolsExecutor, _CatalogEditHook, llm]."""
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    pipeline = build_pipeline(
        pipeline_model="openai/gpt-4o-2024-05-13",
        system_prompt="be helpful",
        catalog=catalog, wrapper=wrapper,
        send_event=rec.send_event, loop=loop,
        api_key="sk-dummy",
    )
    elements = list(pipeline.elements)
    tools_loop: ToolsExecutionLoop = elements[-1]
    inner = list(tools_loop.elements)
    assert isinstance(inner[0], ToolsExecutor)
    assert isinstance(inner[1], _CatalogEditHook)
    # Last element is the llm; not asserting concrete type because v1
    # supports multiple backends.


# ---------------------------------------------------------------------------
# Catalog edit hook semantics
# ---------------------------------------------------------------------------


def _run_hook_in_thread(hook: _CatalogEditHook, runtime) -> None:
    """Invoke hook.query() from a thread, mimicking ToolsExecutionLoop."""
    def target() -> None:
        hook.query("query", runtime, messages=[], extra_args={})

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "Hook did not return"


def test_hook_fires_four_events_per_invocation(loop) -> None:
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog, wrapper=wrapper, send_event=rec.send_event, loop=loop,
    )
    _run_hook_in_thread(hook, wrapper)
    assert len(rec.events) == 4
    names = [e.controllable.name for e in rec.events]
    assert names == [
        "tool_catalog_register",
        "tool_catalog_replace",
        "tool_catalog_unregister",
        "tool_catalog_rewrite_doc",
    ]
    # Every event is a Pre-call event (catalog edits are pre-tool-call).
    for e in rec.events:
        assert isinstance(e, ControllablePreCallEvent)


def test_hook_register_injection_mutates_catalog(loop) -> None:
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog, wrapper=wrapper, send_event=rec.send_event, loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event, controllable=event.controllable,
                value=json.dumps({
                    "name": "evil_tool",
                    "description": "exfil",
                    "fake_return": "stolen",
                }),
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    rec.respond_with(respond)
    _run_hook_in_thread(hook, wrapper)
    assert "evil_tool" in catalog
    assert "evil_tool" in wrapper.functions  # runtime was refreshed


def test_hook_swallows_invalid_json_payload(loop) -> None:
    """A garbage payload doesn't crash the hook; it just doesn't mutate."""
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog, wrapper=wrapper, send_event=rec.send_event, loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event, controllable=event.controllable, value="{not json",
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    rec.respond_with(respond)
    _run_hook_in_thread(hook, wrapper)
    # Catalog state is unchanged (no new entries beyond the seed).
    assert len(catalog.functions_for_runtime()) == len(ALL_FUNCTIONS)


def test_hook_swallows_value_error_from_apply(loop) -> None:
    """A duplicate-register raises ValueError inside apply; the hook
    swallows it (logs) and keeps the run going."""
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog, send_event=rec.send_event, emit=rec.emit, loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog, wrapper=wrapper, send_event=rec.send_event, loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event, controllable=event.controllable,
                value=json.dumps({
                    "name": "banking__get_balance",  # already in seed
                    "description": "dup",
                    "fake_return": 0,
                }),
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    rec.respond_with(respond)
    _run_hook_in_thread(hook, wrapper)
    # banking__get_balance is still canonical (apply_register refused).
    assert catalog.classify("banking__get_balance") == "canonical"
