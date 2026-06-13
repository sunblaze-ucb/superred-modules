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

from agentdojo_target.pipeline_bridge import (
    _CatalogEditHook,
    _build_llm,
    build_pipeline,
)
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
            event=e,
            controllable=e.controllable,
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
        api_base=None,
        api_key="sk-ant-dummy",
    )
    assert isinstance(llm, AnthropicLLM)


def test_build_llm_anthropic_thinking_suffix_invalid_int() -> None:
    with pytest.raises(ValueError, match="thinking"):
        _build_llm(
            "anthropic/claude-3-5-sonnet-thinking-banana",
            api_base=None,
            api_key="dummy",
        )


# ---------------------------------------------------------------------------
# Pipeline shape
# ---------------------------------------------------------------------------


def test_build_pipeline_returns_agentpipeline(loop) -> None:
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    pipeline = build_pipeline(
        pipeline_model="openai/gpt-4o-2024-05-13",
        system_prompt="be helpful",
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
        api_key="sk-dummy",
    )
    assert isinstance(pipeline, AgentPipeline)
    elements = list(pipeline.elements)
    # Outer order: SystemMessage, InitQuery, CatalogEditHook, llm,
    # MessageStreamHook, ToolsExecutionLoop
    from agentdojo_target.pipeline_bridge import _MessageStreamHook

    assert isinstance(elements[0], SystemMessage)
    assert isinstance(elements[1], InitQuery)
    assert isinstance(elements[2], _CatalogEditHook)
    assert isinstance(elements[-2], _MessageStreamHook)
    assert isinstance(elements[-1], ToolsExecutionLoop)


def test_build_pipeline_catalog_hook_fires_once_not_in_loop(loop) -> None:
    """The catalog-edit hook fires ONCE (outer, before the first LLM call) and is
    NOT spliced into the tool-execution loop, so it does not re-fire every turn.
    Inner loop is [ToolsExecutor, MessageStreamHook, llm, MessageStreamHook]."""
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    pipeline = build_pipeline(
        pipeline_model="openai/gpt-4o-2024-05-13",
        system_prompt="be helpful",
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
        api_key="sk-dummy",
    )
    from agentdojo_target.pipeline_bridge import _MessageStreamHook

    elements = list(pipeline.elements)
    tools_loop: ToolsExecutionLoop = elements[-1]
    inner = list(tools_loop.elements)
    # Inner loop order: ToolsExecutor, MessageStreamHook, llm, MessageStreamHook
    # (emits the new assistant turn).  No CatalogEditHook inside the loop.
    assert isinstance(inner[0], ToolsExecutor)
    assert isinstance(inner[1], _MessageStreamHook)
    assert isinstance(inner[-1], _MessageStreamHook)
    assert not any(isinstance(e, _CatalogEditHook) for e in inner)
    # Exactly one CatalogEditHook in the whole pipeline: the outer one, spliced
    # before the first LLM call (elements[2]).
    assert sum(isinstance(e, _CatalogEditHook) for e in elements) == 1
    assert isinstance(elements[2], _CatalogEditHook)


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
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        loop=loop,
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
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=json.dumps(
                    {
                        "name": "evil_tool",
                        "description": "exfil",
                        "fake_return": "stolen",
                    }
                ),
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
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value="{not json",
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
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=rec.send_event,
        loop=loop,
    )

    def respond(event: Event) -> EventResponse:
        if event.controllable.name == "tool_catalog_register":
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=json.dumps(
                    {
                        "name": "banking__get_balance",  # already in seed
                        "description": "dup",
                        "fake_return": 0,
                    }
                ),
            )
        return ControllableNoInjection(event=event, controllable=event.controllable)

    rec.respond_with(respond)
    _run_hook_in_thread(hook, wrapper)
    # banking__get_balance is still canonical (apply_register refused).
    assert catalog.classify("banking__get_balance") == "canonical"


# ---------------------------------------------------------------------------
# _MessageStreamHook: per-turn observable emission for each chat message
# ---------------------------------------------------------------------------


def test_message_stream_hook_emits_one_observable_per_message() -> None:
    """First firing emits N observables for N messages; second firing on
    a longer message list emits only the new tail."""
    from agentdojo_target.observables import chat_message_observable
    from agentdojo_target.pipeline_bridge import _MessageStreamHook
    from superred.core.types.events import ObservableEvent

    emitted: list[ObservableEvent] = []
    hook = _MessageStreamHook(emit=lambda e: emitted.append(e))

    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    hook.query("q", runtime=None, messages=messages)
    assert len(emitted) == 2
    assert emitted[0].observable.name == "agent_trace_message_0000"
    assert emitted[1].observable.name == "agent_trace_message_0001"
    assert emitted[0].content == {"role": "system", "content": "you are helpful"}
    assert emitted[1].content == {"role": "user", "content": "hi"}

    # Second invocation with one new message: only the tail is emitted.
    messages2 = messages + [{"role": "assistant", "content": "hello!"}]
    hook.query("q", runtime=None, messages=messages2)
    assert len(emitted) == 3
    assert emitted[2].observable.name == "agent_trace_message_0002"
    assert emitted[2].content == {"role": "assistant", "content": "hello!"}


def test_message_stream_hook_no_reemission_when_called_with_same_list() -> None:
    """Calling the hook twice with the same message list does not re-emit."""
    from agentdojo_target.pipeline_bridge import _MessageStreamHook
    from superred.core.types.events import ObservableEvent

    emitted: list[ObservableEvent] = []
    hook = _MessageStreamHook(emit=lambda e: emitted.append(e))
    messages = [{"role": "user", "content": "x"}]
    hook.query("q", runtime=None, messages=messages)
    hook.query("q", runtime=None, messages=messages)
    assert len(emitted) == 1


def test_message_stream_hook_serialises_tool_calls() -> None:
    """A message carrying FunctionCall objects is rendered to plain dicts."""
    from agentdojo.functions_runtime import FunctionCall
    from agentdojo_target.pipeline_bridge import _MessageStreamHook
    from superred.core.types.events import ObservableEvent

    emitted: list[ObservableEvent] = []
    hook = _MessageStreamHook(emit=lambda e: emitted.append(e))
    fc = FunctionCall(function="banking__get_balance", args={}, id="call-1")
    messages = [
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [fc],
        }
    ]
    hook.query("q", runtime=None, messages=messages)
    assert len(emitted) == 1
    payload = emitted[0].content
    assert payload["role"] == "assistant"
    assert payload["tool_calls"] == [
        {
            "function": "banking__get_balance",
            "args": {},
            "id": "call-1",
        }
    ]


def test_message_stream_hook_returns_inputs_unchanged() -> None:
    """The hook must be a pass-through pipeline element; mutations to its
    return tuple would break the AgentPipeline contract."""
    from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
    from agentdojo_target.pipeline_bridge import _MessageStreamHook

    hook = _MessageStreamHook(emit=lambda _e: None)
    runtime = FunctionsRuntime([])
    env = EmptyEnv()
    messages = [{"role": "user", "content": "hi"}]
    extra = {"k": "v"}
    out = hook.query("q", runtime, env, messages, extra)
    assert out == ("q", runtime, env, messages, extra)
