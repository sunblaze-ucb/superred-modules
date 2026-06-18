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
        self.emitted: list[Event] = []
        self._response_fn: Any = lambda e: ControllableNoInjection(
            event=e,
            controllable=e.controllable,
        )

    async def send_event(self, event: Event) -> EventResponse:
        self.events.append(event)
        return self._response_fn(event)

    def emit(self, event: Event) -> None:
        """Collect one-way ObservableEvents the element fires fire-and-forget."""
        self.emitted.append(event)

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

    element, close = _build_llm(
        "openai/gpt-4o-2024-05-13", api_base=None, api_key="sk-dummy"
    )
    assert isinstance(element, OpenAILLM)
    # ``close`` releases the provider client; calling it must not raise.
    assert callable(close)
    close()


def test_build_llm_anthropic_thinking_suffix_parsed() -> None:
    from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM

    element, close = _build_llm(
        "anthropic/claude-3-5-sonnet-20241022-thinking-1024",
        api_base=None,
        api_key="sk-ant-dummy",
    )
    assert isinstance(element, AnthropicLLM)
    assert callable(close)
    close()


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
    pipeline, _close = build_pipeline(
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
    pipeline, _close = build_pipeline(
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


def test_build_pipeline_returns_callable_close(loop) -> None:
    """build_pipeline returns a (pipeline, close) pair; close releases the
    provider client and must be callable without raising."""
    catalog = ToolCatalog.from_seed(ALL_FUNCTIONS)
    rec = _Rec()
    wrapper = WrappedFunctionsRuntime(
        catalog=catalog,
        send_event=rec.send_event,
        emit=rec.emit,
        loop=loop,
    )
    pipeline, close = build_pipeline(
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
    assert callable(close)
    close()


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
# Catalog edit hook: per-edit outcome observable emission
# ---------------------------------------------------------------------------


def _catalog_outcomes(rec: _Rec) -> list[Any]:
    """The catalog_edit_*_outcome ObservableEvents the hook emitted."""
    return [
        e
        for e in rec.emitted
        if e.observable.name.startswith("catalog_edit_")
        and e.observable.name.endswith("_outcome")
    ]


def test_hook_emits_applied_outcome_for_valid_register(loop) -> None:
    """A valid register injection emits one catalog_edit_register_outcome
    observable with status 'applied', tagged at the add capability."""
    from agentdojo_target.security_tags import TOOL_CATALOGUE_ADD_TAG

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
        emit=rec.emit,
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

    outcomes = _catalog_outcomes(rec)
    # Only the injected register slot produces an outcome; the three declined
    # slots emit nothing.
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.observable.name == "catalog_edit_register_outcome"
    assert outcome.observable.security_domain == TOOL_CATALOGUE_ADD_TAG
    assert outcome.content["operation"] == "register"
    assert outcome.content["status"] == "applied"
    assert outcome.content["detail"] is None


def test_hook_emits_rejected_outcome_for_malformed_register(loop) -> None:
    """A malformed (non-JSON) register payload emits a register outcome with
    status 'rejected' and a non-empty detail."""
    from agentdojo_target.security_tags import TOOL_CATALOGUE_ADD_TAG

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
        emit=rec.emit,
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

    outcomes = _catalog_outcomes(rec)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.observable.name == "catalog_edit_register_outcome"
    assert outcome.observable.security_domain == TOOL_CATALOGUE_ADD_TAG
    assert outcome.content["operation"] == "register"
    assert outcome.content["status"] == "rejected"
    assert outcome.content["detail"]  # non-empty reason


def test_hook_emits_rejected_outcome_for_duplicate_register(loop) -> None:
    """A duplicate-register (ValueError inside apply) emits a register outcome
    with status 'rejected'."""
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
        emit=rec.emit,
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

    outcomes = _catalog_outcomes(rec)
    assert len(outcomes) == 1
    assert outcomes[0].observable.name == "catalog_edit_register_outcome"
    assert outcomes[0].content["status"] == "rejected"


# ---------------------------------------------------------------------------
# _CompatibleOpenAILLM: discarded-action observable for parallel-wrapper names
# ---------------------------------------------------------------------------


def test_compatible_openai_llm_emits_discarded_action(monkeypatch) -> None:
    """When the assistant response carries a synthetic
    ``multi_tool_use.parallel`` tool-call, the LLM element drops the
    malformed name, emits one discarded_action observable, and the
    sanitised tool-call set is exactly the expanded inner calls (sanitise
    behaviour unchanged)."""
    import openai
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.functions_runtime import EmptyEnv, FunctionCall, FunctionsRuntime
    from superred.core.types.events import ObservableEvent

    from agentdojo_target.pipeline_bridge import _CompatibleOpenAILLM

    # A synthetic parallel-wrapper call: invalid outer name "multi_tool_use.parallel"
    # whose args.tool_uses expand into two valid inner calls.
    wrapper_call = FunctionCall(
        function="multi_tool_use.parallel",
        args={
            "tool_uses": [
                {"recipient_name": "banking__get_balance", "parameters": {}},
                {"recipient_name": "banking__get_iban", "parameters": {"x": 1}},
            ]
        },
        id="call-wrap",
    )
    assistant_msg = {
        "role": "assistant",
        "content": "running tools",
        "tool_calls": [wrapper_call],
    }

    def fake_super_query(self, query, runtime, env, messages, extra_args):  # noqa: ANN001
        # Upstream returns the message list with the assistant turn appended.
        return query, runtime, env, [assistant_msg], extra_args

    monkeypatch.setattr(OpenAILLM, "query", fake_super_query)

    emitted: list[ObservableEvent] = []
    client = openai.OpenAI(api_key="sk-dummy")
    llm = _CompatibleOpenAILLM(
        client, "gpt-4-turbo-2024-04-09", emit=lambda e: emitted.append(e)
    )

    runtime = FunctionsRuntime([])
    env = EmptyEnv()
    out_query, _rt, _env, out_messages, _extra = llm.query("q", runtime, env, [], {})

    # Exactly one discarded_action observable was emitted for the dropped name.
    assert len(emitted) == 1
    assert emitted[0].observable.name == "discarded_action_0000"
    assert emitted[0].content["dropped_names"] == ["multi_tool_use.parallel"]
    assert emitted[0].content["dropped_count"] == 1

    # Sanitise behaviour is unchanged: the synthetic wrapper is expanded into
    # one FunctionCall per inner tool_use; the malformed outer name is gone.
    sanitised = out_messages[-1]["tool_calls"]
    names = [tc.function for tc in sanitised]
    assert names == ["banking__get_balance", "banking__get_iban"]
    assert all(tc.id == "call-wrap" for tc in sanitised)
    assert sanitised[1].args == {"x": 1}

    client.close()


# ---------------------------------------------------------------------------
# _MessageStreamHook: per-turn observable emission for each chat message
# ---------------------------------------------------------------------------


def test_message_stream_hook_emits_one_observable_per_message() -> None:
    """First firing emits N observables for N messages; second firing on
    a longer message list emits only the new tail."""
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


def test_message_stream_hook_strips_tool_calls_from_assistant() -> None:
    """An assistant message's ``tool_calls`` is stripped: the call lives once
    on the per-tool ControllablePostCallEvent, not on the agent-trace message
    stream.  The non-tool reasoning content is still emitted."""
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
    assert payload["content"] == "checking"
    # The tool-call request is NOT mirrored on the agent trace.
    assert "tool_calls" not in payload


def test_message_stream_hook_skips_tool_result_messages() -> None:
    """Tool-result messages (role == 'tool') are skipped entirely: the tool's
    return lives once on the per-tool ControllablePostCallEvent.  The message
    index still advances so non-tool messages keep their position."""
    from agentdojo_target.pipeline_bridge import _MessageStreamHook
    from superred.core.types.events import ObservableEvent

    emitted: list[ObservableEvent] = []
    hook = _MessageStreamHook(emit=lambda e: emitted.append(e))
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "1234.0", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "done"},
    ]
    hook.query("q", runtime=None, messages=messages)
    # Only the user and assistant messages are emitted; the tool message is
    # skipped.  The index still advances over the skipped message.
    names = [e.observable.name for e in emitted]
    assert names == ["agent_trace_message_0000", "agent_trace_message_0002"]
    assert [e.content["role"] for e in emitted] == ["user", "assistant"]


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
