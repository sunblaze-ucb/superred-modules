"""AgentPipeline bridge: build the upstream pipeline with our wrapped runtime.

Provides two things:

- :func:`build_pipeline` - constructs an :class:`AgentPipeline` for a
  given litellm-style model id (e.g. ``openai/gpt-4o-2024-05-13``),
  splicing a per-turn :class:`_CatalogEditHook` so the four tool-catalog
  Controllables fire before every LLM call (including the first one).
- :class:`_CatalogEditHook` - a :class:`BasePipelineElement` that fires
  the four catalog Controllables via the sync-to-async bridge, applies
  any returned injections to the :class:`ToolCatalog`, and refreshes the
  :class:`WrappedFunctionsRuntime`'s function registry so the next
  LLM turn sees the updated catalog.

LLM provider support for v1: OpenAI (``openai/...``) and Anthropic
(``anthropic/...``).  Other providers raise :class:`NotImplementedError`
with a clear message; adding them is mechanical (the AgentDojo
``get_llm`` source covers Cohere, Google, Together, vLLM).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

import anthropic
import openai
from pydantic import ValidationError
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)

from agentdojo_target.controllables import (
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
)
from agentdojo_target.observables import chat_message_observable
from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog-edit hook
# ---------------------------------------------------------------------------


class _CatalogEditHook(BasePipelineElement):
    """Pipeline element that fires the four tool-catalog Controllables
    before every LLM turn.

    The hook fires four :class:`ControllablePreCallEvent`s - one per
    catalog operation - in a fixed order (register, replace,
    unregister, rewrite_doc).  For each, if the optimizer responds with
    :class:`ControllableInjection`, the JSON-decoded payload is applied
    to the catalog via the corresponding :meth:`ToolCatalog.apply_*`
    method.  After all four events, the :class:`WrappedFunctionsRuntime`
    is refreshed so the next LLM call sees the updated function list.

    Application errors (malformed payloads, unknown tool names, etc.)
    are logged and *not* propagated; the hook treats the LLM call's
    progress as more important than enforcing payload correctness.
    """

    name = "catalog_edit_hook"

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        wrapper: WrappedFunctionsRuntime,
        send_event: EventResponseHandler,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._catalog = catalog
        self._wrapper = wrapper
        self._send_event = send_event
        self._loop = loop

    _OPTIMIZER_RESPONSE_TIMEOUT_SECONDS: float = 180.0

    def _await_event(self, event: ControllablePreCallEvent) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._send_event(event), self._loop)
        return future.result(timeout=self._OPTIMIZER_RESPONSE_TIMEOUT_SECONDS)

    def _try_apply(self, method: Any, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Catalog injection payload was not valid JSON: %s", exc)
            return
        if not isinstance(payload, dict):
            logger.warning(
                "Catalog injection payload must be a dict, got %s",
                type(payload).__name__,
            )
            return
        try:
            method(payload)
        except (ValueError, ValidationError) as exc:
            logger.warning("Catalog mutation rejected: %s", exc)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        ops = (
            (TOOL_CATALOG_REGISTER_CTRL, self._catalog.apply_register),
            (TOOL_CATALOG_REPLACE_CTRL, self._catalog.apply_replace),
            (TOOL_CATALOG_UNREGISTER_CTRL, self._catalog.apply_unregister),
            (TOOL_CATALOG_REWRITE_DOC_CTRL, self._catalog.apply_rewrite_doc),
        )
        applied_any = False
        for ctrl, apply_method in ops:
            event = ControllablePreCallEvent(controllable=ctrl, request="catalog edit slot")
            response = self._await_event(event)
            if isinstance(response, ControllableInjection):
                self._try_apply(apply_method, response.value)
                applied_any = True
        if applied_any:
            self._wrapper.refresh_functions()
        return query, runtime, env, messages, extra_args


# ---------------------------------------------------------------------------
# Message-stream hook: per-turn observable emission for chat messages
# ---------------------------------------------------------------------------


class _MessageStreamHook(BasePipelineElement):
    """Pipeline element that emits one
    :class:`agent_trace_message_NNNN` observable per new message in
    the conversation stream.

    Brief Section 2.e mandates "the agent's chat-message stream" be
    exposed as an observable.  We splice an instance of this hook
    after every LLM call (both outside the ToolsExecutionLoop for the
    first turn and inside it for subsequent turns) so the optimizer
    sees each new message in real time as the pipeline progresses,
    not only at run end via the ``conversation_history`` snapshot.

    Tracks the highest message index already emitted so re-firing the
    hook on the same conversation does not re-emit.

    Construction is per-run (one instance per :class:`AgentDojoTarget`
    run); the hook holds no shared state across runs.
    """

    name = "message_stream_hook"

    def __init__(self, *, emit: EventHandler) -> None:
        self._emit = emit
        self._next_idx: int = 0

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # Emit any messages we have not seen yet.  We never re-emit;
        # the optimizer's view of the stream is append-only.
        while self._next_idx < len(messages):
            msg = messages[self._next_idx]
            self._emit(
                ObservableEvent(
                    observable=chat_message_observable(self._next_idx),
                    content=_message_to_jsonable(msg),
                )
            )
            self._next_idx += 1
        return query, runtime, env, messages, extra_args


def _message_to_jsonable(msg: Any) -> dict:
    """Convert a ChatMessage (TypedDict possibly carrying pydantic
    FunctionCall objects) into a JSON-friendly dict for the observable
    payload."""
    out: dict[str, Any] = {}
    for key in ("role", "content", "tool_call_id", "error", "name"):
        if key in msg:
            out[key] = msg[key]
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        out["tool_calls"] = [
            (
                {"function": tc.function, "args": dict(tc.args), "id": tc.id}
                if hasattr(tc, "function") else tc
            )
            for tc in tool_calls
        ]
    elif tool_calls is not None:
        out["tool_calls"] = tool_calls
    tool_call = msg.get("tool_call")
    if tool_call is not None and hasattr(tool_call, "function"):
        out["tool_call"] = {
            "function": tool_call.function,
            "args": dict(tool_call.args),
            "id": tool_call.id,
        }
    return out


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_llm(model_id: str, *, api_base: str | None, api_key: str | None) -> BasePipelineElement:
    """Construct an AgentDojo LLM element from a litellm-style model id.

    Supported providers:
    - ``openai/<model>``  -> :class:`OpenAILLM` with an :class:`openai.OpenAI` client.
    - ``anthropic/<model>`` -> :class:`AnthropicLLM` with an
      :class:`anthropic.Anthropic` client; supports the ``-thinking-N``
      suffix in the model name.

    Args:
        model_id: e.g. ``openai/gpt-4o-2024-05-13``.
        api_base: Override base URL (most relevant for litellm-proxy).
        api_key: Override API key (defaults to environment lookup).

    Raises:
        NotImplementedError: If the provider prefix is not openai or
            anthropic.
        ValueError: If the model id does not include a provider prefix.
    """
    if "/" not in model_id:
        raise ValueError(
            f"pipeline_model must be in 'provider/model' form, got {model_id!r}"
        )
    provider, _, model_name = model_id.partition("/")
    if provider == "openai":
        client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        return OpenAILLM(client, model_name)
    if provider == "anthropic":
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=api_base,
        )
        if "-thinking-" in model_name:
            base_model, _, budget = model_name.partition("-thinking-")
            try:
                budget_tokens = int(budget)
            except ValueError as exc:
                raise ValueError(
                    f"Anthropic 'thinking' suffix must be an integer, got {budget!r}"
                ) from exc
            return AnthropicLLM(client, base_model, thinking_budget_tokens=budget_tokens)
        return AnthropicLLM(client, model_name)
    raise NotImplementedError(
        f"Provider {provider!r} not implemented in v1.  Supported: "
        "openai, anthropic.  Extending is mechanical; see "
        "agentdojo.agent_pipeline.agent_pipeline.get_llm for the upstream "
        "dispatch table."
    )


def build_pipeline(
    *,
    pipeline_model: str,
    system_prompt: str,
    catalog: ToolCatalog,
    wrapper: WrappedFunctionsRuntime,
    send_event: EventResponseHandler,
    emit: EventHandler,
    loop: asyncio.AbstractEventLoop,
    api_base: str | None = None,
    api_key: str | None = None,
) -> AgentPipeline:
    """Build the AgentDojo :class:`AgentPipeline` for one run.

    Splices two hooks around every LLM call:

    - :class:`_CatalogEditHook` (BEFORE the LLM): fires the four
      tool-catalog Controllables so attacker-scoped optimizers can
      mutate the catalog at the start of each agent turn.
    - :class:`_MessageStreamHook` (AFTER the LLM and after the
      ToolsExecutor): emits one ``agent_trace_message_NNNN`` observable
      per new message in the conversation stream.

    Otherwise this is the upstream ``no_defense`` baseline.

    The resulting pipeline shape is::

        AgentPipeline([
            SystemMessage(system_prompt),
            InitQuery(),
            CatalogEditHook,             # fires before first LLM turn
            llm,                          # first agent turn
            MessageStreamHook,           # emits system+user+first assistant
            ToolsExecutionLoop([
                ToolsExecutor(formatter),
                MessageStreamHook,        # emits tool-result messages
                CatalogEditHook,          # fires before every subsequent turn
                llm,
                MessageStreamHook,        # emits the assistant turn output
            ]),
        ])

    The same :class:`_MessageStreamHook` instance is reused across the
    splice points so its ``_next_idx`` cursor advances monotonically
    over the whole conversation -- emitting each message exactly once.
    Same for the :class:`_CatalogEditHook` (one instance, fires at
    each splice point).

    The wrapped runtime is *not* embedded in the pipeline; it is passed
    per-call to :meth:`AgentPipeline.query` (AgentDojo's design).  The
    hook references the same wrapper instance so its
    :meth:`WrappedFunctionsRuntime.refresh_functions` can be invoked
    in-place.
    """
    llm = _build_llm(pipeline_model, api_base=api_base, api_key=api_key)
    msg_hook = _MessageStreamHook(emit=emit)
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=send_event,
        loop=loop,
    )
    # Inner loop: per-turn tool execution -> message-stream emission ->
    # catalog edit -> next LLM call -> emit the assistant turn output.
    tools_loop = ToolsExecutionLoop(
        [ToolsExecutor(), msg_hook, hook, llm, msg_hook],
    )
    # Outer: SystemMessage + InitQuery prep messages, then the first LLM
    # call.  msg_hook fires AFTER the first LLM call to capture the
    # system + user + first assistant messages in one batch (its
    # ``_next_idx`` cursor walks the message list from 0 up).
    pipeline = AgentPipeline(
        [
            SystemMessage(system_prompt),
            InitQuery(),
            hook,
            llm,
            msg_hook,
            tools_loop,
        ]
    )
    pipeline.name = pipeline_model
    return pipeline


__all__ = ["build_pipeline"]
