"""AgentPipeline bridge: build the upstream pipeline with our wrapped runtime.

Provides two things:

- :func:`build_pipeline` - constructs an :class:`AgentPipeline` for a
  given litellm-style model id (e.g. ``openai/gpt-4o-2024-05-13``),
  splicing a :class:`_CatalogEditHook` once so the four tool-catalog
  Controllables fire a single time, before the first LLM call.
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
from collections.abc import Callable, Sequence
from typing import Any

import re

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
from agentdojo_target.observables import (
    catalog_edit_outcome_observable,
    chat_message_observable,
    discarded_action_observable,
)
from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.security_tags import (
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
)
from agentdojo_target.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


def _noop_emit(event: Any) -> None:
    """Default no-op emit so elements constructed without an emitter (e.g.
    in existing unit tests) stay fire-and-forget silent."""
    return None


# ---------------------------------------------------------------------------
# Catalog-edit hook
# ---------------------------------------------------------------------------


class _CatalogEditHook(BasePipelineElement):
    """Pipeline element that fires the four tool-catalog Controllables
    once, before the first LLM turn.

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
        emit: EventHandler = _noop_emit,
    ) -> None:
        self._catalog = catalog
        self._wrapper = wrapper
        self._send_event = send_event
        self._loop = loop
        self._emit = emit

    _OPTIMIZER_RESPONSE_TIMEOUT_SECONDS: float = 180.0

    def _await_event(self, event: ControllablePreCallEvent) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._send_event(event), self._loop)
        return future.result(timeout=self._OPTIMIZER_RESPONSE_TIMEOUT_SECONDS)

    def _emit_outcome(self, operation: str, tag: Any, status: str, detail: str | None) -> None:
        """Emit a one-way observable recording one catalogue-edit outcome.

        Fire-and-forget; never alters the apply/reject decision."""
        self._emit(
            ObservableEvent(
                observable=catalog_edit_outcome_observable(operation, tag),
                content={"operation": operation, "status": status, "detail": detail},
            )
        )

    def _try_apply(self, method: Any, raw: str, operation: str, tag: Any) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Catalog injection payload was not valid JSON: %s", exc)
            self._emit_outcome(operation, tag, "rejected", f"invalid JSON: {exc}")
            return
        if not isinstance(payload, dict):
            logger.warning(
                "Catalog injection payload must be a dict, got %s",
                type(payload).__name__,
            )
            self._emit_outcome(
                operation,
                tag,
                "rejected",
                f"payload not a dict: {type(payload).__name__}",
            )
            return
        try:
            method(payload)
        except (ValueError, ValidationError) as exc:
            logger.warning("Catalog mutation rejected: %s", exc)
            self._emit_outcome(operation, tag, "rejected", str(exc))
            return
        self._emit_outcome(operation, tag, "applied", None)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        ops = (
            (
                TOOL_CATALOG_REGISTER_CTRL,
                self._catalog.apply_register,
                "register",
                TOOL_CATALOGUE_ADD_TAG,
            ),
            (
                TOOL_CATALOG_REPLACE_CTRL,
                self._catalog.apply_replace,
                "replace",
                TOOL_CATALOGUE_EDIT_TAG,
            ),
            (
                TOOL_CATALOG_UNREGISTER_CTRL,
                self._catalog.apply_unregister,
                "unregister",
                TOOL_CATALOGUE_REMOVE_TAG,
            ),
            (
                TOOL_CATALOG_REWRITE_DOC_CTRL,
                self._catalog.apply_rewrite_doc,
                "rewrite_doc",
                TOOL_CATALOGUE_EDIT_TAG,
            ),
        )
        applied_any = False
        for ctrl, apply_method, op_label, op_tag in ops:
            event = ControllablePreCallEvent(controllable=ctrl, request="catalog edit slot")
            response = self._await_event(event)
            if isinstance(response, ControllableInjection):
                self._try_apply(apply_method, response.value, op_label, op_tag)
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
            idx = self._next_idx
            self._next_idx += 1
            # Tool calls and tool-result content are emitted exactly once, on
            # the per-tool ``ControllablePostCallEvent`` — never mirrored on the
            # agent trace.  So skip tool-result messages, and strip the
            # tool-call request from assistant messages: agent_trace_messages
            # carries only the non-tool internal stream (system / assistant
            # reasoning / user).
            if msg.get("role") == "tool":
                continue
            payload = _message_to_jsonable(msg)
            payload.pop("tool_calls", None)
            payload.pop("tool_call", None)
            self._emit(
                ObservableEvent(
                    observable=chat_message_observable(idx),
                    content=payload,
                )
            )
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
                if hasattr(tc, "function")
                else tc
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


class _CompatibleOpenAILLM(OpenAILLM):
    """OpenAILLM that filters parallel-tool-call synthetic wrapper names
    out of assistant responses.

    Older OpenAI models (notably ``gpt-4-turbo-2024-04-09``) sometimes
    emit a synthetic ``multi_tool_use.parallel`` outer tool_call when the
    model wants to invoke several tools in one turn.  That name contains
    a ``.`` and fails OpenAI's input-validation pattern
    ``^[a-zA-Z0-9_-]+$`` on the NEXT submission, hard-aborting the run.
    The default model ``gpt-4o-2024-05-13`` does not exhibit this; this
    subclass is a defensive workaround that keeps the port usable when
    the LiteLLM proxy substitutes an older turbo build.

    Behaviour change vs upstream: after the LLM responds, any tool_call
    whose ``function`` name violates the OpenAI pattern is dropped
    in-place AND the malformed-call's arguments are unpacked into
    additional valid tool_calls when the synthetic shape is recognised
    (``multi_tool_use.parallel`` with ``args.tool_uses = [{...}]``).
    Otherwise the message proceeds with the remaining valid tool_calls.
    """

    _VALID_NAME_RE: Any = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        *,
        emit: EventHandler = _noop_emit,
    ) -> None:
        super().__init__(client, model)
        self._emit = emit
        self._discard_counter = 0

    def query(  # type: ignore[override]
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        out_query, out_runtime, out_env, out_messages, out_extra = super().query(
            query,
            runtime,
            env,
            messages,
            extra_args,
        )
        out_messages = list(out_messages)
        if not out_messages:
            return out_query, out_runtime, out_env, out_messages, out_extra
        last = out_messages[-1]
        if last.get("role") != "assistant":
            return out_query, out_runtime, out_env, out_messages, out_extra
        tool_calls = last.get("tool_calls")
        if not tool_calls:
            return out_query, out_runtime, out_env, out_messages, out_extra
        sanitised = _sanitise_tool_calls(tool_calls)
        if sanitised != tool_calls:
            dropped_names = [
                name
                for call in tool_calls
                if not (
                    isinstance((name := _tool_call_name(call)), str)
                    and self._VALID_NAME_RE.match(name)
                )
            ]
            logger.warning(
                "Filtered %d malformed tool_call name(s) from the assistant "
                "response (parallel-wrapper compatibility)",
                len(dropped_names),
            )
            out_messages[-1] = {**last, "tool_calls": sanitised}
            # One-way notice that a malformed action the model requested was
            # discarded, so a trace-scoped reader sees it rather than the
            # call silently vanishing.  Fire-and-forget; the sanitise
            # decision above is unchanged.
            self._emit(
                ObservableEvent(
                    observable=discarded_action_observable(self._discard_counter),
                    content={
                        "dropped_names": dropped_names,
                        "dropped_count": len(dropped_names),
                    },
                )
            )
            self._discard_counter += 1
        return out_query, out_runtime, out_env, out_messages, out_extra


def _tool_call_name(call: Any) -> Any:
    """Extract the function name from a FunctionCall object or a dict."""
    name = getattr(call, "function", None)
    if name is None and isinstance(call, dict):
        name = call.get("function")
    return name


def _sanitise_tool_calls(tool_calls: Sequence[Any]) -> list[Any]:
    """Return ``tool_calls`` with any pattern-violating names removed.

    If a call's function name matches the OpenAI pattern it passes
    through unchanged.  If it is the synthetic
    ``multi_tool_use.parallel`` wrapper, expand it into one
    :class:`FunctionCall` per inner ``tool_uses`` entry.  Anything else
    that fails the pattern is dropped (logged at call-site).
    """
    from agentdojo.functions_runtime import FunctionCall

    valid_re = re.compile(r"^[a-zA-Z0-9_-]+$")
    out: list[Any] = []
    for call in tool_calls:
        name = getattr(call, "function", None)
        if name is None and isinstance(call, dict):
            name = call.get("function")
        if isinstance(name, str) and valid_re.match(name):
            out.append(call)
            continue
        # Try to unpack the multi_tool_use.parallel synthetic wrapper.
        args = getattr(call, "args", None)
        if args is None and isinstance(call, dict):
            args = call.get("args")
        tool_uses = None
        if isinstance(args, dict):
            tool_uses = args.get("tool_uses")
        if isinstance(tool_uses, list):
            for sub in tool_uses:
                if not isinstance(sub, dict):
                    continue
                inner_name = sub.get("recipient_name") or sub.get("name")
                inner_args = sub.get("parameters") or sub.get("args") or {}
                if isinstance(inner_name, str) and valid_re.match(inner_name):
                    out.append(
                        FunctionCall(
                            function=inner_name,
                            args=inner_args if isinstance(inner_args, dict) else {},
                            id=getattr(call, "id", None)
                            or (call.get("id") if isinstance(call, dict) else None),
                        )
                    )
    return out


def _build_llm(
    model_id: str,
    *,
    api_base: str | None,
    api_key: str | None,
    emit: EventHandler = _noop_emit,
) -> tuple[BasePipelineElement, Callable[[], None]]:
    """Construct an AgentDojo LLM element from a litellm-style model id.

    Supported providers:
    - ``openai/<model>``  -> :class:`_CompatibleOpenAILLM` (subclass of
      AgentDojo's ``OpenAILLM`` that filters malformed parallel-wrapper
      tool-call names; see that class for the rationale).
    - ``anthropic/<model>`` -> :class:`AnthropicLLM` with an
      :class:`anthropic.Anthropic` client; supports the ``-thinking-N``
      suffix in the model name.

    Args:
        model_id: e.g. ``openai/gpt-4o-2024-05-13``.
        api_base: Override base URL (most relevant for litellm-proxy).
        api_key: Override API key (defaults to environment lookup).
        emit: Trajectory emitter passed to the LLM element for one-way
            control observables (e.g. a discarded malformed action).

    Returns:
        A ``(element, close)`` pair.  ``close`` releases the underlying
        provider client's connection pool; the caller must invoke it once
        the run is finished (including on the failure path).

    Raises:
        NotImplementedError: If the provider prefix is not openai or
            anthropic.
        ValueError: If the model id does not include a provider prefix.
    """
    if "/" not in model_id:
        raise ValueError(f"pipeline_model must be in 'provider/model' form, got {model_id!r}")
    provider, _, model_name = model_id.partition("/")
    if provider == "openai":
        client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        return _CompatibleOpenAILLM(client, model_name, emit=emit), client.close
    if provider == "anthropic":
        anthropic_client = anthropic.Anthropic(
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
            return (
                AnthropicLLM(anthropic_client, base_model, thinking_budget_tokens=budget_tokens),
                anthropic_client.close,
            )
        return AnthropicLLM(anthropic_client, model_name), anthropic_client.close
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
) -> tuple[AgentPipeline, Callable[[], None]]:
    """Build the AgentDojo :class:`AgentPipeline` for one run.

    Splices two hooks into the pipeline:

    - :class:`_CatalogEditHook` (ONCE, before the first LLM call): fires the
      four tool-catalog Controllables so attacker-scoped optimizers can mutate
      the catalog a single time at the start of the run.  It is deliberately
      NOT placed inside the tool-execution loop, so it does not re-fire on every
      agent turn (which produced four redundant Controllable events per turn).
    - :class:`_MessageStreamHook` (AFTER the LLM and after the
      ToolsExecutor): emits one ``agent_trace_message_NNNN`` observable
      per new message in the conversation stream.

    Otherwise this is the upstream ``no_defense`` baseline.

    The resulting pipeline shape is::

        AgentPipeline([
            SystemMessage(system_prompt),
            InitQuery(),
            CatalogEditHook,             # fires ONCE, before the first LLM turn
            llm,                          # first agent turn
            MessageStreamHook,           # emits system+user+first assistant
            ToolsExecutionLoop([
                ToolsExecutor(formatter),
                MessageStreamHook,        # emits tool-result messages
                llm,
                MessageStreamHook,        # emits the assistant turn output
            ]),
        ])

    The same :class:`_MessageStreamHook` instance is reused across the
    splice points so its ``_next_idx`` cursor advances monotonically
    over the whole conversation -- emitting each message exactly once.
    The :class:`_CatalogEditHook` is spliced once, before the first LLM
    call, so it fires once per pipeline attempt -- exactly once in the
    normal run.  (AgentDojo's rare empty-output retry re-runs the whole
    pipeline and re-fires the hook up to 3x; this is safe and intentionally
    left unguarded, since the catalog persists across attempts and
    re-applies are idempotent.)  The catalog the optimizer produces then
    stays fixed for the run.

    The wrapped runtime is *not* embedded in the pipeline; it is passed
    per-call to :meth:`AgentPipeline.query` (AgentDojo's design).  The
    hook references the same wrapper instance so its
    :meth:`WrappedFunctionsRuntime.refresh_functions` can be invoked
    in-place.

    Returns:
        A ``(pipeline, close)`` pair.  ``close`` releases the underlying
        provider client opened for this run; the caller must invoke it
        when the run finishes, including on the failure path.
    """
    llm, close_llm = _build_llm(pipeline_model, api_base=api_base, api_key=api_key, emit=emit)
    msg_hook = _MessageStreamHook(emit=emit)
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=send_event,
        loop=loop,
        emit=emit,
    )
    # Inner loop: per-turn tool execution -> message-stream emission ->
    # next LLM call -> emit the assistant turn output.  The catalog-edit hook
    # is intentionally NOT spliced into the loop: it fires once before the first
    # LLM call (in the outer pipeline below), so the optimizer edits the catalog
    # a single time at the start rather than being re-prompted every turn.
    tools_loop = ToolsExecutionLoop(
        [ToolsExecutor(), msg_hook, llm, msg_hook],
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
    return pipeline, close_llm


__all__ = ["build_pipeline"]
