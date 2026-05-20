"""AgentPipeline bridge: build the upstream pipeline with our wrapped runtime.

Provides two things:

- :func:`build_pipeline` — constructs an :class:`AgentPipeline` for a
  given litellm-style model id (e.g. ``openai/gpt-4o-2024-05-13``),
  splicing a per-turn :class:`_CatalogEditHook` so the four tool-catalog
  Controllables fire before every LLM call (including the first one).
- :class:`_CatalogEditHook` — a :class:`BasePipelineElement` that fires
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
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage
from superred.core.types.event import EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
)

from agentdojo_target.controllables import (
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
)
from agentdojo_target.runtime_wrapper import WrappedFunctionsRuntime
from agentdojo_target.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog-edit hook
# ---------------------------------------------------------------------------


class _CatalogEditHook(BasePipelineElement):
    """Pipeline element that fires the four tool-catalog Controllables
    before every LLM turn.

    The hook fires four :class:`ControllablePreCallEvent`s — one per
    catalog operation — in a fixed order (register, replace,
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

    def _await_event(self, event: ControllablePreCallEvent) -> Any:
        future = asyncio.run_coroutine_threadsafe(self._send_event(event), self._loop)
        return future.result()

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
        except ValueError as exc:
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
    loop: asyncio.AbstractEventLoop,
    api_base: str | None = None,
    api_key: str | None = None,
) -> AgentPipeline:
    """Build the AgentDojo :class:`AgentPipeline` for one run.

    Splices a :class:`_CatalogEditHook` before every LLM call (the
    out-of-loop first call AND the in-loop subsequent calls), so the
    optimizer can edit the tool catalog at the start of each agent
    turn.  Otherwise this is the upstream ``no_defense`` baseline.

    The resulting pipeline shape is::

        AgentPipeline([
            SystemMessage(system_prompt),
            InitQuery(),
            CatalogEditHook,            # fires before first LLM turn
            llm,                         # first agent turn
            ToolsExecutionLoop([
                ToolsExecutor(formatter),
                CatalogEditHook,         # fires before every subsequent turn
                llm,
            ]),
        ])

    The wrapped runtime is *not* embedded in the pipeline; it is passed
    per-call to :meth:`AgentPipeline.query` (AgentDojo's design).  The
    hook references the same wrapper instance so its
    :meth:`WrappedFunctionsRuntime.refresh_functions` can be invoked
    in-place.
    """
    llm = _build_llm(pipeline_model, api_base=api_base, api_key=api_key)
    hook = _CatalogEditHook(
        catalog=catalog,
        wrapper=wrapper,
        send_event=send_event,
        loop=loop,
    )
    tools_loop = ToolsExecutionLoop([ToolsExecutor(), hook, llm])
    pipeline = AgentPipeline(
        [
            SystemMessage(system_prompt),
            InitQuery(),
            hook,
            llm,
            tools_loop,
        ]
    )
    pipeline.name = pipeline_model
    return pipeline


__all__ = ["build_pipeline"]
