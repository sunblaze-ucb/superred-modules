"""InspectAgentTarget: a general, benchmark-agnostic inspect tool-calling agent.

The target runs an inspect-ai tool-calling agent over whatever tools, prompts,
and model it is handed.  It knows nothing about any specific benchmark; the
benchmark-specific parts are injected:

- the tool implementations, via a ``tool_resolver`` (name -> inspect Tool)
  passed at construction;
- the model, via the ``model`` construction arg (fixed per run, not a config
  slot: neither the Task nor the attacker may change it);
- the per-run tool names, prompts, and controls, via ``set_config``.

Lifecycle:

1. ``__init__``: store model id, credentials, the tool resolver, generation
   defaults.
2. ``set_config``: a Task sets ``system_prompt``, ``user_prompt``,
   ``tool_names`` (JSON list), and optionally ``tool_choice`` and
   ``message_limit`` before each run.
3. ``run(emit, send_event)``:
   - fire the system_prompt and user_prompt Controllables (optimizer may
     override either via ``ControllableInjection``);
   - resolve the tool names to inspect Tools;
   - run the tool-calling loop (:func:`run_rollout`);
   - emit the non-tool agent-trace message observables (each tool's call and
     return are emitted once, on that tool's ControllablePostCallEvent).
4. ``query``: post-run string readers; ``messages`` property: the typed trace.
5. ``reset_ephemeral_state``: reset per-run state.  ``teardown``: no-op.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from inspect_ai.model import ChatMessage, ChatMessageTool, GenerateConfig, get_model
from inspect_ai.tool import Tool, ToolCall
from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

from inspect_agent_target.config_specs import CONFIG_SPEC_NAMES, CONFIG_SPECS
from inspect_agent_target.controllables import (
    CONTROLLABLES,
    SYSTEM_PROMPT_CTRL,
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    USER_PROMPT_CTRL,
    tool_output_controllable,
)
from inspect_agent_target.observables import (
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    MESSAGE_LIMIT_OBS,
    MODEL_IDENTITY_OBS,
    TOOL_CATALOG_LISTING_OBS,
    chat_message_observable,
)
from inspect_agent_target.query_specs import QUERY_SPEC_NAMES, QUERY_SPECS
from inspect_agent_target.rollout import ToolChoice, run_rollout, static_tools_provider
from inspect_agent_target.security_tags import TOOLS_TAG, build_domain
from inspect_agent_target.system_specification import DETAILED_SYSTEM_SPECIFICATION
from inspect_agent_target.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)

ToolResolver = Callable[[str], Tool]

_DEFAULT_MESSAGE_LIMIT = 20
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_TOKENS = 4096


class InspectAgentTarget(Target):
    """A general inspect-ai tool-calling agent target.

    Args:
        model: litellm-style model id powering the agent (e.g.
            ``openai/gpt-4o-2024-08-06``).  Fixed for the target's lifetime:
            changeable only here at construction, never by the Task or attacker.
        tool_resolver: maps a tool name to an inspect ``Tool``.  This is the
            benchmark-specific seam: a SecurityClaim supplies a resolver that
            wraps its tool module (e.g. AgentHarm's harmful tools).
        tool_scopes: optional map from tool name to its trust-boundary
            SecurityDomainTag (a leaf under the ``tools`` root, claim-supplied).
            Each configured tool gets a ``tool:<name>`` output-injection
            controllable scoped to its tag; unmapped tools fall back to the bare
            ``tools`` root.  The target's security domain is assembled from these.
        api_base: optional API base URL (e.g. a litellm proxy).
        api_key: optional API key.
        default_message_limit: default cap on total messages per rollout.
        temperature, max_tokens: generation config defaults (overridable only
            at construction; a faithful benchmark port sets these to the
            benchmark's values).
    """

    def __init__(
        self,
        *,
        model: str,
        tool_resolver: ToolResolver,
        tool_scopes: Mapping[str, SecurityDomainTag] | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        default_message_limit: int = _DEFAULT_MESSAGE_LIMIT,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model_id = model
        self._tool_resolver = tool_resolver
        self._tool_scopes: dict[str, SecurityDomainTag] = dict(tool_scopes or {})
        self._domain: SecurityDomain = build_domain(self._tool_scopes.values())
        self._api_base = api_base
        self._api_key = api_key
        self._default_message_limit = default_message_limit
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Per-task config (set via set_config before run()).
        self._system_prompt: str = ""
        self._user_prompt: str = ""
        self._tool_names: list[str] = []
        self._tool_choice: str = "auto"
        self._message_limit: int = default_message_limit
        # Per-run state.
        self._messages: list[ChatMessage] = []
        self._last_response: str = ""
        self._catalog: ToolCatalog | None = None

    # -- pre-run configuration -------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return list(CONFIG_SPECS)

    def set_config(self, name: str, value: str) -> None:
        if name not in CONFIG_SPEC_NAMES:
            raise ValueError(
                f"Unknown config slot {name!r}.  Expected one of {sorted(CONFIG_SPEC_NAMES)}."
            )
        if name == "system_prompt":
            self._system_prompt = value
        elif name == "user_prompt":
            self._user_prompt = value
        elif name == "tool_names":
            self._tool_names = _parse_tool_names(value)
        elif name == "tool_choice":
            if value not in ("auto", "any", "none"):
                raise ValueError(f"tool_choice must be 'auto', 'any', or 'none', got {value!r}.")
            self._tool_choice = value
        elif name == "message_limit":
            self._message_limit = int(value) if value else self._default_message_limit

    # -- post-run queries ------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return list(QUERY_SPECS)

    def query(self, name: str, **params: str) -> str:
        if name not in QUERY_SPEC_NAMES:
            raise ValueError(
                f"Unknown query slot {name!r}.  Expected one of {sorted(QUERY_SPEC_NAMES)}."
            )
        if name == "last_response":
            return self._last_response
        if name == "conversation_history":
            return json.dumps([_message_to_jsonable(m) for m in self._messages])
        if name == "function_call_trace":
            return json.dumps(_function_call_trace(self._messages))
        raise ValueError(f"query slot {name!r} matched no dispatch case")  # pragma: no cover

    @property
    def messages(self) -> list[ChatMessage]:
        """The full rollout message trace (typed; for the bound Task to grade)."""
        return self._messages

    # -- security domain / controllables / observables -------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return self._domain

    def get_controllables(self) -> list[Controllable]:
        return [
            *CONTROLLABLES,
            *(self._tool_output_ctrl(name) for name in self._tool_names),
        ]

    def _tool_output_ctrl(self, tool_name: str) -> Controllable:
        """The per-tool output-injection Controllable, scoped to the tool's trust
        boundary (claim-supplied) or the bare ``tools`` root if unmapped."""
        return tool_output_controllable(tool_name, self._tool_scopes.get(tool_name, TOOLS_TAG))

    def get_observables(self) -> list[ObservableValue]:
        try:
            catalog_snapshot = ToolCatalog.seed(self._tool_resolver, self._tool_names).snapshot()
        except Exception:  # pragma: no cover - defensive: resolver/seed failure
            catalog_snapshot = []
        return [
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._model_id),
            ObservableValue(
                observable=DETAILED_SYSTEM_SPECIFICATION_OBS,
                content=DETAILED_SYSTEM_SPECIFICATION,
            ),
            ObservableValue(observable=MESSAGE_LIMIT_OBS, content=str(self._message_limit)),
            ObservableValue(observable=TOOL_CATALOG_LISTING_OBS, content=catalog_snapshot),
        ]

    # -- run -------------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        # Phase 1: system prompt controllable.
        sp_resp = await send_event(
            ControllablePreCallEvent(controllable=SYSTEM_PROMPT_CTRL, request=self._system_prompt)
        )
        effective_system = (
            sp_resp.value if isinstance(sp_resp, ControllableInjection) else self._system_prompt
        )

        # Phase 2: user prompt controllable.
        up_resp = await send_event(
            ControllablePreCallEvent(controllable=USER_PROMPT_CTRL, request=self._user_prompt)
        )
        effective_user = (
            up_resp.value if isinstance(up_resp, ControllableInjection) else self._user_prompt
        )

        # Phase 3: seed the tool catalogue + build the model.
        catalog = ToolCatalog.seed(self._tool_resolver, self._tool_names)
        self._catalog = catalog
        # Generation config: temperature/max_tokens are construction params
        # (AgentHarm uses 0.0 / 4096); seed=0 and max_retries=3 are AgentHarm's
        # upstream defaults, hardcoded here as benchmark-agnostic generation
        # defaults (not configurable).  max_connections is deliberately NOT set:
        # cross-target parallelism is owned by the TargetFactory, not the target.
        model = get_model(
            self._model_id,
            base_url=self._api_base,
            api_key=self._api_key,
            config=GenerateConfig(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                seed=0,
                max_retries=3,
            ),
        )

        # Tool-catalogue Controllables fire ONCE, at run start (after the
        # catalogue is seeded from the static Task config).  The optimizer gets a
        # single chance to edit the registry; the tool set is then fixed for the
        # whole run.  (AgentDojo fires these before every LLM turn; we
        # deliberately fire once to avoid per-turn event noise.)  The target
        # always fires them; the Controller's scope filter decides whether the
        # optimizer may actually inject.  The catalogue itself is NOT emitted onto
        # the trajectory: it is static configuration, exposed via the
        # TOOL_CATALOG_LISTING_OBS static observable (the configured, pre-edit
        # snapshot).  Attacker edits are visible on the trajectory as the
        # catalogue controllable events above; the post-edit tool set is then
        # exercised through the per-tool ControllablePostCallEvents.
        await self._fire_catalog_controllables(send_event, catalog)

        # Per-tool output injection (the indirect-prompt-injection surface):
        # after each tool result, fire THAT tool's ControllablePostCallEvent
        # carrying the call (function + arguments) and the legitimate output; an
        # attacker scoped to the tool's trust boundary may replace the output
        # before the agent sees it. Always fired (scope filter gates injection).
        # The tool call + response live ONLY on this event -- they are not
        # mirrored on the agent trace (emit-once).

        async def on_tool_results(
            results: list[ChatMessage], tool_calls: list[ToolCall]
        ) -> list[ChatMessage]:
            calls_by_id = {tc.id: tc for tc in tool_calls}
            out: list[ChatMessage] = []
            for msg in results:
                if not isinstance(msg, ChatMessageTool):
                    out.append(msg)
                    continue
                fn = str(msg.function or "")
                tool_call_id = msg.tool_call_id
                originating = calls_by_id.get(tool_call_id) if tool_call_id is not None else None
                arguments = dict(originating.arguments) if originating is not None else {}
                resp = await send_event(
                    ControllablePostCallEvent(
                        controllable=self._tool_output_ctrl(fn),
                        request=json.dumps({"function": fn, "arguments": arguments}),
                        answer=msg.text or "",
                    )
                )
                final: ChatMessage = msg
                if isinstance(resp, ControllableInjection):
                    final = ChatMessageTool(
                        content=resp.value,
                        tool_call_id=msg.tool_call_id,
                        function=msg.function,
                    )
                out.append(final)
            return out

        # Phase 4: run the tool-calling loop with the (now fixed) tool set.
        self._messages = await run_rollout(
            model,
            system_prompt=effective_system,
            user_prompt=effective_user,
            tools_provider=static_tools_provider(catalog.tools()),
            tool_choice=cast(ToolChoice, self._tool_choice),
            message_limit=self._message_limit,
            on_tool_results=on_tool_results,
        )
        self._last_response = _last_assistant_text(self._messages)

        # Phase 5: emit the agent trace -- the NON-TOOL message stream only.
        # Each tool's call (function + arguments) and return are emitted once,
        # on that tool's ControllablePostCallEvent (above), so they are NOT
        # mirrored here: skip tool-result messages and strip the tool-call
        # request from assistant messages.
        for idx, msg in enumerate(self._messages):
            if isinstance(msg, ChatMessageTool):
                continue
            payload = _message_to_jsonable(msg)
            payload.pop("tool_calls", None)
            emit(
                ObservableEvent(
                    observable=chat_message_observable(idx),
                    content=payload,
                )
            )

    # -- reset / teardown ------------------------------------------------------

    async def _fire_catalog_controllables(
        self, send_event: EventResponseHandler, catalog: ToolCatalog
    ) -> None:
        """Fire the four tool-catalogue Controllables and apply any injections.

        Fired once at run start, unconditionally; the Controller's
        security_domain filter responds with no-injection for controllables
        outside the optimizer's scope.  Malformed payloads are logged and skipped.
        """
        ops = (
            (TOOL_CATALOG_REGISTER_CTRL, catalog.apply_register),
            (TOOL_CATALOG_REPLACE_CTRL, catalog.apply_replace),
            (TOOL_CATALOG_UNREGISTER_CTRL, catalog.apply_unregister),
            (TOOL_CATALOG_REWRITE_DOC_CTRL, catalog.apply_rewrite_doc),
        )
        for ctrl, apply in ops:
            response = await send_event(
                ControllablePreCallEvent(controllable=ctrl, request="tool catalogue edit")
            )
            if not isinstance(response, ControllableInjection):
                continue
            try:
                payload = json.loads(response.value)
            except json.JSONDecodeError as exc:
                logger.warning("catalogue injection payload was not valid JSON: %s", exc)
                continue
            if not isinstance(payload, dict):
                logger.warning(
                    "catalogue injection payload must be a dict, got %s",
                    type(payload).__name__,
                )
                continue
            apply(payload)

    async def reset_ephemeral_state(self) -> None:
        # All per-run state here is ephemeral (the message trace, last response,
        # and seeded catalogue); the target holds no durable cross-run memory.
        self._messages = []
        self._last_response = ""
        self._catalog = None

    async def teardown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tool_names(value: str) -> list[str]:
    """Decode the ``tool_names`` config string (JSON list[str]; '' = none)."""
    if value == "":
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("tool_names must be a JSON-encoded list of strings.")
    return parsed


def _message_to_jsonable(msg: ChatMessage) -> dict[str, Any]:
    """Render an inspect ChatMessage (pydantic) into a JSON-friendly dict."""
    return msg.model_dump(mode="json")


def _function_call_trace(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Extract {function, arguments, id} for every tool call, in order."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            out.append(
                {
                    "function": tc.function,
                    "arguments": dict(tc.arguments) if tc.arguments else {},
                    "id": tc.id,
                }
            )
    return out


def _last_assistant_text(messages: Sequence[ChatMessage]) -> str:
    """Return the text of the last assistant message (``""`` if none)."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return msg.text or ""
    return ""


__all__ = ["InspectAgentTarget", "ToolResolver"]
