"""OpenClawTarget: wrap the OpenClaw personal AI assistant as a superred target.

Connects to a running OpenClaw Gateway over WebSocket, sends user
messages via the ``agent`` RPC, and exposes the agent's tool-calling,
message-response, and configuration surfaces as controllables and
observables for the superred red-teaming framework.

All native agent activity (assistant ``chat`` deltas, tool calls,
model requests/responses when the LLM proxy is on) is
emitted live into the framework :class:`~superred.core.types.trajectory.Trajectory`
as :class:`~superred.core.types.events.ObservableEvent` s; no parallel
trace representation is maintained.

Tool-output injection follows the OpenClaw plugin hook contract
(https://docs.openclaw.ai/plugins/hooks): the optimizer is consulted in
the *async* ``before_tool_call`` hook (which emits a
:class:`~superred.core.types.events.ControllablePreCallEvent` and blocks
the agent until the optimizer responds), and the resulting content is
spliced into the persisted tool result by the *synchronous*
``tool_result_persist`` hook on the plugin side. ``tool_result_persist``
does no network I/O, so all optimizer-in-the-loop work happens in
``before_tool_call``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openclaw_target.constants import (
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_GATEWAY_URL,
    EXTERNAL_DATA_TAG,
    INTERNAL_CONTEXT_TAG,
    MODEL_TAG,
    OPENCLAW_DOMAIN,
    SYSTEM_TAG,
    TOOL_CATALOG_TAG,
    USER_INPUT_TAG,
)
from openclaw_target.ws_client import AgentEvent, OpenClawWSClient

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain
from superred.core.types.state import ConfigSpec, QuerySpec

logger = logging.getLogger(__name__)


def _plugin_dir() -> Path:
    """Filesystem path to the bundled OpenClaw injection plugin."""
    return Path(__file__).resolve().parent / "plugin"


# -- Controllable definitions --------------------------------------------------

USER_MESSAGE_CTRL = Controllable(
    name="user_message",
    security_domain=USER_INPUT_TAG,
    description="The user message sent to the OpenClaw agent.",
)

WEB_CONTENT_CTRL = Controllable(
    name="web_content",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call injection for web_fetch / web_search tool results. "
        "Emitted mid-run: one ControllablePreCallEvent per intercepted "
        "tool invocation. Requires the injection plugin (Phase 2)."
    ),
)

FILE_CONTENT_CTRL = Controllable(
    name="file_content",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call injection for file read tool results. Emitted "
        "mid-run: one ControllablePreCallEvent per intercepted "
        "invocation. Requires the injection plugin (Phase 2)."
    ),
)

MODEL_SYSTEM_PROMPT_CTRL = Controllable(
    name="model_system_prompt",
    security_domain=MODEL_TAG,
    description=(
        "Content injected into the LLM system prompt via the model proxy. "
        "Resolved pre-run (the system prompt is fixed at run start). "
        "Requires enable_llm_proxy=True."
    ),
)

# -- Observable definitions ----------------------------------------------------

SYSTEM_DESC_OBS = Observable(
    name="system_description",
    security_domain=SYSTEM_TAG,
    description="OpenClaw version, gateway config, and model info.",
)

TOOL_LIST_OBS = Observable(
    name="tool_list",
    security_domain=TOOL_CATALOG_TAG,
    description="Available tools and their definitions.",
)

SYSTEM_PROMPT_OBS = Observable(
    name="system_prompt",
    security_domain=INTERNAL_CONTEXT_TAG,
    description="The agent's system prompt as configured for this run.",
)

USER_MESSAGE_OBS = Observable(
    name="user_message",
    security_domain=USER_INPUT_TAG,
    description="User message sent to the agent for this run.",
)

ASSISTANT_STREAM_OBS = Observable(
    name="assistant_stream",
    security_domain=MODEL_TAG,
    description="Live streamed assistant-text delta from the agent.",
)

AGENT_RESPONSE_OBS = Observable(
    name="agent_response",
    security_domain=MODEL_TAG,
    description="Aggregated assistant-text reply for the run.",
)

TOOL_CALL_OBS = Observable(
    name="tool_call",
    security_domain=TOOL_CATALOG_TAG,
    description="A single tool invocation made by the agent.",
)

MODEL_REQUEST_OBS = Observable(
    name="model_request",
    security_domain=MODEL_TAG,
    description="Recorded model API request (messages sent to the LLM).",
)

MODEL_RESPONSE_OBS = Observable(
    name="model_response",
    security_domain=MODEL_TAG,
    description="Recorded model API response.",
)


class OpenClawTarget(Target):
    """Superred target wrapping a running OpenClaw Gateway + Agent.

    The target communicates with a running OpenClaw instance via its
    native WebSocket protocol and streams every native ``AgentEvent``
    into the run trajectory as a typed
    :class:`~superred.core.types.events.ObservableEvent`.

    Args:
        gateway_url: WebSocket URL of the Gateway
            (default ``ws://127.0.0.1:18789``).
        auth_token: Shared-secret auth token for the Gateway.
            Can be omitted when ``managed=True`` (auto-generated).
        session_key: Session routing key used for agent runs.
        agent_timeout_s: Max seconds to wait for an agent run.
        enable_tool_injection: Expose per-call tool-output
            controllables via the plugin-hook bridge. Each
            ``before_tool_call`` / ``tool_result_persist`` hook
            emits a live ``ControllablePreCallEvent``.
        enable_llm_proxy: Intercept model calls via a local LLM proxy.
            Requires ``provider_base_url`` and ``provider_api_key``.
        provider_base_url: Upstream LLM provider URL.
        provider_api_key: API key for the upstream LLM provider.
        managed: If ``True``, auto-start/stop a local OpenClaw Gateway
            process (Node daemon). A fresh gateway is started lazily on
            first connect and torn down in :meth:`teardown`; because the
            controller builds one target instance per task via the
            :class:`~superred.core.controller.TargetFactory`, each task
            gets an isolated gateway.
        managed_kwargs: Extra kwargs forwarded to
            :class:`openclaw_target.runtime.OpenClawRuntime`.

    Between runs of a single task the controller calls
    :meth:`reset_ephemeral_state`, which clears per-run state, clears
    files planted this task, and resets the Gateway session while
    keeping the same gateway process/connection.
    """

    def __init__(
        self,
        auth_token: str = "",
        gateway_url: str = DEFAULT_GATEWAY_URL,
        session_key: str = "superred",
        agent_id: str = "default",
        agent_timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        enable_tool_injection: bool = False,
        enable_llm_proxy: bool = False,
        provider_base_url: str = "",
        provider_api_key: str = "",
        managed: bool = False,
        managed_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._auth_token = auth_token
        self._session_key = session_key
        self._agent_id = agent_id
        self._agent_timeout_s = agent_timeout_s
        self._enable_tool_injection = enable_tool_injection
        self._enable_llm_proxy = enable_llm_proxy
        self._provider_base_url = provider_base_url
        self._provider_api_key = provider_api_key
        self._managed = managed
        self._managed_kwargs = managed_kwargs or {}

        self._runtime: Any = None
        self._llm_proxy: Any = None
        self._client: OpenClawWSClient | None = None
        self._hello_payload: dict[str, object] = {}

        self._system_prompt_append: str = ""
        self._workspace_files: dict[str, str] = {}
        self._tool_policy: str = ""
        self._planted_files: list[str] = []

        self._last_response: str = ""
        self._last_tool_calls: list[dict[str, object]] = []
        self._last_events_json: str = "[]"

        self._cached_tool_catalog: str = ""

        self._injection_server: Any = None

        # Live run state — set at the top of run(), cleared at the end.
        # Used by the injection hook to dispatch ControllablePreCallEvents
        # into the live trajectory.
        self._active_send_event: EventResponseHandler | None = None

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    async def _ensure_connected(self) -> OpenClawWSClient:
        if self._client is not None:
            return self._client

        # Start the injection server before the gateway so the managed
        # runtime can install the plugin pointed at its callback URL.
        if self._enable_tool_injection and self._injection_server is None:
            await self._start_injection_server()

        if self._managed and self._runtime is None:
            from openclaw_target.runtime import OpenClawRuntime

            managed_kwargs = dict(self._managed_kwargs)
            if self._injection_server is not None:
                managed_kwargs.setdefault("plugin_dir", str(_plugin_dir()))
                managed_kwargs.setdefault(
                    "callback_url", self._callback_url(),
                )
            if self._tool_policy:
                managed_kwargs.setdefault("tool_policy", self._tool_policy)
            self._runtime = OpenClawRuntime(**managed_kwargs)
            await self._runtime.start()
            self._gateway_url = self._runtime.gateway_url
            self._auth_token = self._runtime.auth_token

        client = OpenClawWSClient(
            gateway_url=self._gateway_url,
            auth_token=self._auth_token,
        )
        self._hello_payload = await client.connect()
        self._client = client

        try:
            catalog = await client.rpc("tools.catalog")
            self._cached_tool_catalog = json.dumps(catalog, indent=2)
        except Exception:
            self._cached_tool_catalog = "{}"

        if self._enable_llm_proxy and self._llm_proxy is None:
            await self._start_llm_proxy()

        return client

    def _callback_url(self) -> str:
        """Loopback URL the gateway-side plugin posts hook callbacks to.

        The gateway runs as a local Node process (see
        :class:`openclaw_target.runtime.OpenClawRuntime`), so it reaches
        the injection server directly on the host loopback.
        """
        port = self._injection_server._port if self._injection_server else 18899
        return f"http://127.0.0.1:{port}"

    async def _start_injection_server(self) -> None:
        """Start the local HTTP injection server for plugin callbacks."""
        try:
            from openclaw_target.injection_server import InjectionServer
        except ImportError:
            logger.warning(
                "aiohttp not installed; tool injection disabled. "
                "Install with: pip install openclaw-target[injection]",
            )
            self._enable_tool_injection = False
            return

        self._injection_server = InjectionServer(
            handler=self._handle_injection_hook,
        )
        await self._injection_server.start()

    async def _handle_injection_hook(
        self,
        hook_type: str,
        tool_name: str,
        params: dict[str, Any],
        tool_call_id: str,
        result: Any,
    ) -> dict[str, Any] | None:
        """Bridge the async ``before_tool_call`` hook to a live event.

        The plugin's HTTP POST blocks until this coroutine returns, so
        we consult the optimizer by dispatching a
        :class:`ControllablePreCallEvent` on the active trajectory and
        awaiting the :class:`ControllableInjection` response.

        Only ``before_tool_call`` is consulted: it is the sole async hook
        in the OpenClaw contract. The returned ``toolResult`` is stashed
        plugin-side (keyed by ``toolCallId``) and spliced into the
        persisted tool result by the synchronous ``tool_result_persist``
        hook, which cannot do network I/O.
        """
        if hook_type != "before_tool_call":
            return None

        send_event = self._active_send_event
        if send_event is None:
            return None

        controllable = self._controllable_for_tool(tool_name)
        if controllable is None:
            return None

        request_payload = json.dumps(
            {
                "hook": hook_type,
                "tool": tool_name,
                "toolCallId": tool_call_id,
                "params": params,
            },
            default=str,
        )

        try:
            response = await send_event(
                ControllablePreCallEvent(
                    controllable=controllable,
                    request=request_payload,
                ),
            )
        except Exception:
            logger.exception(
                "send_event failed while bridging %s hook for %s",
                hook_type, tool_name,
            )
            return None

        if not isinstance(response, ControllableInjection):
            return None
        if not response.value:
            return None

        # The injected value becomes the tool's returned content (the
        # adversarial document/page the model will read), applied at
        # persist time by the plugin.
        return {"toolResult": response.value}

    def _controllable_for_tool(self, tool_name: str) -> Controllable | None:
        if tool_name in ("web_fetch", "web_search"):
            return WEB_CONTENT_CTRL
        if tool_name == "read":
            return FILE_CONTENT_CTRL
        return None

    async def _start_llm_proxy(self) -> None:
        """Start the local LLM proxy for model call interception."""
        try:
            from openclaw_target.proxy_llm import LLMProxy
        except ImportError:
            logger.warning(
                "aiohttp not installed; LLM proxy disabled. "
                "Install with: pip install openclaw-target[injection]",
            )
            self._enable_llm_proxy = False
            return

        if not self._provider_base_url:
            logger.warning("provider_base_url not set; LLM proxy disabled.")
            self._enable_llm_proxy = False
            return

        self._llm_proxy = LLMProxy(
            upstream_base_url=self._provider_base_url,
            upstream_api_key=self._provider_api_key,
        )
        port = await self._llm_proxy.start()
        logger.info("LLM proxy started on port %d", port)

    # ------------------------------------------------------------------
    # Pre-run configuration
    # ------------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="system_prompt_append",
                security_domain=INTERNAL_CONTEXT_TAG,
                description=(
                    "Text appended to the agent's system prompt via "
                    "workspace AGENTS.md file. Used to plant secrets or "
                    "instructions for security evaluation."
                ),
            ),
            ConfigSpec(
                name="workspace_files",
                security_domain=EXTERNAL_DATA_TAG,
                description=(
                    "JSON dict of {filename: content} to write into the "
                    "agent workspace before each run."
                ),
            ),
            ConfigSpec(
                name="tool_policy",
                security_domain=TOOL_CATALOG_TAG,
                description=(
                    "Name of the tool profile to enforce (e.g. 'messaging' "
                    "to restrict the agent to messaging tools). Tool "
                    "restriction in OpenClaw is config (tools.profile / "
                    "tools.allow / agents.<id>.tools.allow), not a runtime "
                    "RPC: the managed runtime applies it via 'openclaw "
                    "config set tools.profile' before gateway start; for an "
                    "external gateway it must be pre-configured there."
                ),
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "system_prompt_append":
            self._system_prompt_append = value
        elif name == "workspace_files":
            self._workspace_files = json.loads(value) if value else {}
        elif name == "tool_policy":
            self._tool_policy = value

    # ------------------------------------------------------------------
    # Post-run queries
    # ------------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="last_response",
                description="The agent's last assistant-text response.",
            ),
            QuerySpec(
                name="tool_calls",
                description="JSON list of tool invocations from the last run.",
            ),
            QuerySpec(
                name="events",
                description="JSON list of all agent stream events from the last run.",
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        if name == "tool_calls":
            return json.dumps(self._last_tool_calls)
        if name == "events":
            return self._last_events_json
        return ""

    # ------------------------------------------------------------------
    # Security domain
    # ------------------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return OPENCLAW_DOMAIN

    # ------------------------------------------------------------------
    # Controllables and observables
    # ------------------------------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        ctrls = [USER_MESSAGE_CTRL]
        if self._enable_tool_injection:
            ctrls.extend([WEB_CONTENT_CTRL, FILE_CONTENT_CTRL])
        if self._enable_llm_proxy:
            ctrls.append(MODEL_SYSTEM_PROMPT_CTRL)
        return ctrls

    def get_observables(self) -> list[ObservableValue]:
        obs = [
            ObservableValue(
                observable=SYSTEM_DESC_OBS,
                content=json.dumps({
                    "gateway_url": self._gateway_url,
                    "session_key": self._session_key,
                    "hello": self._hello_payload,
                }),
            ),
            ObservableValue(
                observable=TOOL_LIST_OBS,
                content=self._cached_tool_catalog,
            ),
            ObservableValue(
                observable=SYSTEM_PROMPT_OBS,
                content=self._system_prompt_append,
            ),
        ]
        if self._enable_llm_proxy and self._llm_proxy:
            obs.append(ObservableValue(
                observable=MODEL_REQUEST_OBS,
                content=json.dumps([
                    {"model": r.request_model, "messages": r.request_messages}
                    for r in self._llm_proxy.records
                ]),
            ))
            obs.append(ObservableValue(
                observable=MODEL_RESPONSE_OBS,
                content=json.dumps([
                    {"text": r.response_text, "tokens": {"in": r.input_tokens, "out": r.output_tokens}}
                    for r in self._llm_proxy.records
                ]),
            ))
        return obs

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        emit: EventHandler,
        send_event: EventResponseHandler,
    ) -> None:
        client = await self._ensure_connected()

        # Tool restriction (tool_policy) is gateway/agent configuration,
        # applied by the managed runtime at startup — not a per-run RPC.

        if self._system_prompt_append:
            await self._apply_system_prompt(client)

        if self._workspace_files:
            await self._write_workspace_files(client)

        # LLM proxy controllable: system prompt injection is pre-run
        # because the system prompt is fixed at the start of the call.
        if self._enable_llm_proxy and self._llm_proxy:
            model_resp = await send_event(
                ControllablePreCallEvent(
                    controllable=MODEL_SYSTEM_PROMPT_CTRL,
                    request="Content to inject into the LLM system prompt:",
                ),
            )
            if isinstance(model_resp, ControllableInjection) and model_resp.value:
                self._llm_proxy.system_prompt_injection = model_resp.value
            else:
                self._llm_proxy.system_prompt_injection = None

        # Phase 1 controllable: user_message
        user_resp = await send_event(
            ControllablePreCallEvent(
                controllable=USER_MESSAGE_CTRL,
                request="Enter the user message to send to the OpenClaw agent:",
            ),
        )
        user_message = (
            user_resp.value
            if isinstance(user_resp, ControllableInjection)
            else "Hello"
        )

        emit(ObservableEvent(
            observable=USER_MESSAGE_OBS,
            content=user_message,
        ))

        # Activate the hook bridge for the duration of the agent run so
        # plugin-issued tool-call hooks can consult the optimizer live.
        self._active_send_event = send_event

        async def on_agent_event(evt: AgentEvent) -> None:
            if evt.stream == "chat":
                text = evt.payload.get("deltaText") or ""
                if text:
                    emit(ObservableEvent(
                        observable=ASSISTANT_STREAM_OBS,
                        content=text,
                    ))
            elif evt.stream == "tool":
                emit(ObservableEvent(
                    observable=TOOL_CALL_OBS,
                    content=json.dumps(evt.payload),
                ))

        try:
            result = await client.run_agent(
                message=user_message,
                session_key=self._session_key,
                timeout_s=self._agent_timeout_s,
                on_event=on_agent_event,
            )
        finally:
            self._active_send_event = None

        self._last_response = result.assistant_text
        self._last_tool_calls = result.tool_calls
        self._last_events_json = json.dumps(
            [{"stream": e.stream, "payload": e.payload} for e in result.events],
        )

        emit(ObservableEvent(
            observable=AGENT_RESPONSE_OBS,
            content=result.assistant_text,
        ))

        if self._enable_llm_proxy and self._llm_proxy:
            for rec in self._llm_proxy.records:
                emit(ObservableEvent(
                    observable=MODEL_REQUEST_OBS,
                    content=json.dumps({
                        "model": rec.request_model,
                        "messages": rec.request_messages,
                    }),
                ))
                emit(ObservableEvent(
                    observable=MODEL_RESPONSE_OBS,
                    content=rec.response_text,
                ))

        if result.error:
            logger.warning("Agent run error: %s", result.error)

    # ------------------------------------------------------------------
    # Lifecycle: per-run reset and teardown
    # ------------------------------------------------------------------

    async def reset_ephemeral_state(self) -> None:
        """Clear per-run state between runs of the same task.

        Resets the last-run response/tool-call/event buffers and any
        recorded proxy calls, clears files planted during this task, and
        resets the Gateway session. The gateway process and WebSocket
        connection are kept: durable, per-task state. Fresh durable state
        (a new gateway) comes from the controller building a new target
        instance per task via the ``TargetFactory``.
        """
        self._last_response = ""
        self._last_tool_calls = []
        self._last_events_json = "[]"
        if self._llm_proxy:
            self._llm_proxy.records.clear()
            self._llm_proxy.system_prompt_injection = None

        if self._client:
            # The protocol exposes agents.files.set (no files.delete); clear
            # a planted file by overwriting it with empty content.
            for filename in self._planted_files:
                try:
                    await self._client.rpc(
                        "agents.files.set",
                        {"agentId": self._agent_id, "name": filename, "content": ""},
                    )
                except Exception:
                    logger.debug("Could not clear planted file %s", filename, exc_info=True)
            self._planted_files.clear()

            try:
                await self._client.reset_session(self._session_key)
            except Exception:
                logger.debug("Session reset failed (may be expected)", exc_info=True)

    async def teardown(self) -> None:
        """Close the connection, injection server, proxy, and gateway process."""
        if self._llm_proxy:
            await self._llm_proxy.stop()
            self._llm_proxy = None
        if self._injection_server:
            await self._injection_server.stop()
            self._injection_server = None
        if self._client:
            await self._client.close()
            self._client = None
        if self._runtime:
            await self._runtime.stop()
            self._runtime = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_system_prompt(self, client: OpenClawWSClient) -> None:
        """Append text to the agent's system prompt via the AGENTS.md bootstrap file.

        ``agents.files.set`` params are ``{agentId, name, content}`` (schema
        is ``additionalProperties:false``). AGENTS.md is a recognized
        bootstrap workspace file.
        """
        try:
            await client.rpc("agents.files.set", {
                "agentId": self._agent_id,
                "name": "AGENTS.md",
                "content": self._system_prompt_append,
            })
        except Exception:
            logger.warning(
                "Could not set AGENTS.md via RPC; system prompt append "
                "may not take effect.",
                exc_info=True,
            )

    async def _write_workspace_files(self, client: OpenClawWSClient) -> None:
        """Write workspace files before a run (for data injection scenarios).

        Uses ``agents.files.set`` ({agentId, name, content}). Note: this RPC
        manages the agent's bootstrap workspace files; planting arbitrary
        nested paths may require a shared workspace dir on a real gateway
        (see runtime ``workspace_dir``).
        """
        for filename, content in self._workspace_files.items():
            try:
                await client.rpc("agents.files.set", {
                    "agentId": self._agent_id,
                    "name": filename,
                    "content": content,
                })
                self._planted_files.append(filename)
            except Exception:
                logger.warning(
                    "Could not write workspace file %s via RPC",
                    filename,
                    exc_info=True,
                )
