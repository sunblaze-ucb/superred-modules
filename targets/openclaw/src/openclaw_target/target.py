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

Tool-output injection uses two OpenClaw seams (verified against
openclaw/openclaw source):

* **Live same-turn** — ``registerAgentToolResultMiddleware`` on the
  embedded ``tool_result`` path rewrites the in-flight result before the
  provider continuation (``file_content``, ``web_content``, etc.).
* **Transcript poisoning (next prompt, same session)** — async
  ``before_tool_call`` + sync ``tool_result_persist`` rewrite only what
  gets persisted (``*_transcript`` controllables); surfaces on the next
  ``target.run()`` in the *same* session.

Both paths emit :class:`~superred.core.types.events.ControllablePostCallEvent`
per intercepted tool call (framework convention for tool-output injection).

A third, independent seam covers **cross-session memory poisoning**:

* **Durable memory (next session, any session)** — ``persistent_memory``
  fires once at the end of :meth:`OpenClawTarget.run`, with the full
  session ``chat.history`` as context, and writes its result via the real
  ``agents.files.set`` RPC into ``MEMORY.md``. Unlike ``*_transcript``
  (which only rewrites the session's own JSONL transcript) this is a real
  file write: it survives ``sessions.reset`` and a fresh session, matching
  the cross-session persistence dimension used by SafeClawArena
  (Stateful/Persistent State Exploitation) and CIK-Bench (``mem-long``),
  as distinct from same-session transcript poisoning (``mem-session`` /
  ``*_transcript``). This deliberately does *not* use OpenClaw's
  ``before_prompt_build`` / ``enqueueNextTurnInjection`` seams — those are
  ephemeral prompt context, not durable state, and ``enqueueNextTurnInjection``
  is not late-callable from outside plugin ``register()`` in any case.
"""

from __future__ import annotations

import json
import logging
import secrets
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
from openclaw_target.ws_client import AgentEvent, AgentRunResult, OpenClawWSClient

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
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


# ``agents.files.set``/``agents.files.get`` are capped to this fixed set of
# bootstrap/memory filenames (gateway-enforced; verified live against a real
# gateway - any other name is rejected with
# ``INVALID_REQUEST: unsupported file "<name>"``). See
# ``src/gateway/server-methods/agents.ts`` (``ALLOWED_FILE_NAMES`` =
# ``BOOTSTRAP_FILE_NAMES`` + ``MEMORY_FILE_NAMES``) and
# ``src/agents/workspace.ts`` / ``src/memory/root-memory-files.ts`` for the
# filename constants.
ALLOWED_WORKSPACE_BOOTSTRAP_FILES = frozenset({
    "AGENTS.md",
    "SOUL.md",
    "TOOLS.md",
    "IDENTITY.md",
    "USER.md",
    "HEARTBEAT.md",
    "BOOTSTRAP.md",
    "MEMORY.md",
})


# -- Controllable definitions --------------------------------------------------

USER_MESSAGE_CTRL = Controllable(
    name="user_message",
    security_domain=USER_INPUT_TAG,
    description="The user message sent to the OpenClaw agent.",
)

_LIVE_TOOL_INJECTION_NOTE = (
    "Live same-turn tool-output spoofing via OpenClaw "
    "registerAgentToolResultMiddleware (the in-flight tool_result path). "
    "The model sees the injected value during the current tool-calling loop. "
    "Requires the injection plugin."
)

_TRANSCRIPT_POISON_NOTE = (
    "Transcript / memory poisoning via before_tool_call → tool_result_persist "
    "(persisted session transcript only). Surfaces on the *next* prompt "
    "submitted in this session (poison now, trigger later). "
    "Requires the injection plugin."
)

WEB_CONTENT_CTRL = Controllable(
    name="web_content",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call live injection for web_fetch / web_search tool results. "
        "Emitted mid-run: one ControllablePostCallEvent per intercepted "
        "tool invocation. " + _LIVE_TOOL_INJECTION_NOTE
    ),
)

FILE_CONTENT_CTRL = Controllable(
    name="file_content",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call live injection for file read tool results. Emitted "
        "mid-run: one ControllablePostCallEvent per intercepted "
        "invocation. " + _LIVE_TOOL_INJECTION_NOTE
    ),
)

SHELL_OUTPUT_CTRL = Controllable(
    name="shell_output",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call live injection for exec/shell tool output (the `exec` and "
        "`process` agent tools). Emitted mid-run: one ControllablePostCallEvent "
        "per intercepted invocation. " + _LIVE_TOOL_INJECTION_NOTE
    ),
)

MESSAGE_CONTENT_CTRL = Controllable(
    name="message_content",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call live injection for the messaging tool's returned content "
        "(the `message` tool). Emitted mid-run: one ControllablePostCallEvent "
        "per intercepted invocation. " + _LIVE_TOOL_INJECTION_NOTE
    ),
)

WEB_CONTENT_TRANSCRIPT_CTRL = Controllable(
    name="web_content_transcript",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call transcript poisoning for web_fetch / web_search results. "
        "Emitted mid-run: one ControllablePostCallEvent per intercepted "
        "tool invocation. " + _TRANSCRIPT_POISON_NOTE
    ),
)

FILE_CONTENT_TRANSCRIPT_CTRL = Controllable(
    name="file_content_transcript",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call transcript poisoning for file read tool results. "
        "Emitted mid-run: one ControllablePostCallEvent per intercepted "
        "invocation. " + _TRANSCRIPT_POISON_NOTE
    ),
)

SHELL_OUTPUT_TRANSCRIPT_CTRL = Controllable(
    name="shell_output_transcript",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call transcript poisoning for exec/shell tool output. "
        "Emitted mid-run: one ControllablePostCallEvent per intercepted "
        "invocation. " + _TRANSCRIPT_POISON_NOTE
    ),
)

MESSAGE_CONTENT_TRANSCRIPT_CTRL = Controllable(
    name="message_content_transcript",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Per-call transcript poisoning for the messaging tool's returned "
        "content. Emitted mid-run: one ControllablePostCallEvent per "
        "intercepted invocation. " + _TRANSCRIPT_POISON_NOTE
    ),
)

_PERSISTENT_MEMORY_NOTE = (
    "Cross-session memory poisoning: fires once per run(), after the agent "
    "run completes, with the full session chat.history as context (so the "
    "optimizer has full-trajectory knowledge, not just a single tool call). "
    "The returned value is written via the real agents.files.set RPC into "
    "MEMORY.md — a genuine durable file, not a prompt hook. Survives "
    "sessions.reset and loads as system context in a fresh session/gateway "
    "restart. Distinct threat model from *_transcript (same-session "
    "transcript poisoning): this is cross-session memory poisoning, aligned "
    "with SafeClawArena's persistence dimension and CIK-Bench's `mem-long` "
    "vector. Requires enable_persistent_memory=True."
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

MODEL_RESPONSE_CTRL = Controllable(
    name="model_response_injection",
    security_domain=MODEL_TAG,
    description=(
        "Content injected into the LLM's response text via the model proxy "
        "(simulates a manipulated/compromised model output the agent then acts "
        "on). Resolved pre-run; applied to every model response during the run. "
        "Requires enable_llm_proxy=True."
    ),
)

PERSISTENT_MEMORY_CTRL = Controllable(
    name="persistent_memory",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Content written into the agent's durable MEMORY.md at the end of "
        "a run, with the full run trajectory as context. " + _PERSISTENT_MEMORY_NOTE
    ),
)

# -- Tool-output injection registry -------------------------------------------
#
# Maps an OpenClaw gateway tool name -> live and transcript controllables
# (a ControllablePostCallEvent fires per call when ``enable_tool_injection``
# is on; see each Controllable's description for the threat model).
# This is the single extension point for tool-output injection:
# ``get_controllables`` and the plugin bridge both derive from it, so adding
# a new capability is one entry here — define a Controllable with the right
# security domain and map its gateway tool name(s).
#
# Tool names are the real OpenClaw *agent-facing* tool identifiers (verified in
# openclaw/openclaw src/agents/agent-tools.ts and src/agents/*-tools.*):
#   web_fetch / web_search  -> web content
#   read                    -> file content
#   exec / process          -> exec/shell output (both gated by includeShellTools)
#   message                 -> messaging payloads
#
# Note: "bash" is intentionally NOT here. It is not an agent-catalogue tool: it
# lives in the sessions-SDK / sub-agent surface (src/agents/sessions/tools/bash.ts)
# and the ACP command set (src/acp/commands.ts), neither of which routes through
# the agent tool-call hook the injection plugin attaches to. "memory" is likewise
# excluded — it is a plugin slot (``plugins.slots.memory``), not a tool.
TOOL_OUTPUT_LIVE_CONTROLLABLES: dict[str, Controllable] = {
    "web_fetch": WEB_CONTENT_CTRL,
    "web_search": WEB_CONTENT_CTRL,
    "read": FILE_CONTENT_CTRL,
    "exec": SHELL_OUTPUT_CTRL,
    "process": SHELL_OUTPUT_CTRL,
    "message": MESSAGE_CONTENT_CTRL,
}

TOOL_OUTPUT_TRANSCRIPT_CONTROLLABLES: dict[str, Controllable] = {
    "web_fetch": WEB_CONTENT_TRANSCRIPT_CTRL,
    "web_search": WEB_CONTENT_TRANSCRIPT_CTRL,
    "read": FILE_CONTENT_TRANSCRIPT_CTRL,
    "exec": SHELL_OUTPUT_TRANSCRIPT_CTRL,
    "process": SHELL_OUTPUT_TRANSCRIPT_CTRL,
    "message": MESSAGE_CONTENT_TRANSCRIPT_CTRL,
}

# Back-compat alias: live same-turn registry (primary tool-output surface).
TOOL_OUTPUT_CONTROLLABLES = TOOL_OUTPUT_LIVE_CONTROLLABLES

# -- Observable definitions ----------------------------------------------------

MODEL_IDENTITY_OBS = Observable(
    name="model_identity",
    security_domain=MODEL_TAG,
    description="The configured target model identifier.",
)

SYSTEM_DESC_OBS = Observable(
    name="system_description",
    security_domain=SYSTEM_TAG,
    description="OpenClaw gateway config (url, session, agent, tool profile).",
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
        agent_id: OpenClaw agent id used for ``agents.files.set`` calls.
            Default ``"main"`` matches the agent id ``config.py`` always
            registers for a managed gateway (``agents.defaults`` / ``agents.list``);
            override only when pointing at an externally managed gateway with a
            differently named agent.
        model_id: Configured target model identifier, surfaced as a static
            observable so the optimizer sees it at initialization.
        agent_timeout_s: Max seconds to wait for an agent run.
        enable_tool_injection: Expose per-call tool-output
            controllables via the plugin-hook bridge. The async
            ``before_tool_call`` hook emits a live
            ``ControllablePostCallEvent`` (tool-output injection) and the
            sync ``tool_result_persist`` hook splices the result into the
            persisted session transcript. The real tool always executes
            for real; the injected content poisons what subsequent turns
            of the same session see as history (verified live: it does
            not rewrite the tool result the in-flight tool-calling loop
            continues with, only what later prompt submissions load).
        enable_persistent_memory: Expose the ``persistent_memory``
            controllable. Once per :meth:`run`, after the agent finishes,
            emits a ``ControllablePostCallEvent`` carrying the full session
            ``chat.history``; the returned value is written into
            ``MEMORY.md`` via the real ``agents.files.set`` RPC — a durable
            file, not a prompt hook, so it survives ``sessions.reset`` and a
            fresh session. Independent of ``enable_tool_injection`` (no
            plugin required; uses the same RPC as ``workspace_files``).
        reset_session_between_runs: If ``True``, clear the OpenClaw
            conversation/session in :meth:`reset_ephemeral_state`. Default
            ``False`` keeps the session across runs of a task — OpenClaw
            sessions are durable state, so a fresh conversation would lose
            intended context and break poison-then-trigger attacks.
        enable_llm_proxy: Intercept model calls via a local LLM proxy.
            ``None`` (default) turns it on whenever ``provider_base_url`` is
            set; pass ``False`` to force it off. When on (managed mode), the
            gateway is pointed at the proxy so model requests/responses are
            recorded and the system prompt can be injected.
        provider_base_url: Upstream LLM provider URL.
        provider_api_key: API key for the upstream LLM provider.
        managed: If ``True``, auto-start/stop an OpenClaw Gateway managed by
            this target. A fresh gateway is started lazily on first connect and
            torn down in :meth:`teardown`; because the controller builds one
            target instance per task via the
            :class:`~superred.core.controller.TargetFactory`, each task gets an
            isolated gateway.
        managed_runtime: Which managed backend to use when ``managed=True``:
            ``"local"`` (default) runs ``openclaw gateway`` as a loopback Node
            subprocess; ``"docker"`` runs the whole gateway in a fresh container
            per task (full host isolation, dynamic ports for safe
            ``concurrency>1``). Docker mode reaches host-side services (injection
            server + LLM proxy) via ``host.docker.internal``.
        managed_kwargs: Extra kwargs forwarded to the runtime
            (:class:`openclaw_target.runtime.OpenClawRuntime` or
            :class:`openclaw_target.docker_runtime.OpenClawDockerRuntime`).

    Between runs of a single task the controller calls
    :meth:`reset_ephemeral_state`, which clears only per-run (ephemeral)
    state. Durable task state — planted files / AGENTS.md and the
    conversation/session — is preserved across runs and discarded only
    when the controller obtains a fresh instance from the
    ``TargetFactory`` between tasks (which, when ``managed``, is a fresh
    gateway). Planted files are cleaned up in :meth:`teardown`.
    """

    def __init__(
        self,
        auth_token: str = "",
        gateway_url: str = DEFAULT_GATEWAY_URL,
        session_key: str = "superred",
        agent_id: str = "main",
        model_id: str = "",
        agent_timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
        enable_tool_injection: bool = False,
        enable_persistent_memory: bool = False,
        enable_llm_proxy: bool | None = None,
        provider_base_url: str = "",
        provider_api_key: str = "",
        managed: bool = False,
        managed_runtime: str = "local",
        reset_session_between_runs: bool = False,
        managed_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._gateway_url = gateway_url
        self._auth_token = auth_token
        self._session_key = session_key
        self._agent_id = agent_id
        self._model_id = model_id
        self._agent_timeout_s = agent_timeout_s
        self._reset_session_between_runs = reset_session_between_runs
        self._enable_tool_injection = enable_tool_injection
        self._enable_persistent_memory = enable_persistent_memory
        # The LLM proxy is on by default whenever a provider is configured
        # (it's the only way to inject model responses, track usage, and
        # enumerate models). It auto-disables when no provider is given.
        self._enable_llm_proxy = (
            enable_llm_proxy if enable_llm_proxy is not None else bool(provider_base_url)
        )
        self._provider_base_url = provider_base_url
        self._provider_api_key = provider_api_key
        self._managed = managed
        self._managed_runtime = managed_runtime
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
        # Shared secrets that authenticate the gateway (and its in-container
        # plugin) to the host-side callback server / LLM proxy. Generated when
        # those servers start; empty until then.
        self._callback_token: str = ""
        self._proxy_token: str = ""

        # Live run state — set at the top of run(), cleared at the end.
        # Used by the injection hook to dispatch ControllablePreCallEvents
        # into the live trajectory.
        self._active_send_event: EventResponseHandler | None = None

    # ------------------------------------------------------------------
    # Lazy connection
    # ------------------------------------------------------------------

    @property
    def _is_docker(self) -> bool:
        """Whether the managed gateway runs in a container."""
        return self._managed and self._managed_runtime == "docker"

    def _container_host(self) -> str:
        """Hostname the gateway uses to reach host-side services.

        A local (loopback) gateway reaches the host injection server / LLM proxy
        directly on ``127.0.0.1``; a containerised gateway reaches them via
        ``host.docker.internal`` (mapped with ``--add-host`` in the Docker
        runtime). For an external (unmanaged) gateway we assume loopback.
        """
        return "host.docker.internal" if self._is_docker else "127.0.0.1"

    def _bind_host(self) -> str:
        """Interface the host-side servers (injection server + LLM proxy) bind to.

        A containerised gateway reaches the host via ``host.docker.internal``,
        which on Docker Desktop resolves through a VM — so loopback is not
        reachable and the servers must listen on a non-loopback interface
        (``0.0.0.0``). To avoid that being an *unauthenticated* relay, both
        servers require a per-instance bearer token (the plugin sends
        ``SUPERRED_CALLBACK_TOKEN``; the gateway sends the proxy token as its
        provider ``apiKey``), so an unauthenticated caller on another interface
        is rejected with 401. A local gateway keeps everything on ``127.0.0.1``.
        """
        return "0.0.0.0" if self._is_docker else "127.0.0.1"

    async def _ensure_connected(self) -> OpenClawWSClient:
        if self._client is not None:
            return self._client

        # Start the injection server before the gateway so the managed
        # runtime can install the plugin pointed at its callback URL.
        if self._enable_tool_injection and self._injection_server is None:
            await self._start_injection_server()

        # Start the LLM proxy before the gateway so the managed runtime can
        # point the gateway's provider base URL at it (otherwise model calls
        # bypass the proxy and we lose response injection / usage tracking).
        if self._enable_llm_proxy and self._llm_proxy is None:
            await self._start_llm_proxy()

        if self._managed and self._runtime is None:
            self._runtime = self._build_runtime()
            await self._runtime.start()
            self._gateway_url = self._runtime.gateway_url
            self._auth_token = self._runtime.auth_token

        client = OpenClawWSClient(
            gateway_url=self._gateway_url,
            auth_token=self._auth_token,
            use_device_identity=(
                self._runtime.use_device_identity
                if self._runtime is not None
                else False
            ),
            device_identity_path=(
                self._runtime.device_identity_path
                if self._runtime is not None
                else None
            ),
        )
        self._hello_payload = await client.connect()
        self._client = client

        try:
            catalog = await client.rpc("tools.catalog")
            self._cached_tool_catalog = json.dumps(catalog, indent=2)
        except Exception:
            self._cached_tool_catalog = "{}"

        return client

    async def warmup_static_observables(self) -> None:
        """Connect and cache live static facts before optimizer initialization.

        The controller calls :meth:`get_observables` synchronously immediately
        after :meth:`~superred.core.interfaces.task.Task.configure_target`.
        The gateway tool catalogue is only available post-connect, so tasks
        should ``await target.warmup_static_observables()`` from
        ``configure_target`` when the optimizer needs the catalogue at init
        (in addition to the per-run emission in :meth:`run`).
        """
        await self._ensure_connected()

    def _build_runtime(self) -> Any:
        """Construct the managed runtime (local or Docker) with grounded config.

        Provider routing, tool policy, and the injection extension are passed as
        config (written to ``openclaw.json`` / installed under
        ``<stateDir>/extensions``) — not as env vars. The provider base URL and
        callback URL are expressed via :meth:`_container_host` so a containerised
        gateway can reach the host-side proxy / injection server.
        """
        kwargs: dict[str, Any] = dict(self._managed_kwargs)
        kwargs.setdefault("model_id", self._model_id)
        if self._injection_server is not None:
            kwargs.setdefault("plugin_dir", str(_plugin_dir()))
            kwargs.setdefault("callback_url", self._callback_url())
            # The in-container plugin authenticates to the callback server with
            # this token (exported as SUPERRED_CALLBACK_TOKEN).
            kwargs.setdefault("callback_token", self._callback_token)
        if self._tool_policy:
            kwargs.setdefault("tool_policy", self._tool_policy)
        # Route the gateway's model calls through the proxy when active;
        # otherwise straight at the configured provider.
        if self._llm_proxy is not None:
            provider_url = (
                f"http://{self._container_host()}:{self._llm_proxy.actual_port}"
            )
            # The gateway authenticates to the proxy with the proxy token (sent
            # as the OpenAI ``Authorization: Bearer`` header because it is the
            # provider apiKey); the proxy forwards upstream with the real key.
            # Never write the real provider key into the gateway config in this
            # path.
            kwargs.setdefault("provider_api_key", self._proxy_token)
            from openclaw_target.config import PROVIDER_ENV_API_KEYS, _split_model_id

            provider, _ = _split_model_id(self._model_id or "")
            env_keys = PROVIDER_ENV_API_KEYS.get(provider)
            if env_keys:
                kwargs.setdefault("unset_env_keys", env_keys)
        else:
            provider_url = self._provider_base_url or None
            if self._provider_api_key:
                kwargs.setdefault("provider_api_key", self._provider_api_key)
        if provider_url:
            kwargs.setdefault("provider_base_url", provider_url)

        if self._is_docker:
            from openclaw_target.docker_runtime import OpenClawDockerRuntime

            return OpenClawDockerRuntime(**kwargs)

        from openclaw_target.runtime import OpenClawRuntime

        return OpenClawRuntime(**kwargs)

    def _callback_url(self) -> str:
        """URL the gateway-side plugin posts hook callbacks to.

        Built from :meth:`_container_host` so a containerised gateway reaches
        the host injection server via ``host.docker.internal`` while a local
        gateway uses loopback.
        """
        port = self._injection_server.actual_port if self._injection_server else 18899
        return f"http://{self._container_host()}:{port}"

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

        self._callback_token = secrets.token_hex(24)
        self._injection_server = InjectionServer(
            handler=self._handle_injection_hook,
            host=self._bind_host(),
            port=0,
            auth_token=self._callback_token,
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
        """Bridge plugin HTTP callbacks to live optimizer controllables.

        ``tool_result_middleware`` → live same-turn controllables.
        ``before_tool_call`` → transcript-poison controllables (stashed for
        the sync ``tool_result_persist`` hook on the plugin side).
        """
        send_event = self._active_send_event
        if send_event is None:
            return None

        if hook_type == "tool_result_middleware":
            controllable = self._live_controllable_for_tool(tool_name)
        elif hook_type == "before_tool_call":
            controllable = self._transcript_controllable_for_tool(tool_name)
        else:
            return None

        if controllable is None:
            return None

        request_payload = json.dumps(
            {
                "hook": hook_type,
                "tool": tool_name,
                "toolCallId": tool_call_id,
                "params": params,
                **(
                    {"result": result}
                    if hook_type == "tool_result_middleware"
                    else {}
                ),
            },
            default=str,
        )

        try:
            response = await send_event(
                ControllablePostCallEvent(
                    controllable=controllable,
                    request=request_payload,
                    answer=str(result) if result is not None else "",
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

        return {"toolResult": response.value}

    def _live_controllable_for_tool(self, tool_name: str) -> Controllable | None:
        return TOOL_OUTPUT_LIVE_CONTROLLABLES.get(tool_name)

    def _transcript_controllable_for_tool(self, tool_name: str) -> Controllable | None:
        return TOOL_OUTPUT_TRANSCRIPT_CONTROLLABLES.get(tool_name)

    def _controllable_for_tool(self, tool_name: str) -> Controllable | None:
        """Back-compat: live same-turn lookup."""
        return self._live_controllable_for_tool(tool_name)

    @staticmethod
    def _tool_name_from_payload(payload: dict[str, Any]) -> str:
        """Best-effort tool name from a ``session.tool`` event payload."""
        for key in ("toolName", "tool", "name"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

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

        self._proxy_token = secrets.token_hex(24)
        self._llm_proxy = LLMProxy(
            upstream_base_url=self._provider_base_url,
            upstream_api_key=self._provider_api_key,
            host=self._bind_host(),
            inbound_token=self._proxy_token,
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
                    "agent workspace before each run. ``agents.files.set`` "
                    "only accepts the fixed bootstrap/memory filenames in "
                    f"{sorted(ALLOWED_WORKSPACE_BOOTSTRAP_FILES)} "
                    "(gateway-enforced allowlist, src/gateway/server-methods/"
                    "agents.ts ALLOWED_FILE_NAMES); any other name is "
                    "rejected by the gateway and silently skipped (logged)."
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
            live = dict.fromkeys(TOOL_OUTPUT_LIVE_CONTROLLABLES.values())
            transcript = dict.fromkeys(TOOL_OUTPUT_TRANSCRIPT_CONTROLLABLES.values())
            ctrls.extend(live)
            ctrls.extend(transcript)
        if self._enable_persistent_memory:
            ctrls.append(PERSISTENT_MEMORY_CTRL)
        if self._enable_llm_proxy:
            ctrls.append(MODEL_SYSTEM_PROMPT_CTRL)
            ctrls.append(MODEL_RESPONSE_CTRL)
        return ctrls

    def get_observables(self) -> list[ObservableValue]:
        # Static observables are derived from configuration so they are
        # populated at optimizer-initialization time (before the first run),
        # matching the framework convention used by the other agentic targets
        # (agentdojo / inspect_agent). The live gateway tool catalog is
        # cached on connect; before that it is empty.
        obs = [
            ObservableValue(
                observable=MODEL_IDENTITY_OBS,
                content=self._model_id,
            ),
            ObservableValue(
                observable=SYSTEM_DESC_OBS,
                content=json.dumps({
                    "gateway_url": self._gateway_url,
                    "session_key": self._session_key,
                    "agent_id": self._agent_id,
                    "model_id": self._model_id,
                    "tool_policy": self._tool_policy or None,
                    "managed": self._managed,
                    "managed_runtime": self._managed_runtime if self._managed else None,
                    "hello": self._hello_payload or None,
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

        # Emit the live gateway tool catalogue as a run observable. It is only
        # available after connecting (fetched in _ensure_connected), so
        # get_observables() — read once at optimizer init, before any run —
        # cannot carry it. The tool catalogue is the most attack-relevant
        # static fact, so surface it here where the optimizer will see it.
        if self._cached_tool_catalog and self._cached_tool_catalog != "{}":
            emit(ObservableEvent(
                observable=TOOL_LIST_OBS,
                content=self._cached_tool_catalog,
            ))

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

            # Model-response injection: rewrite the model's reply before the
            # agent acts on it (a manipulated/compromised model output).
            # Resolved pre-run; the proxy applies it to every response.
            resp_inj = await send_event(
                ControllablePreCallEvent(
                    controllable=MODEL_RESPONSE_CTRL,
                    request="Content to inject into the LLM response text:",
                ),
            )
            if isinstance(resp_inj, ControllableInjection) and resp_inj.value:
                self._llm_proxy.response_injection = resp_inj.value
            else:
                self._llm_proxy.response_injection = None

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
        # The user message is recorded by the framework from the
        # USER_MESSAGE_CTRL controllable above; we do NOT also emit it as an
        # observable (it would double-record the same content).

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
                # Tool calls that are injection points are recorded via their
                # ControllablePostCallEvent (its request carries the call), so
                # don't also emit them as observables. Non-injection tool calls
                # have no controllable, so they are surfaced here.
                tool_name = self._tool_name_from_payload(evt.payload)
                if (
                    self._enable_tool_injection
                    and self._controllable_for_tool(tool_name) is not None
                ):
                    return
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

        if self._enable_persistent_memory and result.status == "ok":
            await self._apply_persistent_memory(client, result, send_event)

        if result.error:
            logger.warning("Agent run error: %s", result.error)

    # ------------------------------------------------------------------
    # Lifecycle: per-run reset and teardown
    # ------------------------------------------------------------------

    async def reset_ephemeral_state(self) -> None:
        """Clear only per-run (ephemeral) state between runs of the same task.

        Ephemeral state = the last-run response/tool-call/event buffers and
        any recorded proxy calls. Durable task state is intentionally
        preserved per the framework contract (``Target.reset_ephemeral_state``:
        durable state must survive this call and is discarded only via a fresh
        ``TargetFactory`` instance between tasks):

        - Planted files / AGENTS.md remain (cleaned up in :meth:`teardown`).
        - The OpenClaw conversation/session is kept, since OpenClaw sessions
          are durable; wiping them would lose intended context and break
          poison-then-trigger attacks. Opt into per-run conversation isolation
          with ``reset_session_between_runs=True``.
        """
        self._last_response = ""
        self._last_tool_calls = []
        self._last_events_json = "[]"
        if self._llm_proxy:
            self._llm_proxy.records.clear()
            self._llm_proxy.system_prompt_injection = None
            self._llm_proxy.response_injection = None

        if self._reset_session_between_runs and self._client:
            # Surface a failed reset: a swallowed error here means the
            # conversation silently persists across runs, which corrupts
            # per-run isolation (the whole point of opting in).
            try:
                await self._client.reset_session(self._session_key)
            except Exception:
                logger.warning(
                    "sessions.reset failed; conversation may persist across "
                    "runs (per-run isolation not guaranteed)",
                    exc_info=True,
                )

    async def teardown(self) -> None:
        """Close the connection, injection server, proxy, and gateway process.

        Best-effort clears files planted during this task before closing the
        connection. For a managed gateway the process is destroyed anyway;
        this matters for an external gateway shared across tasks.
        """
        if self._client and self._planted_files:
            # The protocol exposes agents.files.set (no files.delete); clear a
            # planted file by overwriting it with empty content.
            for filename in self._planted_files:
                try:
                    result = await self._client.rpc(
                        "agents.files.set",
                        {"agentId": self._agent_id, "name": filename, "content": ""},
                    )
                    if isinstance(result, dict) and result.get("error"):
                        logger.debug(
                            "Could not clear planted file %s: %s",
                            filename, result["error"],
                        )
                except Exception:
                    logger.debug("Could not clear planted file %s", filename, exc_info=True)
            self._planted_files.clear()

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
        bootstrap workspace file. ``rpc`` returns an ``{"error": ...}`` dict
        rather than raising on a gateway-level rejection (e.g. unknown
        ``agentId``), so that must be checked explicitly - an exception
        handler alone silently no-ops the injection.
        """
        try:
            result = await client.rpc("agents.files.set", {
                "agentId": self._agent_id,
                "name": "AGENTS.md",
                "content": self._system_prompt_append,
            })
            if isinstance(result, dict) and result.get("error"):
                logger.warning(
                    "agents.files.set(AGENTS.md) rejected by gateway: %s; "
                    "system prompt append will not take effect.",
                    result["error"],
                )
        except Exception:
            logger.warning(
                "Could not set AGENTS.md via RPC; system prompt append "
                "may not take effect.",
                exc_info=True,
            )

    async def _write_workspace_files(self, client: OpenClawWSClient) -> None:
        """Write workspace files before a run (for data injection scenarios).

        Uses ``agents.files.set`` ({agentId, name, content}). The gateway
        caps this RPC to a fixed allowlist of bootstrap/memory filenames
        (``ALLOWED_WORKSPACE_BOOTSTRAP_FILES`` -
        ``src/gateway/server-methods/agents.ts`` ``ALLOWED_FILE_NAMES`` -
        verified live: any other name is rejected with
        ``INVALID_REQUEST: unsupported file "<name>"``). It cannot plant
        arbitrary nested paths; that requires a shared workspace dir on a
        real gateway (see runtime ``workspace_dir``).
        """
        for filename, content in self._workspace_files.items():
            try:
                result = await client.rpc("agents.files.set", {
                    "agentId": self._agent_id,
                    "name": filename,
                    "content": content,
                })
                if isinstance(result, dict) and result.get("error"):
                    logger.warning(
                        "agents.files.set(%s) rejected by gateway: %s; "
                        "file was not planted.",
                        filename, result["error"],
                    )
                    continue
                self._planted_files.append(filename)
            except Exception:
                logger.warning(
                    "Could not write workspace file %s via RPC",
                    filename,
                    exc_info=True,
                )

    async def _apply_persistent_memory(
        self,
        client: OpenClawWSClient,
        result: AgentRunResult,
        send_event: EventResponseHandler,
    ) -> None:
        """Offer the optimizer one grouped, end-of-run memory-poisoning edit.

        Fires once per :meth:`run`, after the agent finishes, with the full
        session ``chat.history`` as context — the optimizer sees everything
        that happened in the run, not just a single tool call, before
        deciding what (if anything) to persist. Unlike the rejected
        ``before_prompt_build`` / ``enqueueNextTurnInjection`` approach, the
        returned value is written via the real ``agents.files.set`` RPC into
        ``MEMORY.md``: a durable file that survives ``sessions.reset`` and a
        fresh session, not an ephemeral prompt-context queue.
        """
        try:
            history = await client.get_session_history(self._session_key)
        except Exception:
            logger.debug("Could not fetch session history for persistent_memory", exc_info=True)
            history = []

        current_memory = ""
        try:
            got = await client.rpc(
                "agents.files.get",
                {"agentId": self._agent_id, "name": "MEMORY.md"},
            )
            if isinstance(got, dict) and not got.get("error"):
                current_memory = got.get("file", {}).get("content", "") or ""
        except Exception:
            logger.debug("Could not fetch current MEMORY.md for persistent_memory", exc_info=True)

        request_payload = json.dumps(
            {
                "session_history": history,
                "assistant_text": result.assistant_text,
                "tool_calls": result.tool_calls,
            },
            default=str,
        )

        try:
            response = await send_event(
                ControllablePostCallEvent(
                    controllable=PERSISTENT_MEMORY_CTRL,
                    request=request_payload,
                    answer=current_memory,
                ),
            )
        except Exception:
            logger.exception("send_event failed while emitting persistent_memory")
            return

        if not isinstance(response, ControllableInjection) or not response.value:
            return

        try:
            write_result = await client.rpc("agents.files.set", {
                "agentId": self._agent_id,
                "name": "MEMORY.md",
                "content": response.value,
            })
            if isinstance(write_result, dict) and write_result.get("error"):
                logger.warning(
                    "agents.files.set(MEMORY.md) rejected by gateway: %s; "
                    "persistent memory poisoning will not take effect.",
                    write_result["error"],
                )
                return
        except Exception:
            logger.warning(
                "Could not write MEMORY.md via RPC for persistent_memory",
                exc_info=True,
            )
            return

        if "MEMORY.md" not in self._planted_files:
            self._planted_files.append("MEMORY.md")
