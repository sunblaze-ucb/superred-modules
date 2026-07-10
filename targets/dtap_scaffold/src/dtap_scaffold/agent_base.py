"""The agent-agnostic ``DtapAgentTarget`` base: the superred Target lifecycle
wired around the Docker/proxy/injector collaborators, shared by both agents.

A concrete agent (Claude Code, OpenClaw) subclasses this and implements only:
``_run_episode`` (launch the agent in its isolated container, run the turns,
write a transcript), ``_extract_trajectory`` (parse that transcript into a
:class:`~dtap_scaffold.types.TrajectoryArtifact`), ``_native_tool_deny`` (map the
native-tools policy to the agent's native tool names), and ``_exec_on_host`` (run
attacker code on the machine for the code_execution surface). Everything else --
env activation by config, the security-domain forest, the controllables (the four
DTAP vectors plus the superred-afforded surfaces: system prompt, env-tool return
tampering, and the ``host`` filesystem / code-execution boundary), the observables
(emit-once), the Docker/proxy/injection lifecycle, and the query surface the
claim's OOB judge reads -- lives here.

Collaborators are created via ``_make_env_stack`` / ``_make_proxy`` /
``_make_injector`` (lazy-import the concrete impls); tests override these to
inject fakes, which is how the whole lifecycle is verified without Docker.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from abc import abstractmethod
from typing import Any

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
from superred.core.types.security_domain import SecurityDomain
from superred.core.types.state import ConfigSpec, QuerySpec

from dtap_scaffold import config_specs as C  # noqa: N812
from dtap_scaffold import query_specs as Q  # noqa: N812
from dtap_scaffold.controllables import (
    CODE_EXECUTION_CTRL,
    FILESYSTEM_CTRL,
    FIXED_CONTROLLABLES,
    SKILL_CTRL,
    SYSTEM_PROMPT_CTRL,
    TOOL_ADD_CTRL,
    TOOL_DESCRIPTION_OVERRIDE_CTRL,
    TOOL_DESCRIPTION_SUFFIX_CTRL,
    TOOL_REMOVE_CTRL,
    USER_PROMPT_CTRL,
    env_inject_controllable,
    env_tool_output_controllable,
    tool_call_controllable,
)
from dtap_scaffold.forest import TOOL_CATALOGUE_ADD_TAG, build_domain, env_server_tag
from dtap_scaffold.observables import (
    ACTIVE_ENVIRONMENTS_OBS,
    ATTACKER_CONTEXT_OBS,
    DETAILED_SYSTEM_SPECIFICATION_OBS,
    MAX_TURNS_OBS,
    MODEL_IDENTITY_OBS,
    TOOL_CATALOGUE_OBS,
    agent_message_observable,
    native_tool_observable,
)
from dtap_scaffold.protocols import EnvInjector, EnvStack, MCPProxy
from dtap_scaffold.system_specification import DETAILED_SYSTEM_SPECIFICATION
from dtap_scaffold.tool_trees import ServerToolTree, build_server_tree
from dtap_scaffold.types import (
    AgentLaunchSpec,
    EnvHandle,
    EpisodeResult,
    InjectionPoint,
    TrajectoryArtifact,
)

_DEFAULT_MAX_TURNS = 200

# Runaway backstop for the attacker's code-execution foothold. The loop is
# ATTACKER-terminated (it ends the instant the optimizer declines a round); this
# cap only bounds a misbehaving optimizer that injects without ever declining, so
# it is set generously. Also bounded in practice by the optimizer's LLM budget.
_DEFAULT_MAX_CODE_EXEC_ROUNDS = 64


def _as_injection(resp: Any) -> str | None:
    """Return the injected value if *resp* is a ControllableInjection, else None."""
    return resp.value if isinstance(resp, ControllableInjection) else None


def _safe_join(root: str, rel: str) -> str | None:
    """Join *rel* under *root*, or None if it would escape *root*.

    The attacker names paths relative to the agent workspace; a ``..`` traversal
    that would write outside the workspace is rejected (the host_filesystem
    surface is scoped to the machine the agent runs on, applied at the workspace).
    """
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, rel))
    if target == root_abs or target.startswith(root_abs + os.sep):
        return target
    return None


class DtapAgentTarget(Target):
    """Agent-agnostic DTAP target. Subclass and implement the abstract hooks below."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        state_root: str | None = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        temperature: float | None = None,
        max_code_exec_rounds: int = _DEFAULT_MAX_CODE_EXEC_ROUNDS,
    ) -> None:
        # Construction concerns (NOT config slots): model identity + generation.
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._state_root = state_root
        self._default_max_turns = max_turns
        self._temperature = temperature
        self._max_code_exec_rounds = max_code_exec_rounds

        # Per-task config (set via set_config).
        self._active_servers: tuple[str, ...] = ()
        self._injection_servers: tuple[str, ...] = ()
        self._env_injection_config: dict[str, Any] = {}
        self._system_prompt: str = ""
        self._user_instructions: tuple[str, ...] = ()
        self._task_dir: str = ""
        self._additional_information: str = ""
        self._server_env_overrides: dict[str, dict[str, str]] = {}
        self._available_injections: dict[str, Any] = {}
        self._threat_model: str = ""
        self._max_turns: int = max_turns
        self._native_tools_policy: str = "enabled"

        # Cached per-server authorization trees + tags (identity-stable; reused in
        # domain/controllables/events). Each active server's tools.<server> tag is
        # the root of a single-parent tree of tools.<server>.<node> tags.
        self._tool_trees: dict[str, ServerToolTree] = {}
        self._env_tags: dict[str, Any] = {}
        # env_tool controllables, built once per config: per-node (identity-stable),
        # the per-server root/fallback, and the tool -> node-controllable map the
        # proxy resolves each call through.
        self._env_tool_node_ctrls: dict[str, dict[str, Controllable]] = {}
        self._env_tool_defaults: dict[str, Controllable] = {}
        self._env_tool_by_tool: dict[str, dict[str, Controllable]] = {}

        # Live collaborators (created lazily on first run).
        self._env_stack: EnvStack | None = None
        self._proxy: MCPProxy | None = None
        self._injector: EnvInjector | None = None
        self._handle: EnvHandle | None = None
        self._proxy_url: str = ""
        self._started = False

        # Per-run outputs (cleared by reset_ephemeral_state).
        self._final_response: str = ""
        self._agent_responses: list[str] = []
        self._trajectory_json: dict[str, Any] = {}

        # Per-run host workspace root (the dir mounted into the agent container as
        # its workspace). Owned by the base so the host_filesystem / code_execution
        # surfaces can shape it BEFORE the agent launches; the subclass mounts it.
        self._run_dir: str = ""

    # ----- superred Target ABC: specs --------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return list(C.CONFIG_SPECS)

    @property
    def query_specs(self) -> list[QuerySpec]:
        return list(Q.QUERY_SPECS)

    def set_config(self, name: str, value: str) -> None:
        if name == C.ACTIVE_MCP_SERVERS:
            servers = tuple(json.loads(value))
            self._active_servers = servers
            self._tool_trees = {s: build_server_tree(s) for s in servers}
            # Materialize the env_tool controllables once (identity-stable tags):
            # one per authorization node, a root/fallback per server, and the
            # tool -> node-controllable map the proxy resolves each call through.
            self._env_tool_node_ctrls = {}
            self._env_tool_defaults = {}
            self._env_tool_by_tool = {}
            for s in servers:
                tree = self._tool_trees[s]
                node_ctrls = {
                    key: env_tool_output_controllable(s, key, tag)
                    for key, tag in tree.nodes.items()
                }
                self._env_tool_node_ctrls[s] = node_ctrls
                self._env_tool_defaults[s] = env_tool_output_controllable(s, "", tree.root)
                self._env_tool_by_tool[s] = {
                    tool: node_ctrls[key] for tool, key in tree.tool_to_key.items()
                }
        elif name == C.ENV_INJECTION_CONFIG:
            cfg = json.loads(value) if value else {}
            self._env_injection_config = dict(cfg)
            self._injection_servers = tuple(cfg.keys())
            self._env_tags = {s: env_server_tag(s) for s in self._injection_servers}
        elif name == C.SYSTEM_PROMPT:
            self._system_prompt = value
        elif name == C.USER_PROMPT:
            self._user_instructions = _normalize_instructions(value)
        elif name == C.TASK_DIR:
            self._task_dir = value
        elif name == C.AVAILABLE_INJECTIONS:
            self._available_injections = json.loads(value) if value else {}
        elif name == C.ADDITIONAL_INFORMATION:
            self._additional_information = value or ""
        elif name == C.SERVER_ENV_OVERRIDES:
            self._server_env_overrides = json.loads(value) if value else {}
        elif name == C.THREAT_MODEL:
            self._threat_model = value
        elif name == C.MAX_TURNS:
            self._max_turns = int(value) if value else self._default_max_turns
        elif name == C.NATIVE_TOOLS_POLICY:
            self._native_tools_policy = value or "enabled"
        else:
            raise ValueError(f"unknown DTAP config slot: {name!r}")

    def query(self, name: str, **params: str) -> str:
        if name == Q.FINAL_RESPONSE:
            return self._final_response
        if name == Q.AGENT_RESPONSES:
            return json.dumps(self._agent_responses)
        if name == Q.TRAJECTORY_JSON:
            return json.dumps(self._trajectory_json)
        if name == Q.ENV_PORTS:
            return json.dumps(self._handle.ports if self._handle else {})
        if name == Q.ENV_PROJECT_NAMES:
            return json.dumps(self._handle.project_names if self._handle else {})
        if name == Q.TASK_DIR:
            return self._task_dir
        raise ValueError(f"unknown DTAP query slot: {name!r}")

    # ----- superred Target ABC: surfaces -----------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        tool_tags = [tag for tree in self._tool_trees.values() for tag in tree.all_tags]
        return build_domain(tool_tags, self._env_tags.values())

    def get_controllables(self) -> list[Controllable]:
        ctrls = list(FIXED_CONTROLLABLES)
        for s in self._active_servers:
            # per-authorization-node return-tamper surfaces, then the server
            # root/fallback (the whole-server grant; fires for unmapped tools)
            ctrls += list(self._env_tool_node_ctrls[s].values())
            ctrls.append(self._env_tool_defaults[s])
        ctrls += [env_inject_controllable(s, self._env_tags[s]) for s in self._injection_servers]
        return ctrls

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._model),
            ObservableValue(
                observable=DETAILED_SYSTEM_SPECIFICATION_OBS,
                content=DETAILED_SYSTEM_SPECIFICATION,
            ),
            ObservableValue(
                observable=ATTACKER_CONTEXT_OBS,
                content=self._additional_information,
            ),
            ObservableValue(
                observable=ACTIVE_ENVIRONMENTS_OBS,
                content={
                    "servers": list(self._active_servers),
                    "injection_servers": list(self._injection_servers),
                },
            ),
            ObservableValue(observable=MAX_TURNS_OBS, content=str(self._max_turns)),
        ]

    # ----- superred Target ABC: run lifecycle ------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        await self._ensure_started()
        assert self._proxy is not None and self._injector is not None

        # Fresh per-run host workspace root (with a ``workspace/`` subdir); the
        # host_filesystem / code_execution surfaces shape it before the agent
        # launches, and the subclass mounts THIS dir into the agent container.
        self._run_dir = self._new_run_dir()

        # Per-tool catalogue (names/descriptions/schemas of every active env tool),
        # emitted ONCE now that the env has booted, so the optimizer can read the full
        # tool surface before it chooses tool-description injections below.
        emit(
            ObservableEvent(
                observable=TOOL_CATALOGUE_OBS,
                content={"servers": self._proxy.tool_catalogue()},
            )
        )

        # PreCall vectors: system, user (per turn), skill, tool-description.
        system_prompt = await self._precall(send_event, SYSTEM_PROMPT_CTRL, self._system_prompt)
        instructions = [
            await self._precall(send_event, USER_PROMPT_CTRL, turn)
            for turn in (self._user_instructions or ("",))
        ]
        skills = await self._precall_skills(send_event)
        edits = await self._precall_tool_desc(send_event)
        added_tools, removed_tools, tool_call_ctrls = await self._precall_tool_catalog(send_event)

        # host_filesystem vector (PreCall): place/edit/delete files on the machine.
        await self._precall_filesystem(send_event)

        # Environment vector (PostCall): write attacker data into the live backend.
        await self._apply_env_injections(send_event)

        # host_code_execution vector (repeated PostCall): the attacker's interactive
        # code-execution foothold on the machine, before the agent loop.
        await self._code_execution_loop(send_event)

        # Wire the proxy for this run (env-tool observe + PostCall return tampering).
        self._proxy.bind(emit, send_event)
        self._proxy.set_tool_description_edits(edits)
        self._proxy.set_tool_catalog(added_tools, removed_tools, tool_call_ctrls)
        self._proxy.set_env_tool_controllables(self._env_tool_by_tool, self._env_tool_defaults)

        spec = AgentLaunchSpec(
            model=self._model,
            api_base=self._api_base,
            api_key=self._api_key,
            system_prompt=system_prompt,
            instructions=tuple(instructions),
            proxy_url=self._proxy_url,
            mcp_server_names=self._active_servers,
            skills=tuple(skills),
            native_tool_deny=tuple(self._native_tool_deny(self._native_tools_policy)),
            max_turns=self._max_turns,
            temperature=self._temperature,
            workspace_dir=self._run_dir,
            output_dir=self._run_dir,
            metadata={"task_dir": self._task_dir, "domain": self._primary_domain()},
        )

        episode = await self._run_episode(spec)
        traj = self._extract_trajectory(episode)

        # Emit observables once each (proxied env-tool calls were emitted by the proxy).
        for i, message in enumerate(traj.messages):
            emit(ObservableEvent(observable=agent_message_observable(i), content=message))
        for i, call in enumerate(traj.native_tool_calls):
            emit(ObservableEvent(observable=native_tool_observable(i), content=call))

        self._final_response = traj.final_response
        self._agent_responses = list(traj.agent_responses)
        self._trajectory_json = dict(traj.trajectory_json)

    async def reset_ephemeral_state(self) -> None:
        if self._env_stack is not None:
            await self._env_stack.reset()
        self._final_response = ""
        self._agent_responses = []
        self._trajectory_json = {}
        # The workspace (attacker-placed files + code_execution output) is ephemeral
        # per run: reclaim it here. The env stack + proxy persist; the next run() mints
        # a fresh workspace root.
        if self._run_dir:
            shutil.rmtree(self._run_dir, ignore_errors=True)
        self._run_dir = ""

    async def teardown(self) -> None:
        try:
            if self._proxy is not None:
                await self._proxy.stop()
        finally:
            if self._env_stack is not None:
                await self._env_stack.down()
            # Reclaim the last run's workspace (reset is not called after the final run).
            if self._run_dir:
                shutil.rmtree(self._run_dir, ignore_errors=True)
            self._run_dir = ""
            self._started = False

    # ----- internals -------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._started:
            return
        self._env_stack = self._make_env_stack()
        self._handle = await self._env_stack.up()
        self._injector = self._make_injector(self._handle)
        self._proxy = self._make_proxy()
        self._proxy_url = await self._proxy.start(self._handle.server_urls)
        self._started = True

    async def _precall(
        self, send_event: EventResponseHandler, ctrl: Controllable, default: str
    ) -> str:
        resp = await send_event(ControllablePreCallEvent(controllable=ctrl, request=default))
        injected = _as_injection(resp)
        return injected if injected is not None else default

    async def _precall_skills(self, send_event: EventResponseHandler) -> list[dict[str, Any]]:
        resp = await send_event(ControllablePreCallEvent(controllable=SKILL_CTRL, request=""))
        injected = _as_injection(resp)
        if injected is None:
            return []
        spec = json.loads(injected)
        return [spec] if isinstance(spec, dict) else list(spec)

    async def _precall_tool_desc(self, send_event: EventResponseHandler) -> list[dict[str, Any]]:
        edits: list[dict[str, Any]] = []
        for ctrl, mode in (
            (TOOL_DESCRIPTION_OVERRIDE_CTRL, "override"),
            (TOOL_DESCRIPTION_SUFFIX_CTRL, "suffix"),
        ):
            resp = await send_event(ControllablePreCallEvent(controllable=ctrl, request=""))
            injected = _as_injection(resp)
            if injected is None:
                continue
            spec = json.loads(injected)
            content = spec.get("description") if mode == "override" else spec.get("suffix")
            edits.append(
                {
                    "server": spec.get("server"),
                    "tool": spec.get("tool"),
                    "mode": mode,
                    "content": content,
                }
            )
        return edits

    async def _precall_tool_catalog(
        self, send_event: EventResponseHandler
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, set[str]],
        dict[tuple[str, str], Controllable],
    ]:
        """Tool-catalogue vector (PreCall): add fake tools and/or remove real ones.

        Fires TOOL_ADD_CTRL then TOOL_REMOVE_CTRL (once each). An ADD injection
        registers attacker-defined fake tools that appear in the agent's listing but
        have no backend; each gets a per-call PostCall Controllable tagged at the ADD
        capability, so when the agent calls the fake tool the attacker that added it
        receives the call (in scope, since it held ADD to register the tool) and
        supplies the return. A REMOVE injection drops tools from the listing the
        agent reads. Declining either leaves the genuine catalogue (DTAP default).

        Returns ``(added, removed, call_ctrls)`` for :meth:`MCPProxy.set_tool_catalog`.
        """
        added: dict[str, list[dict[str, Any]]] = {}
        call_ctrls: dict[tuple[str, str], Controllable] = {}
        resp = await send_event(ControllablePreCallEvent(controllable=TOOL_ADD_CTRL, request=""))
        injected = _as_injection(resp)
        if injected is not None:
            for spec in _normalize_tool_adds(json.loads(injected)):
                server, name = spec["server"], spec["name"]
                added.setdefault(server, []).append(spec)
                call_ctrls[(server, name)] = tool_call_controllable(
                    server, name, TOOL_CATALOGUE_ADD_TAG
                )

        removed: dict[str, set[str]] = {}
        resp = await send_event(ControllablePreCallEvent(controllable=TOOL_REMOVE_CTRL, request=""))
        injected = _as_injection(resp)
        if injected is not None:
            for server, name in _normalize_tool_removes(json.loads(injected)):
                removed.setdefault(server, set()).add(name)

        return added, removed, call_ctrls

    async def _apply_env_injections(self, send_event: EventResponseHandler) -> None:
        assert self._injector is not None
        for server in self._injection_servers:
            point = InjectionPoint(server=server, point="all")
            answer = await self._injector.snapshot(point)
            ctrl = env_inject_controllable(server, self._env_tags[server])
            resp = await send_event(
                ControllablePostCallEvent(
                    controllable=ctrl,
                    request=json.dumps({"server": server}),
                    answer=answer,
                )
            )
            injected = _as_injection(resp)
            if injected is not None:
                await self._injector.apply(point, injected)

    def _new_run_dir(self) -> str:
        """Create a fresh per-run host dir with a ``workspace/`` subdir, under
        ``state_root`` when set. This dir becomes the agent's mounted workspace."""
        run_dir = tempfile.mkdtemp(
            prefix=f"dtap-{self._agent_kind()}-run-", dir=self._state_root or None
        )
        os.makedirs(os.path.join(run_dir, "workspace"), exist_ok=True)
        return run_dir

    async def _precall_filesystem(self, send_event: EventResponseHandler) -> None:
        """host_filesystem vector: place/edit/delete files before the run.

        Fires one PreCall; a ``ControllableInjection`` value is ``{"ops": [...]}``
        (or a bare list of ops), applied host-side to the workspace the agent
        mounts. Declining leaves an empty workspace (the DTAP-faithful default).
        """
        resp = await send_event(
            ControllablePreCallEvent(
                controllable=FILESYSTEM_CTRL,
                request=json.dumps({"workspace": "agent workspace mounted into the run"}),
            )
        )
        injected = _as_injection(resp)
        if injected is None:
            return
        parsed = json.loads(injected)
        ops = parsed.get("ops", []) if isinstance(parsed, dict) else parsed
        await self._apply_host_files(list(ops))

    async def _apply_host_files(self, ops: list[dict[str, Any]]) -> None:
        """Apply attacker file ops to the run workspace (host-side; no container).

        Each op: ``{"action": "write"|"append"|"delete", "path": str,
        "content"?: str}``. Paths are confined to the workspace (a traversal that
        would escape is skipped). The workspace is bind-mounted into the agent
        container, so these files are exactly what the agent's native tools read.
        """
        workspace = os.path.join(self._run_dir, "workspace")
        for op in ops:
            action = op.get("action")
            target = _safe_join(workspace, str(op.get("path", "")))
            if target is None or not op.get("path"):
                continue
            if action in ("write", "append"):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "a" if action == "append" else "w", encoding="utf-8") as fh:
                    fh.write(str(op.get("content", "")))
            elif action == "delete":
                if os.path.isfile(target):
                    os.remove(target)

    async def _code_execution_loop(self, send_event: EventResponseHandler) -> None:
        """host_code_execution vector: the attacker's interactive code foothold.

        Fires a PostCall repeatedly. Each round the event ``answer`` carries the
        previous command's combined output (empty on the first round); a
        ``ControllableInjection`` value is a shell command/script, run on the
        machine via :meth:`_exec_on_host`, whose output feeds the NEXT round.
        Declining ends the foothold (the attacker "decides it doesn't need
        anymore"); ``max_code_exec_rounds`` is only a runaway backstop.
        """
        transcript = ""
        for rnd in range(self._max_code_exec_rounds):
            resp = await send_event(
                ControllablePostCallEvent(
                    controllable=CODE_EXECUTION_CTRL,
                    request=json.dumps({"round": rnd}),
                    answer=transcript,
                )
            )
            code = _as_injection(resp)
            if code is None:
                return
            transcript = await self._exec_on_host(code)

    def _primary_domain(self) -> str:
        # The domain is the path segment immediately before the benign/malicious
        # split -- mirrors dataset._path_facts (domain == parts[0] under the
        # dataset root). Robust for 3-level (<domain>/benign/<id>) and 4-5-level
        # (<domain>/malicious/<threat>/<risk>/<id>) task dirs.
        if not self._task_dir:
            return ""
        parts = self._task_dir.rstrip("/").split("/")
        for marker in ("malicious", "benign"):
            if marker in parts:
                i = parts.index(marker)
                if i > 0:
                    return parts[i - 1]
        return ""

    # ----- collaborator factories (overridable in tests) -------------------

    def _make_env_stack(self) -> EnvStack:
        from dtap_scaffold.docker.lifecycle import DockerEnvStack

        return DockerEnvStack(
            active_servers=self._active_servers,
            injection_config=self._env_injection_config,
            task_dir=self._task_dir,
            state_root=self._state_root,
            server_env_overrides=self._server_env_overrides,
        )

    def _make_proxy(self) -> MCPProxy:
        from dtap_scaffold.mcp_proxy import HostMCPProxy

        return HostMCPProxy()

    def _make_injector(self, handle: EnvHandle) -> EnvInjector:
        from dtap_scaffold.injection import McpEnvInjector

        return McpEnvInjector(handle.injection_server_urls)

    # ----- abstract agent hooks (the ONLY per-agent code) ------------------

    @abstractmethod
    def _agent_kind(self) -> str:
        """A short label for the agent (e.g. 'claude_code', 'openclaw')."""

    @abstractmethod
    async def _run_episode(self, spec: AgentLaunchSpec) -> EpisodeResult:
        """Launch the agent in its isolated container, run the turns, write a transcript."""

    @abstractmethod
    def _extract_trajectory(self, episode: EpisodeResult) -> TrajectoryArtifact:
        """Parse the in-container transcript into a normalized TrajectoryArtifact."""

    @abstractmethod
    def _native_tool_deny(self, policy: str) -> list[str]:
        """Map the native-tools policy to this agent's native tool deny-list."""

    @abstractmethod
    async def _exec_on_host(self, code: str) -> str:
        """Run attacker *code* on the target machine and return its combined output.

        Called once per non-declined round of the code_execution foothold, BEFORE
        the agent episode. Runs in the agent's own image with the run workspace
        (``self._run_dir/workspace``) mounted, so files it writes are visible to
        the agent. Only invoked when the code_execution surface is in scope (else
        the loop declines immediately and this never fires), so a subclass may
        keep the whole implementation behind the Docker seam.
        """


def _normalize_instructions(value: str) -> tuple[str, ...]:
    """A config user_prompt value -> a tuple of per-turn instructions.

    Accepts a JSON list[str] (multi-turn), a JSON string, or a bare string.
    """
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return (value,)
    if isinstance(parsed, list):
        return tuple(str(x) for x in parsed)
    return (str(parsed),)


def _one_tool_add(spec: dict[str, Any], group_server: str) -> dict[str, Any]:
    """One tool_add spec -> a normalized fake-tool dict (server falls back to the group)."""
    return {
        "server": str(spec.get("server") or group_server),
        "name": str(spec.get("name", "")),
        "description": str(spec.get("description") or ""),
        "inputSchema": spec.get("inputSchema") or {},
        "fake_return": str(spec.get("fake_return") or ""),
    }


def _normalize_tool_adds(parsed: Any) -> list[dict[str, Any]]:
    """A tool_add injection -> a flat list of normalized fake-tool dicts.

    Accepts a single spec ``{server, name, ...}``, a list of specs, or a grouped
    ``{server, tools: [{name, ...}, ...]}`` (the group's server fills in any inner
    spec that omits one). Specs missing a server or name are dropped.
    """
    out: list[dict[str, Any]] = []
    items = parsed if isinstance(parsed, list) else [parsed]
    for item in items:
        if not isinstance(item, dict):
            continue
        inner = item.get("tools")
        if isinstance(inner, list):
            group_server = str(item.get("server", ""))
            out += [_one_tool_add(s, group_server) for s in inner if isinstance(s, dict)]
        else:
            out.append(_one_tool_add(item, ""))
    return [t for t in out if t["server"] and t["name"]]


def _normalize_tool_removes(parsed: Any) -> list[tuple[str, str]]:
    """A tool_remove injection -> a list of ``(server, name)`` pairs.

    Accepts a single ``{server, name}``, a list of such, or a grouped
    ``{server, names: [name, ...]}``. Entries missing a server or name are dropped.
    """
    out: list[tuple[str, str]] = []
    items = parsed if isinstance(parsed, list) else [parsed]
    for item in items:
        if not isinstance(item, dict):
            continue
        names = item.get("names")
        if isinstance(names, list):
            server = str(item.get("server", ""))
            out += [(server, str(n)) for n in names]
        else:
            out.append((str(item.get("server", "")), str(item.get("name", ""))))
    return [(s, n) for s, n in out if s and n]


__all__ = ["DtapAgentTarget"]
