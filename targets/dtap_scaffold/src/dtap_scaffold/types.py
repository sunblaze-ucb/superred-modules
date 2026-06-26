"""Shared data structures across the DTAP scaffolding and the two agent targets.

These are the FROZEN contracts that let the agent-agnostic base, the Docker/proxy
collaborators, and the two concrete agent drivers integrate. The base
(:class:`~dtap_scaffold.agent_base.DtapAgentTarget`) wires the superred lifecycle
around collaborators typed by :mod:`dtap_scaffold.protocols`; the concrete agents
implement only the abstract hooks, consuming :class:`AgentLaunchSpec` and
producing :class:`EpisodeResult` / :class:`TrajectoryArtifact`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvHandle:
    """Live handle to one task's started environment (returned by EnvStack.up).

    *server_urls* maps each active MCP server name to the URL the host MCP proxy
    forwards to. *injection_server_urls* maps each ``<server>-injection`` name to
    its URL. *ports* is the host port map (e.g. ``{"TRAVEL_PORT": 10312}``) the
    OOB judge subprocess needs to reach the live containers.
    """

    server_urls: dict[str, str]
    injection_server_urls: dict[str, str]
    ports: dict[str, int]
    network: str | None = None


@dataclass(frozen=True)
class ProxyTool:
    """An env MCP tool the proxy exposes (server + tool + agent-visible description)."""

    server: str
    tool: str
    description: str


@dataclass(frozen=True)
class InjectionPoint:
    """A DTAP environment-injection point (a ``<server>-injection`` tool)."""

    server: str  # the injection server name, e.g. "gmail-injection"
    point: str  # the inject_* tool, e.g. "inject_email", or "all"


@dataclass(frozen=True)
class AgentLaunchSpec:
    """Everything a concrete agent driver needs to run ONE episode in isolation.

    The agent connects to the host MCP proxy at *proxy_url* for the env tools
    (named ``mcp_server_names``), has its NATIVE tools enabled except
    *native_tool_deny*, loads *skills*, and runs *instructions* turn by turn under
    *max_turns*, writing its transcript under *output_dir* (a per-instance host
    path the driver mounts/uses). Tool-DESCRIPTION edits are applied by the proxy,
    not here.
    """

    model: str
    api_base: str | None
    api_key: str | None
    system_prompt: str
    instructions: tuple[str, ...]
    proxy_url: str
    mcp_server_names: tuple[str, ...]
    skills: tuple[dict[str, Any], ...] = ()  # {name, content, mode, row?}
    native_tool_deny: tuple[str, ...] = ()
    max_turns: int = 200
    temperature: float | None = None
    workspace_dir: str | None = None
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one agent episode (the driver's return)."""

    output_dir: str
    final_output: str = ""
    error: str | None = None
    duration: float = 0.0
    raw: Any = None


@dataclass(frozen=True)
class TrajectoryArtifact:
    """Normalized trajectory extracted from an agent's in-container transcript.

    Proxied env-tool calls are EXCLUDED (the proxy already emitted them); only the
    agent's native tool calls and its non-tool messages appear here, so the base
    can emit them once. *trajectory_json* is the DTAP-schema dict the upstream
    judge optionally consumes; *agent_responses* is the per-turn final-output list
    the judge's ``eval_task``/``eval_attack`` take.
    """

    native_tool_calls: tuple[dict[str, Any], ...] = ()
    messages: tuple[dict[str, Any], ...] = ()
    final_response: str = ""
    agent_responses: tuple[str, ...] = ()
    trajectory_json: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EnvHandle",
    "ProxyTool",
    "InjectionPoint",
    "AgentLaunchSpec",
    "EpisodeResult",
    "TrajectoryArtifact",
]
