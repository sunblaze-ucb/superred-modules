"""Parse an OpenClaw session-trajectory JSONL into a superred ``TrajectoryArtifact``.

OpenClaw, run headless with ``OPENCLAW_TRAJECTORY=1``, writes a per-session JSONL
trace (the "openclaw-trajectory" runtime schema): one JSON object per line, each a
runtime event with a ``type`` and a ``data`` payload. The richest event is
``model.completed``, whose ``data.messagesSnapshot`` is the cumulative message list
(roles ``user`` / ``assistant`` / ``toolResult``); an assistant message's
``content`` is a list of blocks of ``{type: "text"}`` or
``{type: "toolCall", id, name, arguments}``.

This converter mirrors upstream ``agent/openclaw/src/utils.py``
(``OpenClawTrajectoryConverter._convert_runtime_trajectory_entries`` /
``_append_message_steps``) for parsing, including its dedup of repeated assistant
texts / tool-call ids across the cumulative snapshots, but it emits the
agent-agnostic :class:`~dtap_scaffold.types.TrajectoryArtifact` rather than the
DT-Arena ``Trajectory`` object, and it **splits** tool calls:

* env / MCP tool calls (routed through the host proxy) are EXCLUDED from
  ``native_tool_calls`` because the proxy already emitted one ObservableEvent +
  one PostCall per call -- re-emitting them here would double-count;
* the agent's NATIVE tool calls (``exec`` / ``fs`` / ...) are KEPT.

A call is classified as env iff its name is attributable to one of the configured
MCP server names (see :func:`resolve_server`); everything else is native. The full
trace (env + native + messages) is still assembled into ``trajectory_json`` in the
DT-Arena schema, which the OOB judge optionally consumes.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from dtap_scaffold.types import TrajectoryArtifact

__all__ = [
    "convert",
    "find_session_log",
    "parse_entries",
    "resolve_server",
    "is_env_tool",
]

# Fallback "server" recorded for a native (non-MCP) OpenClaw tool call, matching
# upstream ``OpenClawTrajectoryConverter._extract_server_name``.
_NATIVE_SERVER = "openclaw"

# Separators OpenClaw / bundle-mcp may place between an MCP server name and the
# bare tool name when it surfaces an env tool to the model.
_SERVER_SEPARATORS = ("__", "_", ".", "-", ":", "/")


# --------------------------------------------------------------------------- #
# tool-name classification                                                    #
# --------------------------------------------------------------------------- #


def resolve_server(tool_name: str, mcp_servers: tuple[str, ...] | list[str]) -> str | None:
    """Return the MCP server *tool_name* is attributable to, or ``None`` if native.

    Recognizes upstream's static-plugin naming (``workspace_<Server>_<tool>``) and
    bundle-mcp prefixing (``<server><sep><tool>``, optionally ``mcp<sep>...`` /
    ``workspace<sep>...``) for any configured server name. A bare native name
    (``exec``, ``fs``, ...) matches nothing and yields ``None``.
    """
    servers = tuple(mcp_servers)
    parts = tool_name.split("_", 2)
    if len(parts) >= 3 and parts[0] == "workspace" and parts[1] in servers:
        return parts[1]
    for server in servers:
        for sep in _SERVER_SEPARATORS:
            prefixes = (
                f"{server}{sep}",
                f"mcp{sep}{server}{sep}",
                f"workspace{sep}{server}{sep}",
            )
            if any(tool_name.startswith(p) for p in prefixes):
                return server
    return None


def is_env_tool(tool_name: str, mcp_servers: tuple[str, ...] | list[str]) -> bool:
    """True iff *tool_name* belongs to a configured MCP server (so the proxy owns it)."""
    return resolve_server(tool_name, mcp_servers) is not None


# --------------------------------------------------------------------------- #
# DT-Arena trajectory_json builder (minimal, schema-faithful)                 #
# --------------------------------------------------------------------------- #


class _TrajBuilder:
    """Assembles a DT-Arena-schema ``trajectory_json`` dict (see upstream
    ``dt_arena/src/types/trajectory.py``); a tiny reimplementation so the port
    need not import the un-packaged ``dt_arena`` tree."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        meta = metadata or {}
        self.data: dict[str, Any] = {
            "task_info": {
                "task_id": meta.get("task_id", "unknown"),
                "original_instruction": meta.get("instruction", ""),
                "malicious_instruction": meta.get("malicious_goal", ""),
                "domain": meta.get("domain"),
                "risk_category": meta.get("category"),
            },
            "traj_info": {
                "step_count": 0,
                "actions_count": 0,
                "tool_count": 0,
                "user_turn": 0,
                "duration": 0.0,
                "timestamp": datetime.now(UTC).isoformat(),
                "agent_final_response": None,
                "metadata": {"framework": "openclaw", "trace_schema": "openclaw-trajectory"},
            },
            "trajectory": [],
        }
        self._next = 0

    def _step(self, step: dict[str, Any]) -> None:
        step["step_id"] = self._next
        self.data["trajectory"].append(step)
        self._next += 1
        self.data["traj_info"]["step_count"] = len(self.data["trajectory"])

    def user(self, message: str) -> None:
        self._step({"role": "user", "state": message, "metadata": {}})
        self.data["traj_info"]["user_turn"] += 1

    def agent_action(
        self, action: str, *, tool_name: str, tool_params: dict[str, Any], server: str
    ) -> None:
        self._step(
            {
                "role": "agent",
                "action": action,
                "metadata": {"tool_name": tool_name, "tool_params": tool_params, "server": server},
            }
        )
        self.data["traj_info"]["actions_count"] += 1

    def agent_message(self, text: str) -> None:
        self._step(
            {"role": "agent", "action": "send_message_to_user", "metadata": {"message": text}}
        )
        self.data["traj_info"]["actions_count"] += 1

    def tool_return(self, result: Any, *, tool_name: str, server: str) -> None:
        meta = {"tool_name": tool_name, "server": server}
        self._step({"role": "tool", "state": result, "metadata": meta})
        self.data["traj_info"]["tool_count"] += 1

    def set_final(self, response: str) -> None:
        self.data["traj_info"]["agent_final_response"] = response

    def set_meta(self, key: str, value: Any) -> None:
        self.data["traj_info"]["metadata"][key] = value


# --------------------------------------------------------------------------- #
# parsing                                                                      #
# --------------------------------------------------------------------------- #


def _extract_texts(content: Any) -> list[str]:
    """Text fragments from an OpenClaw content list (mirrors upstream)."""
    texts: list[str] = []
    if not isinstance(content, list):
        return texts
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text", "")).strip()
            if text:
                texts.append(text)
        elif isinstance(block, str):
            text = block.strip()
            if text:
                texts.append(text)
    return texts


def _format_action(tool_name: str, arguments: dict[str, Any]) -> str:
    """Human-readable ``tool(arg="...")`` action string (mirrors upstream)."""
    params = []
    for key, value in arguments.items():
        if isinstance(value, str):
            shown = value[:100] + "..." if len(value) > 100 else value
            params.append(f'{key}="{shown}"')
        else:
            params.append(f"{key}={value}")
    return f"{tool_name}({', '.join(params)})"


def _parse_tool_output(payload: Any) -> Any:
    """JSON-decode a string tool result when possible (mirrors upstream)."""
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload
    return payload


class _SnapshotResult:
    __slots__ = ("last_text",)

    def __init__(self, last_text: str | None) -> None:
        self.last_text = last_text


def _process_snapshot(
    messages: list[dict[str, Any]],
    *,
    mcp_servers: tuple[str, ...],
    builder: _TrajBuilder,
    out_messages: list[dict[str, Any]],
    out_native: list[dict[str, Any]],
    seen_users: set[str],
    seen_tool_ids: set[str],
    seen_texts: set[str],
) -> _SnapshotResult:
    """Append steps from one (cumulative) ``messagesSnapshot``; dedup across calls."""
    tool_results: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") == "toolResult":
            call_id = msg.get("toolCallId")
            if call_id:
                tool_results[call_id] = msg

    last_text: str | None = None
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            texts = _extract_texts(msg.get("content", []))
            if not texts:
                continue
            user_text = "\n".join(texts)
            if user_text in seen_users:
                continue
            seen_users.add(user_text)
            builder.user(user_text)
        elif role == "assistant":
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = str(block.get("text", "")).strip()
                    if not text:
                        continue
                    last_text = text
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)
                    out_messages.append({"role": "assistant", "text": text})
                    builder.agent_message(text)
                    builder.set_final(text)
                elif btype == "toolCall":
                    call_id = block.get("id", "")
                    if call_id and call_id in seen_tool_ids:
                        continue
                    if call_id:
                        seen_tool_ids.add(call_id)
                    tool_name = block.get("name", "unknown")
                    arguments = block.get("arguments", {}) or {}
                    server = resolve_server(tool_name, mcp_servers) or _NATIVE_SERVER
                    builder.agent_action(
                        _format_action(tool_name, arguments),
                        tool_name=tool_name,
                        tool_params=arguments,
                        server=server,
                    )
                    result_payload: Any = None
                    result_msg = tool_results.get(call_id)
                    if result_msg is not None:
                        parts = _extract_texts(result_msg.get("content", []))
                        result_payload = _parse_tool_output("\n".join(parts))
                        builder.tool_return(result_payload, tool_name=tool_name, server=server)
                    # Split: env/MCP calls already emitted by the proxy -> skip;
                    # keep native calls for the base to emit once.
                    if not is_env_tool(tool_name, mcp_servers):
                        out_native.append(
                            {
                                "tool": tool_name,
                                "arguments": arguments,
                                "id": call_id,
                                "server": server,
                                "result": result_payload,
                            }
                        )
    return _SnapshotResult(last_text)


def parse_entries(
    entries: list[dict[str, Any]],
    *,
    mcp_servers: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> TrajectoryArtifact:
    """Convert decoded openclaw-trajectory JSONL *entries* into a TrajectoryArtifact."""
    servers = tuple(mcp_servers)
    builder = _TrajBuilder(metadata or {})
    out_messages: list[dict[str, Any]] = []
    out_native: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    seen_tool_ids: set[str] = set()
    seen_texts: set[str] = set()

    agent_responses: list[str] = []
    turn_started = False
    current_turn_text: str | None = None
    last_assistant_text: str | None = None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event_type = entry.get("type")
        data = entry.get("data") or {}

        if event_type == "prompt.submitted":
            if turn_started:
                agent_responses.append(current_turn_text or "")
            turn_started = True
            current_turn_text = None
            prompt = data.get("prompt")
            if prompt and prompt not in seen_users:
                seen_users.add(prompt)
                builder.user(prompt)

        elif event_type == "context.compiled":
            system_prompt = data.get("systemPrompt")
            if system_prompt:
                builder.set_meta("system_prompt", system_prompt)

        elif event_type == "model.completed":
            snapshot = data.get("messagesSnapshot")
            if isinstance(snapshot, list):
                res = _process_snapshot(
                    snapshot,
                    mcp_servers=servers,
                    builder=builder,
                    out_messages=out_messages,
                    out_native=out_native,
                    seen_users=seen_users,
                    seen_tool_ids=seen_tool_ids,
                    seen_texts=seen_texts,
                )
                if res.last_text:
                    current_turn_text = res.last_text
                    last_assistant_text = res.last_text
            else:
                joined = _join_texts(data.get("assistantTexts"))
                if joined:
                    current_turn_text = joined
                    last_assistant_text = joined

        elif event_type == "session.ended":
            builder.set_meta("final_status", data.get("status"))

    if turn_started:
        agent_responses.append(current_turn_text or "")
    final_response = last_assistant_text or ""
    if not agent_responses and final_response:
        agent_responses = [final_response]

    return TrajectoryArtifact(
        native_tool_calls=tuple(out_native),
        messages=tuple(out_messages),
        final_response=final_response,
        agent_responses=tuple(agent_responses),
        trajectory_json=builder.data,
    )


def _join_texts(texts: Any) -> str | None:
    """Normalize an ``assistantTexts`` array into one response (mirrors upstream)."""
    if not texts or not isinstance(texts, list):
        return None
    parts = [str(t).strip() for t in texts if str(t).strip()]
    return "\n\n".join(parts) if parts else None


# --------------------------------------------------------------------------- #
# file location + top-level entry point                                       #
# --------------------------------------------------------------------------- #


def find_session_log(output_dir: str) -> str | None:
    """Locate the OpenClaw session-trajectory JSONL beneath *output_dir*.

    The in-container driver writes the trace under ``<state>/traces`` (the bound
    ``OPENCLAW_TRAJECTORY_DIR``); a profile-managed fallback path is searched too.
    Returns the most recently modified ``*.jsonl`` found, or ``None``.
    """
    candidates: list[str] = []
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith(".jsonl"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _read_entries(path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def convert(
    output: str,
    *,
    mcp_servers: tuple[str, ...] | list[str] = (),
    metadata: dict[str, Any] | None = None,
) -> TrajectoryArtifact:
    """Convert an OpenClaw episode's transcript into a TrajectoryArtifact.

    *output* is either the episode output directory (the JSONL is located within)
    or a path to the session JSONL itself. A missing / empty / unreadable trace
    degrades gracefully to an empty artifact (never raises) -- the run still
    produces a (vacuous) result rather than crashing the controller.
    """
    path = output if os.path.isfile(output) else find_session_log(output)
    if not path or not os.path.exists(path):
        return TrajectoryArtifact(trajectory_json=_TrajBuilder(metadata or {}).data)
    entries = _read_entries(path)
    if not entries:
        return TrajectoryArtifact(trajectory_json=_TrajBuilder(metadata or {}).data)
    return parse_entries(entries, mcp_servers=mcp_servers, metadata=metadata)
