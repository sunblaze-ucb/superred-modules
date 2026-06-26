"""Convert a Claude Code in-container transcript into a normalized TrajectoryArtifact.

This runs on the HOST. The agent driver (``driver.py``, run inside the Docker
image) writes a JSONL transcript in the upstream ``ClaudeSDKTraceProcessor``
schema (records: ``trace_start`` / ``user_input`` / ``message`` / ``error`` /
``trace_end``) plus a ``result.json``. :func:`convert` parses the transcript and
produces a :class:`~dtap_scaffold.types.TrajectoryArtifact`, mirroring the
upstream ``ClaudeSDKTrajectoryConverter`` with one DTAP-specific rule:

**Proxied env tools are SKIPPED.** In the superred DTAP design every environment
MCP tool is exposed through a single host proxy server named ``dtap_proxy``, so
those calls appear in the transcript as ``mcp__dtap_proxy__<tool>``. The host MCP
proxy already observed each one (it emitted the ObservableEvent and fired the
per-server env-tool PostCall controllable), so the converter drops every
``mcp__``-prefixed tool from ``native_tool_calls`` AND from the DTAP-schema
``trajectory`` steps -- only the agent's NATIVE (container) tools and its non-tool
messages survive here, which is what the base emits once.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from dtap_scaffold.types import TrajectoryArtifact

# Filenames the in-container driver writes under the mounted instance dir.
# These are the host<->container contract; driver.py defines matching constants.
TRANSCRIPT_FILENAME = "transcript.jsonl"
RESULT_FILENAME = "result.json"

# The single proxy MCP server name the agent connects to (see driver.py). Tools
# named ``mcp__<PROXY_SERVER_NAME>__*`` are proxy-owned and skipped here.
PROXY_SERVER_NAME = "dtap_proxy"

__all__ = [
    "convert",
    "TRANSCRIPT_FILENAME",
    "RESULT_FILENAME",
    "PROXY_SERVER_NAME",
]


def convert(output_dir: str) -> TrajectoryArtifact:
    """Parse ``<output_dir>/transcript.jsonl`` into a TrajectoryArtifact.

    Missing/empty transcript -> an empty artifact (defensive: a crashed episode
    still yields a well-formed, queryable trajectory).
    """
    records = _load_records(output_dir)

    native_tool_calls: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    tool_index: dict[Any, dict[str, Any]] = {}  # tool_use_id -> {server, tool, native}
    ntc_by_id: dict[Any, dict[str, Any]] = {}  # tool_use_id -> native_tool_calls entry

    task_info: dict[str, Any] = {
        "task_id": "unknown",
        "original_instruction": "",
        "malicious_instruction": "",
        "domain": None,
        "risk_category": None,
    }
    trace_id: str | None = None
    start_ts: str | None = None
    end_ts: str | None = None

    # Per-turn final outputs (one per user_input turn) -> agent_responses.
    turns: list[str] = []
    turn_open = False
    turn_final_text = ""
    turn_result_text = ""

    def close_turn() -> None:
        nonlocal turn_open, turn_final_text, turn_result_text
        if turn_open:
            turns.append(turn_final_text or turn_result_text or "")
        turn_open = False
        turn_final_text = ""
        turn_result_text = ""

    sid = 0

    def add_step(step: dict[str, Any]) -> None:
        nonlocal sid
        step["step_id"] = sid
        sid += 1
        steps.append(step)

    for rec in records:
        rtype = rec.get("record")

        if rtype == "trace_start":
            trace_id = rec.get("trace_id")
            start_ts = rec.get("ts")
            _seed_task_info(task_info, rec.get("metadata") or {}, trace_id)

        elif rtype == "trace_end":
            end_ts = rec.get("ts")

        elif rtype == "user_input":
            close_turn()
            turn_open = True
            content = rec.get("content", "")
            if not task_info["original_instruction"] and isinstance(content, str):
                task_info["original_instruction"] = content
            add_step({"role": "user", "state": content, "metadata": {}})

        elif rtype == "message":
            msg = rec.get("message") or {}
            mtype = msg.get("type")

            if mtype == "assistant":
                turn_open = True  # lazily open a turn if no user_input preceded
                for block in msg.get("content") or []:
                    btype = block.get("type")
                    if btype == "tool_use":
                        bid = block.get("id")
                        server, tool, native = _parse_tool_name(block.get("name", ""))
                        tool_index[bid] = {"server": server, "tool": tool, "native": native}
                        if not native:
                            continue  # proxy-owned env tool -> skip
                        tool_input = block.get("input") or {}
                        entry = {
                            "tool": tool,
                            "id": bid,
                            "input": tool_input,
                            "result": None,
                            "is_error": False,
                        }
                        native_tool_calls.append(entry)
                        if bid is not None:
                            ntc_by_id[bid] = entry
                        add_step(
                            {
                                "role": "agent",
                                "action": _action_str(tool, tool_input),
                                "metadata": {
                                    "tool_name": tool,
                                    "tool_params": tool_input,
                                    "server": server or "native",
                                },
                            }
                        )
                    elif btype == "text":
                        text = block.get("text", "")
                        if text:
                            messages.append({"role": "assistant", "text": text})
                            add_step(
                                {
                                    "role": "agent",
                                    "action": "send_message_to_user",
                                    "metadata": {"message": text},
                                }
                            )
                            turn_final_text = text
                    # thinking / unknown blocks are not surfaced (upstream parity)

            elif mtype == "user":
                for block in msg.get("content") or []:
                    if block.get("type") != "tool_result":
                        continue
                    tuid = block.get("tool_use_id")
                    info = tool_index.get(tuid)
                    if not info or not info.get("native"):
                        continue  # proxy-owned or orphan result -> skip
                    is_err = bool(block.get("is_error"))
                    result = _parse_tool_result(block.get("content"))
                    state = {"error": result} if is_err else result
                    paired = ntc_by_id.get(tuid)
                    if paired is not None:
                        paired["result"] = state
                        paired["is_error"] = is_err
                    add_step(
                        {
                            "role": "tool",
                            "state": state,
                            "metadata": {
                                "tool_name": info.get("tool"),
                                "server": info.get("server") or "native",
                            },
                        }
                    )

            elif mtype == "result":
                turn_open = True
                rtext = msg.get("result")
                if isinstance(rtext, str) and rtext:
                    turn_result_text = rtext

        # "error" records carry no trajectory step.

    close_turn()
    agent_responses = turns
    final_response = agent_responses[-1] if agent_responses else ""

    traj_info = {
        "step_count": len(steps),
        "actions_count": sum(1 for s in steps if s["role"] == "agent"),
        "tool_count": sum(1 for s in steps if s["role"] == "tool"),
        "user_turn": sum(1 for s in steps if s["role"] == "user"),
        "duration": _duration(start_ts, end_ts),
        "timestamp": end_ts or start_ts or "",
        "agent_final_response": final_response,
        "metadata": {"trace_id": trace_id},
    }
    trajectory_json: dict[str, Any] = {
        "task_info": task_info,
        "traj_info": traj_info,
        "trajectory": steps,
    }

    return TrajectoryArtifact(
        native_tool_calls=tuple(native_tool_calls),
        messages=tuple(messages),
        final_response=final_response,
        agent_responses=tuple(agent_responses),
        trajectory_json=trajectory_json,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_records(output_dir: str) -> list[dict[str, Any]]:
    """Read the transcript JSONL; tolerate a missing file and malformed lines."""
    path = os.path.join(output_dir, TRANSCRIPT_FILENAME)
    records: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except OSError:
        return []
    return records


def _parse_tool_name(full_name: str) -> tuple[str | None, str, bool]:
    """``mcp__server__tool`` -> (server, tool, native=False); else (None, name, native=True).

    Native (in-container) tools have no ``mcp__`` prefix.
    """
    if full_name.startswith("mcp__"):
        rest = full_name[len("mcp__") :]
        parts = rest.split("__", 1)
        if len(parts) == 2:
            return parts[0], parts[1], False
        return rest, "", False
    return None, full_name, True


def _action_str(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Render a call as ``tool(k1="v1", k2=2)`` (mirrors the upstream converter)."""
    params: list[str] = []
    for key, value in (tool_input or {}).items():
        if isinstance(value, str):
            params.append(f'{key}="{value}"')
        else:
            params.append(f"{key}={value}")
    return f"{tool_name}(" + ", ".join(params) + ")"


def _parse_tool_result(content: Any) -> Any:
    """Parse an MCP/SDK tool-result payload (mirrors the upstream converter)."""
    if content is None:
        return None
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    if isinstance(content, list):
        texts: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                try:
                    texts.append(json.loads(text))
                except (json.JSONDecodeError, ValueError, TypeError):
                    texts.append(text)
            else:
                texts.append(item)
        return texts[0] if len(texts) == 1 else texts
    return content


def _seed_task_info(task_info: dict[str, Any], meta: dict[str, Any], trace_id: str | None) -> None:
    task_info["task_id"] = meta.get("task_id") or (trace_id[:16] if trace_id else "unknown")
    instr = meta.get("instruction")
    if isinstance(instr, str):
        task_info["original_instruction"] = instr
    elif isinstance(instr, list) and instr:
        task_info["original_instruction"] = str(instr[0])
    task_info["malicious_instruction"] = meta.get("malicious_goal") or ""
    task_info["domain"] = meta.get("domain")
    task_info["risk_category"] = meta.get("category") or meta.get("risk_category")


def _duration(start_ts: str | None, end_ts: str | None) -> float:
    if not start_ts or not end_ts:
        return 0.0
    try:
        start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    return round((end - start).total_seconds(), 3)
