"""In-container Claude Code driver. Runs INSIDE the agent Docker image; never
imported by the host.

It reads the mounted ``task.json``, drives ``claude_agent_sdk``'s
``ClaudeSDKClient`` turn-by-turn against the host MCP proxy (a single HTTP MCP
server named ``dtap_proxy``), and writes ``transcript.jsonl`` (the upstream
``ClaudeSDKTraceProcessor`` schema) plus ``result.json`` to the mounted instance
dir. The host then parses the transcript with ``trajectory.convert``.

The ``claude_agent_sdk`` import is deferred into the run path so the host package
imports cleanly without the SDK (it lives only in the image). The message
serializers below use duck typing by class name, so they are import-safe and
unit-testable on the host without the SDK installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

# Host<->container contract. trajectory.py defines matching constants; a host
# test asserts they agree to prevent drift.
TRANSCRIPT_FILENAME = "transcript.jsonl"
RESULT_FILENAME = "result.json"
PROXY_SERVER_NAME = "dtap_proxy"
DEFAULT_TASK_FILE = "/dtap/task.json"

# Claude Code's native (in-container) tool menu. The agent gets all of these
# EXCEPT the per-task deny list (applied via ``disallowed_tools``), plus the
# proxy's env tools via the ``mcp__dtap_proxy__*`` glob. This list mirrors the
# Claude Code built-in toolset; it is the single calibration point if the CLI's
# native tool names change (see ASSUMPTIONS.md C).
DEFAULT_NATIVE_TOOLS = (
    "Task",
    "Bash",
    "BashOutput",
    "KillShell",
    "Glob",
    "Grep",
    "Read",
    "Edit",
    "MultiEdit",
    "Write",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "SlashCommand",
    "AskUserQuestion",
    "ExitPlanMode",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _serialize_blocks(content: Any) -> list[dict[str, Any]]:
    """Serialize a message's content blocks to the transcript schema (duck-typed)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    out: list[dict[str, Any]] = []
    for block in content or []:
        name = type(block).__name__
        if name == "TextBlock":
            out.append({"type": "text", "text": getattr(block, "text", "")})
        elif name == "ToolUseBlock":
            out.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )
        elif name == "ToolResultBlock":
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": getattr(block, "tool_use_id", None),
                    "content": getattr(block, "content", None),
                    "is_error": bool(getattr(block, "is_error", False)),
                }
            )
        elif name == "ThinkingBlock":
            out.append({"type": "thinking", "thinking": getattr(block, "thinking", None)})
        else:
            out.append({"type": "unknown", "raw": str(block)})
    return out


def _serialize_message(message: Any) -> dict[str, Any]:
    """Serialize a Claude SDK message to the transcript schema (duck-typed)."""
    name = type(message).__name__
    if name == "AssistantMessage":
        return {"type": "assistant", "content": _serialize_blocks(getattr(message, "content", []))}
    if name == "UserMessage":
        return {"type": "user", "content": _serialize_blocks(getattr(message, "content", []))}
    if name == "SystemMessage":
        return {
            "type": "system",
            "subtype": getattr(message, "subtype", None),
            "data": getattr(message, "data", {}),
        }
    if name == "ResultMessage":
        return {
            "type": "result",
            "subtype": getattr(message, "subtype", None),
            "result": getattr(message, "result", None),
            "is_error": bool(getattr(message, "is_error", False)),
            "total_cost_usd": getattr(message, "total_cost_usd", None),
        }
    return {"type": "unknown", "raw": str(message)}


def _build_options(task: dict[str, Any], options_cls: Any) -> Any:
    """Build a ``ClaudeAgentOptions`` from the task spec.

    Native tools are enabled (the full ``DEFAULT_NATIVE_TOOLS`` menu) plus the
    proxy's env tools via the ``mcp__dtap_proxy__*`` glob, with the per-task deny
    list subtracted through ``disallowed_tools``. ``options_cls`` is injected so
    this is unit-testable without the SDK.
    """
    deny = list(task.get("native_tool_deny") or [])
    allowed = list(DEFAULT_NATIVE_TOOLS) + [f"mcp__{PROXY_SERVER_NAME}__*"]
    kwargs: dict[str, Any] = {
        "model": task.get("model"),
        "permission_mode": "bypassPermissions",
        "mcp_servers": {PROXY_SERVER_NAME: {"type": "http", "url": task.get("proxy_url")}},
        "allowed_tools": allowed,
        "disallowed_tools": deny,
        "include_partial_messages": True,
        "max_turns": int(task.get("max_turns") or 200),
    }
    system_prompt = task.get("system_prompt")
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    workspace = task.get("workspace_dir")
    if workspace:
        kwargs["cwd"] = workspace
    return options_cls(**kwargs)


def _final_text_from_message(message: Any) -> str | None:
    """Return this message's contribution to the running final-output, if any."""
    name = type(message).__name__
    if name == "AssistantMessage":
        text = ""
        for block in _serialize_blocks(getattr(message, "content", [])):
            if block["type"] == "text" and block["text"]:
                text = block["text"]
        return text or None
    if name == "ResultMessage":
        result = getattr(message, "result", None)
        return result if isinstance(result, str) and result else None
    return None


def materialize_skills(skills: Any, workspace_dir: str) -> list[str]:
    """Write the DTAP skill-vector into ``<workspace>/.claude/skills/<name>/SKILL.md``.

    Claude Code discovers skills from ``.claude/skills`` under its ``cwd`` (the
    workspace), so materializing here is how the optimizer's ``skill`` controllable
    actually reaches the agent. ``mode`` mirrors the upstream skill injection:
    ``create`` (default, overwrite), ``append`` (extend an existing SKILL.md), or
    ``insert`` (insert ``content`` before 1-indexed ``row``). Pure file I/O, so it
    is unit-testable on the host without the SDK. Returns the written paths.
    """
    written: list[str] = []
    base = os.path.join(workspace_dir, ".claude", "skills")
    for skill in skills or []:
        name = skill.get("name") or skill.get("skill_name")
        if not name:
            continue
        content = skill.get("content", "") or ""
        mode = skill.get("mode", "create")
        skill_dir = os.path.join(base, name)
        os.makedirs(skill_dir, exist_ok=True)
        path = os.path.join(skill_dir, "SKILL.md")
        if mode == "append" and os.path.exists(path):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n" + content)
        elif mode == "insert" and os.path.exists(path):
            row = int(skill.get("row", 1) or 1)
            lines = open(path, encoding="utf-8").read().splitlines()
            idx = max(0, min(len(lines), row - 1))
            lines.insert(idx, content)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        else:  # create / overwrite
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        written.append(path)
    return written


async def run_episode(task: dict[str, Any], output_dir: str) -> dict[str, Any]:  # pragma: no cover
    """Run all turns, write the transcript, and return ``{final_output, error, duration}``.

    Requires ``claude_agent_sdk`` + a working Claude Code CLI; exercised only by
    the docker+live e2e test.
    """
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # deferred SDK import

    transcript_path = os.path.join(output_dir, TRANSCRIPT_FILENAME)
    trace_id = str(uuid.uuid4())
    started = datetime.now(UTC)
    final_output = ""
    error: str | None = None

    with open(transcript_path, "w", encoding="utf-8") as fh:

        def write(obj: dict[str, Any]) -> None:
            fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
            fh.flush()

        write(
            {
                "record": "trace_start",
                "trace_id": trace_id,
                "metadata": task.get("metadata") or {},
                "ts": _now(),
            }
        )

        materialize_skills(task.get("skills"), task.get("workspace_dir") or output_dir)
        options = _build_options(task, ClaudeAgentOptions)
        client = ClaudeSDKClient(options=options)
        await client.connect()
        try:
            for turn in task.get("instructions") or [""]:
                write({"record": "user_input", "trace_id": trace_id, "content": turn, "ts": _now()})
                await client.query(turn)
                async for message in client.receive_response():
                    write(
                        {
                            "record": "message",
                            "trace_id": trace_id,
                            "message": _serialize_message(message),
                            "ts": _now(),
                        }
                    )
                    candidate = _final_text_from_message(message)
                    if candidate:
                        final_output = candidate
        except Exception as exc:  # noqa: BLE001 - surface any agent failure
            error = str(exc)
            write({"record": "error", "trace_id": trace_id, "error": error, "ts": _now()})
        finally:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001
                pass

        write({"record": "trace_end", "trace_id": trace_id, "ts": _now()})

    duration = (datetime.now(UTC) - started).total_seconds()
    return {"final_output": final_output, "error": error, "duration": round(duration, 3)}


def main() -> int:  # pragma: no cover - container entrypoint
    task_file = os.environ.get("DTAP_TASK_FILE", DEFAULT_TASK_FILE)
    with open(task_file, encoding="utf-8") as fh:
        task = json.load(fh)
    output_dir = task.get("output_dir") or os.path.dirname(task_file) or "."
    result = run_episode_sync(task, output_dir)
    with open(os.path.join(output_dir, RESULT_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 1 if result.get("error") else 0


def run_episode_sync(task: dict[str, Any], output_dir: str) -> dict[str, Any]:  # pragma: no cover
    return asyncio.run(run_episode(task, output_dir))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
