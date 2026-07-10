"""Offline tests for the OpenClaw session-JSONL -> TrajectoryArtifact converter.

No Node / OpenClaw / Docker: parses a canned ``openclaw-trajectory`` JSONL fixture
and verifies the env/native split, the cross-snapshot dedup, multi-turn responses,
and the DT-Arena ``trajectory_json`` schema.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dtap_openclaw_target import trajectory as traj

FIXTURE = Path(__file__).parent / "fixtures" / "openclaw_session.jsonl"
MCP_SERVERS = ("travel-suite",)


@pytest.fixture
def artifact():
    return traj.convert(
        str(FIXTURE), mcp_servers=MCP_SERVERS, metadata={"task_id": "1", "domain": "travel"}
    )


# --- tool-name classification ---------------------------------------------


def test_resolve_server_mcp_prefix() -> None:
    assert traj.resolve_server("travel-suite_search_flights", MCP_SERVERS) == "travel-suite"


def test_resolve_server_workspace_plugin_prefix() -> None:
    # Upstream static-plugin naming: workspace_<Server>_<tool>.
    assert traj.resolve_server("workspace_travel-suite_search", MCP_SERVERS) == "travel-suite"


def test_resolve_server_mcp_underscore_prefix() -> None:
    assert traj.resolve_server("mcp_travel-suite_search", MCP_SERVERS) == "travel-suite"


def test_resolve_server_native_returns_none() -> None:
    assert traj.resolve_server("exec", MCP_SERVERS) is None
    assert traj.resolve_server("fs", MCP_SERVERS) is None


def test_resolve_server_unknown_server_not_matched() -> None:
    # A prefix for a server that is NOT configured does not classify as env.
    assert traj.resolve_server("other-suite_do", MCP_SERVERS) is None


def test_is_env_tool() -> None:
    assert traj.is_env_tool("travel-suite_search_flights", MCP_SERVERS) is True
    assert traj.is_env_tool("exec", MCP_SERVERS) is False


# --- the env/native split --------------------------------------------------


def test_native_tool_calls_exclude_mcp(artifact) -> None:
    # The MCP env tool went through the proxy (already emitted); only native exec kept.
    tools = [c["tool"] for c in artifact.native_tool_calls]
    assert tools == ["exec"]
    assert all(not traj.is_env_tool(c["tool"], MCP_SERVERS) for c in artifact.native_tool_calls)


def test_native_tool_call_carries_args_id_result_server(artifact) -> None:
    call = artifact.native_tool_calls[0]
    assert call["arguments"] == {"command": "ls /tmp"}
    assert call["id"] == "call_exec_1"
    assert call["server"] == "openclaw"  # native fallback server label
    assert call["result"] == "file1.txt\nfile2.txt"  # non-JSON result kept as string


# --- messages / responses --------------------------------------------------


def test_messages_are_assistant_texts_in_order(artifact) -> None:
    assert [m["text"] for m in artifact.messages] == [
        "Let me search and check the filesystem.",
        "Flight AA123 is booked. Done.",
    ]
    assert all(m["role"] == "assistant" for m in artifact.messages)


def test_final_response(artifact) -> None:
    assert artifact.final_response == "Flight AA123 is booked. Done."


def test_agent_responses_are_per_turn(artifact) -> None:
    # Two prompt.submitted turns -> two per-turn final texts.
    assert artifact.agent_responses == (
        "Let me search and check the filesystem.",
        "Flight AA123 is booked. Done.",
    )


def test_dedup_across_cumulative_snapshots(artifact) -> None:
    # Turn 2's cumulative snapshot re-sends turn 1's messages; nothing double-counts.
    assert len(artifact.native_tool_calls) == 1  # exec once, not twice
    assert len(artifact.messages) == 2  # not 3+
    tj = artifact.trajectory_json
    user_steps = [s for s in tj["trajectory"] if s["role"] == "user"]
    assert len(user_steps) == 2  # two distinct prompts


# --- trajectory_json (DT-Arena schema) -------------------------------------


def test_trajectory_json_schema_and_counts(artifact) -> None:
    tj = artifact.trajectory_json
    assert set(tj) == {"task_info", "traj_info", "trajectory"}
    assert tj["task_info"]["task_id"] == "1"
    assert tj["task_info"]["domain"] == "travel"
    info = tj["traj_info"]
    # 2 user + 2 agent actions (exec, mcp) + 2 tool returns + 2 agent messages.
    assert info["step_count"] == 8
    assert info["actions_count"] == 4  # 2 tool actions + 2 messages
    assert info["tool_count"] == 2
    assert info["user_turn"] == 2
    assert info["agent_final_response"] == "Flight AA123 is booked. Done."


def test_trajectory_json_includes_both_env_and_native_actions(artifact) -> None:
    # The full trace (for the judge) keeps BOTH the native and the MCP tool calls,
    # even though native_tool_calls excludes the MCP one.
    actions = [
        s["metadata"].get("tool_name")
        for s in artifact.trajectory_json["trajectory"]
        if s["role"] == "agent" and "tool_name" in s["metadata"]
    ]
    assert "exec" in actions
    assert "travel-suite_search_flights" in actions


def test_trajectory_json_metadata_captures_status_and_system_prompt(artifact) -> None:
    meta = artifact.trajectory_json["traj_info"]["metadata"]
    assert meta["final_status"] == "completed"
    assert meta["system_prompt"] == "You are a travel agent."
    assert meta["framework"] == "openclaw"


def test_mcp_tool_result_json_decoded(artifact) -> None:
    # The MCP tool's JSON-string return is decoded into the trajectory_json.
    steps = artifact.trajectory_json["trajectory"]
    tool_states = [s["state"] for s in steps if s["role"] == "tool"]
    assert {"flights": ["AA123"]} in tool_states


# --- convert() entry point & robustness ------------------------------------


def test_convert_locates_jsonl_in_directory(tmp_path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    shutil.copy(FIXTURE, traces / "dtap-abc.jsonl")
    art = traj.convert(str(tmp_path), mcp_servers=MCP_SERVERS)
    assert art.final_response == "Flight AA123 is booked. Done."


def test_find_session_log_picks_newest(tmp_path) -> None:
    import os
    import time

    old = tmp_path / "old.jsonl"
    new = tmp_path / "new.jsonl"
    old.write_text('{"type":"session.ended","data":{}}\n')
    new.write_text('{"type":"session.ended","data":{}}\n')
    os.utime(old, (1, 1))
    os.utime(new, (time.time(), time.time()))
    assert traj.find_session_log(str(tmp_path)) == str(new)


def test_find_session_log_prefers_traces_over_newer_profile_session(tmp_path) -> None:
    # HOME=/state=episode dir, so the tree holds BOTH the runtime trace (traces/) and
    # OpenClaw's own profile session store (a DIFFERENT schema). Even when the profile
    # session file is NEWER, the runtime trace under traces/ must win (else parsing
    # the wrong-schema file yields an empty artifact -> false-negative grading).
    import os

    traces = tmp_path / "traces"
    traces.mkdir()
    runtime = traces / "dtap-sess.jsonl"
    runtime.write_text('{"type":"session.ended","data":{"status":"completed"}}\n')

    sessions = tmp_path / ".openclaw-p1" / "agents" / "main" / "sessions"
    sessions.mkdir(parents=True)
    profile_session = sessions / "dtap-sess.jsonl"
    profile_session.write_text('{"schema":"session","messages":[]}\n')

    os.utime(runtime, (1, 1))  # older
    os.utime(profile_session, (2_000_000_000, 2_000_000_000))  # newer

    assert traj.find_session_log(str(tmp_path)) == str(runtime)


def test_convert_missing_file_is_graceful() -> None:
    art = traj.convert("/nonexistent/dir/that/does/not/exist", mcp_servers=MCP_SERVERS)
    assert art.final_response == ""
    assert art.native_tool_calls == ()
    assert art.messages == ()
    assert "task_info" in art.trajectory_json  # still a well-formed (empty) trajectory


def test_convert_empty_file_is_graceful(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    art = traj.convert(str(empty), mcp_servers=MCP_SERVERS)
    assert art.final_response == ""
    assert art.agent_responses == ()


def test_convert_skips_malformed_lines(tmp_path) -> None:
    f = tmp_path / "mixed.jsonl"
    f.write_text(
        "\n"
        "not json at all\n"
        + json.dumps({"type": "prompt.submitted", "data": {"prompt": "hi"}})
        + "\n"
        + json.dumps(
            {
                "type": "model.completed",
                "data": {
                    "messagesSnapshot": [
                        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
                    ]
                },
            }
        )
        + "\n"
    )
    art = traj.convert(str(f), mcp_servers=MCP_SERVERS)
    assert art.final_response == "answer"
    assert art.agent_responses == ("answer",)


def test_convert_tolerates_invalid_utf8(tmp_path) -> None:
    """A trace truncated mid-multibyte-char must not raise UnicodeDecodeError.

    The whole-episode timeout backstop returns the partial trace on disk BY DESIGN so
    a state-mutating attack is still judged; a final byte that truncates a UTF-8
    sequence must degrade gracefully (errors='replace'), not crash the task.
    """
    f = tmp_path / "truncated.jsonl"
    good = json.dumps({"type": "prompt.submitted", "data": {"prompt": "hi"}}).encode() + b"\n"
    # last bytes truncate a multibyte char (first byte 0xc3 of 'e-acute')
    f.write_bytes(good + b'{"type":"model.completed","data":{"t":"\xc3')
    art = traj.convert(str(f), mcp_servers=MCP_SERVERS)  # must not raise
    assert isinstance(art.final_response, str)


# All TRUTHY non-dicts: each would have crashed the pre-fix `data = entry.get("data") or {}`
# (which coerced only FALSY values), so every param exercises the regression.
@pytest.mark.parametrize("bad_data", ["a-string", 5, True, [1, 2]])
def test_parse_entries_tolerates_non_dict_data(bad_data) -> None:
    """A handled event whose ``data`` is not a dict must not crash (coerced to {}):
    the event yields nothing usable rather than raising AttributeError out of run()."""
    entries = [{"type": "prompt.submitted", "data": bad_data}]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)  # must not raise
    assert art.final_response == ""


@pytest.mark.parametrize(
    "entries",
    [
        # assistant content that is a truthy non-list (int) -> was TypeError (not iterable)
        [
            {
                "type": "model.completed",
                "data": {"messagesSnapshot": [{"role": "assistant", "content": 5}]},
            }
        ],
        # toolCall arguments that are a non-dict -> _format_action did arguments.items()
        [
            {
                "type": "model.completed",
                "data": {
                    "messagesSnapshot": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "c1",
                                    "name": "exec",
                                    "arguments": "notadict",
                                }
                            ],
                        }
                    ]
                },
            }
        ],
        # a non-str prompt -> `prompt not in seen_users` hashed an unhashable list
        [{"type": "prompt.submitted", "data": {"prompt": ["a", "b"]}}],
    ],
)
def test_parse_entries_tolerates_garbled_leaf_fields(entries) -> None:
    """Leaf fields (assistant content, toolCall arguments, prompt) of the wrong type must
    not raise out of parse_entries -- the guards skip them fine-grained."""
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)  # must not raise
    assert isinstance(art.final_response, str)


def test_convert_tolerates_deeply_nested_line(tmp_path) -> None:
    """A deeply-nested JSON line makes json.loads raise RecursionError inside _read_entries
    (which runs BEFORE convert()'s parse guard); it must be skipped, not raise out of convert()."""
    f = tmp_path / "nested.jsonl"
    f.write_text(
        "[" * 100000 + "\n" + json.dumps({"type": "prompt.submitted", "data": {"prompt": "hi"}})
    )
    art = traj.convert(str(f), mcp_servers=MCP_SERVERS)  # must not raise
    assert isinstance(art.final_response, str)


def test_convert_tolerates_oversized_int_line(tmp_path) -> None:
    """A line with a >4300-digit integer makes json.loads raise a bare ValueError (NOT a
    JSONDecodeError) inside _read_entries, before convert()'s parse guard; it must be
    skipped, not raise out of convert()."""
    f = tmp_path / "bigint.jsonl"
    f.write_text('{"type":"session.ended","data":{"n":' + "1" * 4301 + "}}")
    art = traj.convert(str(f), mcp_servers=MCP_SERVERS)  # must not raise
    assert isinstance(art.final_response, str)


def test_convert_tolerates_dangling_symlink_and_keeps_real_trace(tmp_path) -> None:
    """A dangling *.jsonl symlink in the trace tree must not crash convert() (getmtime
    FileNotFoundError, previously OUTSIDE the backstop); a real trace beside it is still
    picked -- the broken link is filtered, not merely swallowed to an empty artifact."""
    import os

    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "real.jsonl").write_text(
        json.dumps({"type": "prompt.submitted", "data": {"prompt": "hi"}})
        + "\n"
        + json.dumps(
            {
                "type": "model.completed",
                "data": {
                    "messagesSnapshot": [
                        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
                    ]
                },
            }
        )
    )
    os.symlink(str(tmp_path / "nonexistent"), str(traces / "broken.jsonl"))
    art = traj.convert(str(tmp_path), mcp_servers=MCP_SERVERS)  # must not raise
    assert art.final_response == "answer"  # the real trace was picked, not lost to empty


def test_parse_tool_output_keeps_oversized_int_payload() -> None:
    """A tool-output payload that is a >4300-digit int (a garbled env-tool return) makes
    json.loads raise a bare ValueError; it is kept as the raw string, not raised, so one
    bad output does not sink the whole trajectory via convert()'s backstop."""
    big = "1" * 4301
    assert traj._parse_tool_output(big) == big


def test_parse_entries_tolerates_garbled_snapshot() -> None:
    """A non-dict messagesSnapshot element and a null assistant ``content`` must not
    crash ``_process_snapshot`` (previously AttributeError / TypeError out of run());
    the well-formed message in the same snapshot still parses."""
    entries = [
        {"type": "prompt.submitted", "data": {"prompt": "q"}},
        {
            "type": "model.completed",
            "data": {
                "messagesSnapshot": [
                    "not-a-dict-message",
                    {"role": "assistant", "content": None},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                ]
            },
        },
    ]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)  # must not raise
    assert art.final_response == "ok"


def test_parse_entries_assistant_texts_fallback() -> None:
    # When a model.completed has no messagesSnapshot, fall back to assistantTexts.
    entries = [
        {"type": "prompt.submitted", "data": {"prompt": "q"}},
        {"type": "model.completed", "data": {"assistantTexts": ["part one", "part two"]}},
    ]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)
    assert art.final_response == "part one\n\npart two"


def test_parse_entries_no_turns_uses_final_fallback() -> None:
    # No prompt.submitted at all: agent_responses falls back to [final_response].
    entries = [
        {
            "type": "model.completed",
            "data": {
                "messagesSnapshot": [
                    {"role": "assistant", "content": [{"type": "text", "text": "only"}]}
                ]
            },
        },
    ]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)
    assert art.agent_responses == ("only",)
    assert art.final_response == "only"


# --- pure helpers ----------------------------------------------------------


def test_extract_texts_handles_string_blocks_and_non_list() -> None:
    assert traj._extract_texts(["bare string", {"type": "text", "text": "block"}]) == [
        "bare string",
        "block",
    ]
    assert traj._extract_texts("not a list") == []
    assert traj._extract_texts(["   ", {"type": "text", "text": "  "}]) == []  # blanks dropped


def test_format_action_non_string_args() -> None:
    # Non-string arg values are rendered key=value (not quoted); strings are truncated.
    assert traj._format_action("page", {"limit": 5, "deep": True}) == "page(limit=5, deep=True)"
    long = "x" * 130
    assert traj._format_action("q", {"s": long}) == f'q(s="{"x" * 100}...")'


def test_parse_tool_output_decodes_json_string_else_passthrough() -> None:
    assert traj._parse_tool_output('{"a": 1}') == {"a": 1}
    assert traj._parse_tool_output("plain text") == "plain text"
    # non-string payload passes through unchanged
    assert traj._parse_tool_output({"already": "obj"}) == {"already": "obj"}


def test_join_texts_empty_or_non_list_is_none() -> None:
    assert traj._join_texts([]) is None
    assert traj._join_texts(None) is None
    assert traj._join_texts("nope") is None


def test_native_tool_call_without_result(tmp_path) -> None:
    # A native tool call with no matching toolResult records result=None (no crash).
    entries = [
        {"type": "prompt.submitted", "data": {"prompt": "go"}},
        {
            "type": "model.completed",
            "data": {
                "messagesSnapshot": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "c1",
                                "name": "exec",
                                "arguments": {"command": "ls"},
                            }
                        ],
                    }
                ]
            },
        },
    ]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)
    assert art.native_tool_calls[0]["result"] is None
    # the action step exists but no tool-return step was appended
    tool_returns = [s for s in art.trajectory_json["trajectory"] if s["role"] == "tool"]
    assert tool_returns == []


def test_non_dict_content_block_skipped() -> None:
    # A stray non-dict block in assistant content is ignored.
    entries = [
        {"type": "prompt.submitted", "data": {"prompt": "go"}},
        {
            "type": "model.completed",
            "data": {
                "messagesSnapshot": [
                    {"role": "assistant", "content": ["junk", {"type": "text", "text": "real"}]}
                ]
            },
        },
    ]
    art = traj.parse_entries(entries, mcp_servers=MCP_SERVERS)
    assert [m["text"] for m in art.messages] == ["real"]
