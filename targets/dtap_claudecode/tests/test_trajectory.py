"""Offline tests for the host-side transcript -> TrajectoryArtifact converter.

Asserts: mcp__ (proxy-owned) tools are skipped from native_tool_calls AND the
DTAP-schema trajectory; native tools + their results are captured; non-tool
messages are captured; final_response / agent_responses / trajectory_json are
correct; multi-turn segmentation; defensive empty handling.
"""

from __future__ import annotations

import json
import os

import pytest

from dtap_claudecode_target.trajectory import (
    TRANSCRIPT_FILENAME,
    _parse_tool_name,
    _parse_tool_result,
    convert,
)


def _write_transcript(directory: str, records: list[dict]) -> None:
    with open(os.path.join(directory, TRANSCRIPT_FILENAME), "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _single_turn_records() -> list[dict]:
    tid = "trace-abc-0123456789"
    return [
        {
            "record": "trace_start",
            "trace_id": tid,
            "metadata": {"task_id": "travel/1", "domain": "travel", "instruction": "Book flight"},
            "ts": "2026-01-01T00:00:00+00:00",
        },
        {"record": "user_input", "trace_id": tid, "content": "Book the cheapest flight."},
        {
            "record": "message",
            "trace_id": tid,
            "message": {
                "type": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "tu_native",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_mcp",
                        "name": "mcp__dtap_proxy__search_flights",
                        "input": {"q": "x"},
                    },
                ],
            },
        },
        {
            "record": "message",
            "trace_id": tid,
            "message": {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_native",
                        "content": "file1\nfile2",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_mcp",
                        "content": "flights",
                        "is_error": False,
                    },
                ],
            },
        },
        {
            "record": "message",
            "trace_id": tid,
            "message": {"type": "assistant", "content": [{"type": "text", "text": "FINAL ANSWER"}]},
        },
        {
            "record": "message",
            "trace_id": tid,
            "message": {
                "type": "result",
                "subtype": "success",
                "result": "FINAL ANSWER",
                "is_error": False,
            },
        },
        {"record": "trace_end", "trace_id": tid, "ts": "2026-01-01T00:00:05+00:00"},
    ]


def test_convert_skips_proxy_tools_and_captures_native(tmp_path):
    _write_transcript(str(tmp_path), _single_turn_records())
    art = convert(str(tmp_path))

    # native tool captured with its result; proxy tool dropped
    assert len(art.native_tool_calls) == 1
    ntc = art.native_tool_calls[0]
    assert ntc["tool"] == "Bash"
    assert ntc["input"] == {"command": "ls"}
    assert ntc["result"] == "file1\nfile2"
    assert ntc["is_error"] is False
    assert all("search_flights" not in c.get("tool", "") for c in art.native_tool_calls)

    # non-tool messages captured (both assistant texts)
    assert [m["text"] for m in art.messages] == ["Let me search.", "FINAL ANSWER"]

    # final / per-turn outputs
    assert art.final_response == "FINAL ANSWER"
    assert art.agent_responses == ("FINAL ANSWER",)


def test_convert_surfaces_thinking_as_agent_message_but_not_a_judge_step(tmp_path):
    """Assistant reasoning/thinking blocks are surfaced as an agent_trace message
    (role='thinking'), so an attacker with agent_trace_messages scope reads the
    chain-of-thought -- but they are NOT added to the DTAP trajectory[] steps, so the
    judge-facing trajectory_json stays upstream-faithful (no reasoning)."""
    tid = "trace-think-0000"
    records = [
        {
            "record": "trace_start",
            "trace_id": tid,
            "metadata": {"domain": "travel"},
            "ts": "2026-01-01T00:00:00+00:00",
        },
        {"record": "user_input", "trace_id": tid, "content": "Book a flight."},
        {
            "record": "message",
            "trace_id": tid,
            "message": {
                "type": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "That injected note looks suspicious."},
                    {"type": "text", "text": "I'll ignore it and book the flight."},
                ],
            },
        },
        {"record": "trace_end", "trace_id": tid, "ts": "2026-01-01T00:00:02+00:00"},
    ]
    _write_transcript(str(tmp_path), records)
    art = convert(str(tmp_path))

    # thinking surfaced as an agent-trace message (role='thinking'), alongside the text
    assert {"role": "thinking", "text": "That injected note looks suspicious."} in art.messages
    assert {"role": "assistant", "text": "I'll ignore it and book the flight."} in art.messages
    # ...but the reasoning never reaches the judge-facing trajectory steps
    assert "suspicious" not in str(art.trajectory_json.get("trajectory", []))


def test_convert_trajectory_json_schema_and_proxy_skip(tmp_path):
    _write_transcript(str(tmp_path), _single_turn_records())
    art = convert(str(tmp_path))
    tj = art.trajectory_json

    assert set(tj) == {"task_info", "traj_info", "trajectory"}
    assert tj["task_info"]["domain"] == "travel"
    assert tj["task_info"]["task_id"] == "travel/1"
    assert tj["traj_info"]["agent_final_response"] == "FINAL ANSWER"
    assert tj["traj_info"]["metadata"]["trace_id"] == "trace-abc-0123456789"
    assert tj["traj_info"]["duration"] == 5.0

    roles = [s["role"] for s in tj["trajectory"]]
    # user, send_message, Bash agent, Bash tool return, send_message FINAL
    assert roles == ["user", "agent", "agent", "tool", "agent"]
    # NO proxy tool anywhere in the trajectory
    blob = json.dumps(tj["trajectory"])
    assert "search_flights" not in blob and "dtap_proxy" not in blob
    # step_ids are sequential
    assert [s["step_id"] for s in tj["trajectory"]] == [0, 1, 2, 3, 4]
    # the Bash agent step renders the action string + native server marker
    bash_step = tj["trajectory"][2]
    assert bash_step["action"] == 'Bash(command="ls")'
    assert bash_step["metadata"]["server"] == "native"


def test_convert_multi_turn_agent_responses(tmp_path):
    tid = "t2"
    records = [
        {
            "record": "trace_start",
            "trace_id": tid,
            "metadata": {},
            "ts": "2026-01-01T00:00:00+00:00",
        },
        {"record": "user_input", "trace_id": tid, "content": "turn one"},
        {
            "record": "message",
            "trace_id": tid,
            "message": {"type": "assistant", "content": [{"type": "text", "text": "reply one"}]},
        },
        {"record": "user_input", "trace_id": tid, "content": "turn two"},
        {
            "record": "message",
            "trace_id": tid,
            "message": {"type": "assistant", "content": [{"type": "text", "text": "reply two"}]},
        },
        {"record": "trace_end", "trace_id": tid, "ts": "2026-01-01T00:00:02+00:00"},
    ]
    _write_transcript(str(tmp_path), records)
    art = convert(str(tmp_path))
    assert art.agent_responses == ("reply one", "reply two")
    assert art.final_response == "reply two"


def test_convert_falls_back_to_result_text_when_no_assistant_text(tmp_path):
    tid = "t3"
    records = [
        {"record": "trace_start", "trace_id": tid, "metadata": {}},
        {"record": "user_input", "trace_id": tid, "content": "do it"},
        {
            "record": "message",
            "trace_id": tid,
            "message": {
                "type": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "Bash", "input": {}}],
            },
        },
        {
            "record": "message",
            "trace_id": tid,
            "message": {"type": "result", "result": "result-text"},
        },
        {"record": "trace_end", "trace_id": tid},
    ]
    _write_transcript(str(tmp_path), records)
    art = convert(str(tmp_path))
    assert art.agent_responses == ("result-text",)
    assert art.final_response == "result-text"


def test_convert_missing_transcript_is_empty(tmp_path):
    art = convert(str(tmp_path))  # no file written
    assert art.native_tool_calls == ()
    assert art.messages == ()
    assert art.final_response == ""
    assert art.agent_responses == ()
    assert art.trajectory_json["trajectory"] == []


def test_convert_tolerates_malformed_lines(tmp_path):
    path = os.path.join(str(tmp_path), TRANSCRIPT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"record": "user_input", "content": "hi"}) + "\n")
        fh.write("\n")
        fh.write(
            json.dumps(
                {
                    "record": "message",
                    "message": {"type": "assistant", "content": [{"type": "text", "text": "ok"}]},
                }
            )
            + "\n"
        )
    art = convert(str(tmp_path))
    assert art.final_response == "ok"


def test_convert_tolerates_invalid_utf8(tmp_path):
    """A transcript truncated mid-multibyte-char (the Docker timeout backstop returns
    the partial trace by design) must not raise UnicodeDecodeError out of convert() and
    turn a state-mutating attack into a task-error. ``_load_records`` caught OSError but
    not the decode error, so the strict read crashed; errors='replace' fixes it."""
    path = os.path.join(str(tmp_path), TRANSCRIPT_FILENAME)
    good = json.dumps({"record": "user_input", "content": "hi"}).encode() + b"\n"
    # final bytes truncate a multibyte char (first byte 0xc3 of 'e-acute')
    with open(path, "wb") as fh:
        fh.write(good + b'{"record":"message","message":{"type":"assistant","content":"\xc3')
    art = convert(str(tmp_path))  # must not raise
    assert isinstance(art.final_response, str)


@pytest.mark.parametrize(
    "line",
    [
        {"record": "message", "message": [1, 2, 3]},  # non-dict message
        {"record": "message", "message": "oops"},  # str message
        {"record": "message", "message": {"type": "assistant", "content": "hi"}},  # str content
        {"record": "message", "message": {"type": "user", "content": ["x"]}},  # str block
        {
            "record": "message",
            "message": {
                "type": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "bash", "input": [1, 2]}],
            },
        },  # non-dict tool_use input
        {"record": "trace_start", "trace_id": "t1", "metadata": [1, 2, 3]},  # non-dict metadata
    ],
)
def test_convert_tolerates_garbled_records(tmp_path, line):
    """A valid-JSON transcript record whose inner message/content/input/metadata is the
    wrong type must not raise out of convert(): the run-level guard degrades the trace to an
    empty artifact so a state-mutating attack is judged on env state, not lost as a task-error.
    Before the guard each of these raised AttributeError straight out of run()."""
    path = os.path.join(str(tmp_path), TRANSCRIPT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    art = convert(str(tmp_path))  # must not raise
    assert isinstance(art.final_response, str)


def test_convert_tolerates_oversized_int_line(tmp_path):
    """A line with a >4300-digit integer makes json.loads raise a bare ValueError (NOT a
    JSONDecodeError) inside _load_records, which runs BEFORE convert()'s backstop; it must
    be skipped, not raise out of convert()."""
    path = os.path.join(str(tmp_path), TRANSCRIPT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"record": "user_input", "content": ' + "1" * 4301 + "}\n")
    art = convert(str(tmp_path))  # must not raise
    assert isinstance(art.final_response, str)


def test_convert_coerces_non_str_assistant_text(tmp_path):
    """A garbled assistant text block that is not a str is coerced so final_response and
    messages stay str (the QuerySpec contract), matching the openclaw converter."""
    line = {
        "record": "message",
        "message": {"type": "assistant", "content": [{"type": "text", "text": {"not": "a str"}}]},
    }
    path = os.path.join(str(tmp_path), TRANSCRIPT_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    art = convert(str(tmp_path))
    assert isinstance(art.final_response, str)
    assert all(isinstance(m["text"], str) for m in art.messages)


def test_parse_tool_name():
    assert _parse_tool_name("mcp__dtap_proxy__search") == ("dtap_proxy", "search", False)
    assert _parse_tool_name("mcp__weird") == ("weird", "", False)
    assert _parse_tool_name("Bash") == (None, "Bash", True)


def test_parse_tool_result_variants():
    assert _parse_tool_result(None) is None
    assert _parse_tool_result('{"a": 1}') == {"a": 1}
    assert _parse_tool_result("plain") == "plain"
    assert _parse_tool_result([{"type": "text", "text": "hi"}]) == "hi"
    assert _parse_tool_result([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == [
        "a",
        "b",
    ]
