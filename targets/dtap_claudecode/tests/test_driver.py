"""Offline tests for the in-container driver's pure helpers.

The driver imports ``claude_agent_sdk`` only inside its run path, so it imports
on the host without the SDK. Its serializers use class-name duck typing, so we
exercise them with fakes whose class names match the SDK message/block types,
and we inject a fake ``ClaudeAgentOptions`` to test option construction.
"""

from __future__ import annotations

from dtap_claudecode_target import driver

# ----- fake SDK message/block types (class names matter, not the module) ---


class TextBlock:
    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    def __init__(self, id, name, input):  # noqa: A002 - mirror SDK field name
        self.id = id
        self.name = name
        self.input = input


class ToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class ThinkingBlock:
    def __init__(self, thinking):
        self.thinking = thinking


class AssistantMessage:
    def __init__(self, content):
        self.content = content


class UserMessage:
    def __init__(self, content):
        self.content = content


class ResultMessage:
    def __init__(self, result, subtype="success", is_error=False):
        self.result = result
        self.subtype = subtype
        self.is_error = is_error


def test_serialize_blocks_all_types():
    blocks = [
        TextBlock("hi"),
        ToolUseBlock("id1", "Bash", {"command": "ls"}),
        ToolResultBlock("id1", "out", is_error=False),
        ThinkingBlock("hmm"),
        object(),  # unknown
    ]
    out = driver._serialize_blocks(blocks)
    assert out[0] == {"type": "text", "text": "hi"}
    assert out[1] == {"type": "tool_use", "id": "id1", "name": "Bash", "input": {"command": "ls"}}
    assert out[2] == {
        "type": "tool_result",
        "tool_use_id": "id1",
        "content": "out",
        "is_error": False,
    }
    assert out[3] == {"type": "thinking", "thinking": "hmm"}
    assert out[4]["type"] == "unknown"


def test_serialize_blocks_string_content():
    assert driver._serialize_blocks("plain") == [{"type": "text", "text": "plain"}]


def test_serialize_message_types():
    assert driver._serialize_message(AssistantMessage([TextBlock("a")])) == {
        "type": "assistant",
        "content": [{"type": "text", "text": "a"}],
    }
    assert driver._serialize_message(UserMessage([ToolResultBlock("i", "r")]))["type"] == "user"
    rm = driver._serialize_message(ResultMessage("final"))
    assert rm["type"] == "result" and rm["result"] == "final"
    assert driver._serialize_message(object())["type"] == "unknown"


def test_final_text_tracking():
    assert (
        driver._final_text_from_message(AssistantMessage([TextBlock("x"), TextBlock("y")])) == "y"
    )
    assert (
        driver._final_text_from_message(AssistantMessage([ToolUseBlock("i", "Bash", {})])) is None
    )
    assert driver._final_text_from_message(ResultMessage("r")) == "r"
    assert driver._final_text_from_message(object()) is None


def test_is_tool_use_turn():
    # Only assistant messages that used a tool count (mirrors upstream _turn_count).
    assert driver._is_tool_use_turn(AssistantMessage([ToolUseBlock("i", "Bash", {})])) is True
    assert driver._is_tool_use_turn(AssistantMessage([TextBlock("hi")])) is False
    assert driver._is_tool_use_turn(AssistantMessage([])) is False
    assert driver._is_tool_use_turn(ResultMessage("r")) is False
    assert driver._is_tool_use_turn(object()) is False


def test_build_options_wiring():
    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    task = {
        "model": "claude-x",
        "proxy_url": "http://host.docker.internal:9000/mcp",
        "native_tool_deny": ["Bash"],
        "max_turns": 7,
        "system_prompt": "sys",
        "workspace_dir": "/dtap/workspace",
    }
    driver._build_options(task, FakeOptions)

    assert captured["model"] == "claude-x"
    assert captured["permission_mode"] == "bypassPermissions"
    assert captured["mcp_servers"] == {
        "dtap_proxy": {"type": "http", "url": "http://host.docker.internal:9000/mcp"}
    }
    # native menu enabled + proxy glob; deny applied separately
    assert "Bash" in captured["allowed_tools"]
    assert "mcp__dtap_proxy__*" in captured["allowed_tools"]
    assert captured["disallowed_tools"] == ["Bash"]
    assert "include_partial_messages" not in captured  # dropped to match upstream
    assert captured["max_turns"] == 7
    assert captured["system_prompt"] == "sys"
    assert captured["cwd"] == "/dtap/workspace"


def test_build_options_omits_optional_fields():
    captured = {}

    class FakeOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    driver._build_options({"model": "m", "proxy_url": "u"}, FakeOptions)
    assert "system_prompt" not in captured
    assert "cwd" not in captured
    assert captured["max_turns"] == 200  # default
    assert captured["disallowed_tools"] == []


def test_driver_constants_match_trajectory():
    from dtap_claudecode_target import trajectory

    assert driver.TRANSCRIPT_FILENAME == trajectory.TRANSCRIPT_FILENAME
    assert driver.RESULT_FILENAME == trajectory.RESULT_FILENAME
    assert driver.PROXY_SERVER_NAME == trajectory.PROXY_SERVER_NAME
