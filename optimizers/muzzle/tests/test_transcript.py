"""Tests for muzzle_optimizer.transcript.

Covers build_transcript, find_victim_system_prompt, recover_user_goal,
recover_action_url, and extract_placeholder_template.
"""

from __future__ import annotations

import pytest

from superred.core.types.events import ControllableInjection

from muzzle_optimizer.constants import PLACEHOLDER_ALIAS, PLACEHOLDER_INSTR
from muzzle_optimizer.transcript import (
    PlaceholderNotFoundError,
    build_transcript,
    extract_placeholder_template,
    find_victim_system_prompt,
    recover_action_url,
    recover_user_goal,
)

from conftest import (
    SYSTEM_PROMPT_TAG,
    TOOL_A_TAG,
    TRACE_TAG,
    USER_TAG,
    make_controllable,
    make_observable_value,
    observable_event,
    post_call,
    run_end,
)

# ---------------------------------------------------------------------------
# build_transcript
# ---------------------------------------------------------------------------

_CTRL = make_controllable("tool:fetch", TOOL_A_TAG, "text", "fetch tool")
_EMPTY_RESP_NAMES: frozenset[str] = frozenset()


def _make_injection(pc_event: object, value: str) -> ControllableInjection:
    return ControllableInjection(event=pc_event, controllable=_CTRL, value=value)  # type: ignore[arg-type]


def test_build_transcript_four_items_produce_four_records() -> None:
    """system_prompt obs + trace obs + post_call + injection -> 4 transcript records."""
    pc_event = post_call(_CTRL, request="req", answer="original")
    injection = _make_injection(pc_event, "PAYLOAD")
    items: list[object] = [
        observable_event("system_prompt", "You are a helpful agent.", SYSTEM_PROMPT_TAG),
        observable_event("agent_trace_message_0001", "I will proceed.", TRACE_TAG),
        pc_event,
        injection,
    ]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert len(result) == 4


def test_build_transcript_system_observable_gets_system_role() -> None:
    """Observable named 'system_prompt' maps to role 'system'."""
    items: list[object] = [
        observable_event("system_prompt", "You are an agent.", SYSTEM_PROMPT_TAG),
    ]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an agent."


def test_build_transcript_trace_message_observable_gets_assistant_role() -> None:
    """Observable named 'agent_trace_message_0001' contains 'trace_message' -> role 'assistant'."""
    items: list[object] = [
        observable_event("agent_trace_message_0001", "Calling the tool.", TRACE_TAG),
    ]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "Calling the tool."


def test_build_transcript_post_call_is_tool_record_with_request_and_answer() -> None:
    """ControllablePostCallEvent becomes a tool record carrying request and answer."""
    pc = post_call(_CTRL, request="https://example.com", answer="fetched content")
    items: list[object] = [pc]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert len(result) == 1
    rec = result[0]
    assert rec["role"] == "tool"
    assert rec["tool"] == "tool:fetch"
    assert rec["request"] == "https://example.com"
    assert rec["answer"] == "fetched content"


def test_build_transcript_injection_answer_is_injected_value() -> None:
    """ControllableInjection appears as a tool record; its 'answer' is the injected value."""
    pc = post_call(_CTRL)
    injection = _make_injection(pc, "[INSTRUCTION_PLACEHOLDER]")
    items: list[object] = [injection]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert len(result) == 1
    rec = result[0]
    assert rec["role"] == "tool"
    assert rec["answer"] == "[INSTRUCTION_PLACEHOLDER]"
    # request is empty for injections
    assert rec["request"] == ""


def test_build_transcript_run_end_event_is_skipped() -> None:
    """RunEndEvent carries lifecycle feedback and must not appear in the transcript."""
    items: list[object] = [
        observable_event("system_prompt", "System.", SYSTEM_PROMPT_TAG),
        run_end(success=False, score=0.0),
    ]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    # Only the system observable ends up in the transcript; run_end is filtered.
    assert len(result) == 1
    assert result[0]["role"] == "system"


def test_build_transcript_ordering_preserved() -> None:
    """Output messages appear in the same order as the input items."""
    ctrl_b = make_controllable("tool:write", TOOL_A_TAG)
    pc_a = post_call(_CTRL, answer="first")
    pc_b = post_call(ctrl_b, answer="second")
    items: list[object] = [pc_a, pc_b]
    result = build_transcript(items, response_observable_names=_EMPTY_RESP_NAMES)
    assert result[0]["answer"] == "first"
    assert result[1]["answer"] == "second"


def test_build_transcript_response_observable_names_override_role() -> None:
    """An observable whose name is in response_observable_names maps to 'assistant'."""
    items: list[object] = [
        observable_event("model_reply", "Sure thing.", TRACE_TAG),
    ]
    result = build_transcript(items, response_observable_names=frozenset({"model_reply"}))
    assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# find_victim_system_prompt
# ---------------------------------------------------------------------------


def test_find_victim_system_prompt_prefers_static_observable() -> None:
    """Static observable named 'system_prompt' is preferred over an ObservableEvent."""
    static = make_observable_value("system_prompt", "Static system prompt.", SYSTEM_PROMPT_TAG)
    items: list[object] = [
        observable_event("system_prompt", "Live event system prompt.", SYSTEM_PROMPT_TAG),
    ]
    result = find_victim_system_prompt(items, [static])
    assert result == "Static system prompt."


def test_find_victim_system_prompt_falls_back_to_observable_event() -> None:
    """When no static 'system_prompt' observable exists, fall back to an event with 'system' in name."""
    items: list[object] = [
        observable_event("system_instructions", "Fallback system text.", SYSTEM_PROMPT_TAG),
    ]
    result = find_victim_system_prompt(items, [])
    assert result == "Fallback system text."


def test_find_victim_system_prompt_ignores_non_system_events() -> None:
    """An observable without 'system' in its name is not treated as a system prompt."""
    items: list[object] = [
        observable_event("agent_trace", "Not a system prompt.", TRACE_TAG),
    ]
    result = find_victim_system_prompt(items, [])
    assert result is None


def test_find_victim_system_prompt_returns_none_when_absent() -> None:
    """Returns None when no system prompt can be recovered from either source."""
    result = find_victim_system_prompt([], [])
    assert result is None


def test_find_victim_system_prompt_skips_empty_static_observable() -> None:
    """A static observable whose content is whitespace-only is ignored."""
    static = make_observable_value("system_prompt", "   ", SYSTEM_PROMPT_TAG)
    items: list[object] = [
        observable_event("system_prompt", "Real content.", SYSTEM_PROMPT_TAG),
    ]
    result = find_victim_system_prompt(items, [static])
    assert result == "Real content."


# ---------------------------------------------------------------------------
# recover_user_goal
# ---------------------------------------------------------------------------


def test_recover_user_goal_from_static_observable() -> None:
    """Observable with 'task' in its name and string content is returned as the goal."""
    obs = make_observable_value("user_task", "book a flight to Paris", USER_TAG)
    result = recover_user_goal([], [obs])
    assert result == "book a flight to Paris"


def test_recover_user_goal_from_static_observable_with_query_hint() -> None:
    """Observable with 'query' hint also matches."""
    obs = make_observable_value("search_query", "latest weather", USER_TAG)
    result = recover_user_goal([], [obs])
    assert result == "latest weather"


def test_recover_user_goal_from_transcript_user_role_message() -> None:
    """Falls back to the first user-role transcript message when no matching observable."""
    # "user_message" -> 'user' hint -> role "user"
    items: list[object] = [
        observable_event("user_message", "send the report", USER_TAG),
    ]
    result = recover_user_goal(items, [])
    assert result == "send the report"


def test_recover_user_goal_skips_non_user_roles() -> None:
    """Assistant-role events are not returned as the goal."""
    items: list[object] = [
        observable_event("agent_trace_message_0001", "I am thinking.", TRACE_TAG),
    ]
    result = recover_user_goal(items, [])
    assert result is None


def test_recover_user_goal_returns_none_when_absent() -> None:
    """Returns None when no user goal can be recovered."""
    result = recover_user_goal([], [])
    assert result is None


# ---------------------------------------------------------------------------
# recover_action_url
# ---------------------------------------------------------------------------


def test_recover_action_url_from_static_observable() -> None:
    """Observable with 'url' in its name supplies the action URL."""
    obs = make_observable_value("start_url", "https://app.example.com/home", USER_TAG)
    result = recover_action_url([], [obs])
    assert result == "https://app.example.com/home"


def test_recover_action_url_from_system_message_text() -> None:
    """Falls back to the first http(s) URL found in a system-role transcript message."""
    items: list[object] = [
        observable_event(
            "system_prompt",
            "Visit https://example.com/start to begin your task.",
            SYSTEM_PROMPT_TAG,
        ),
    ]
    result = recover_action_url(items, [])
    assert result == "https://example.com/start"


def test_recover_action_url_from_user_message_text() -> None:
    """Also scans user-role messages for a URL."""
    items: list[object] = [
        observable_event("user_message", "Go to https://target.example.com/page", USER_TAG),
    ]
    result = recover_action_url(items, [])
    assert result == "https://target.example.com/page"


def test_recover_action_url_strips_trailing_punctuation() -> None:
    """Trailing punctuation characters are stripped from the recovered URL."""
    items: list[object] = [
        observable_event(
            "system_prompt", "Navigate to https://example.com/task.", SYSTEM_PROMPT_TAG
        ),
    ]
    result = recover_action_url(items, [])
    assert result == "https://example.com/task"


def test_recover_action_url_returns_none_when_absent() -> None:
    """Returns None when no http(s) URL is found in any candidate."""
    items: list[object] = [
        observable_event("system_prompt", "No URL here.", SYSTEM_PROMPT_TAG),
    ]
    result = recover_action_url(items, [])
    assert result is None


def test_recover_action_url_does_not_scan_assistant_messages() -> None:
    """Assistant-role messages are not checked for URLs."""
    items: list[object] = [
        observable_event(
            "agent_trace_message_0001",
            "I will visit https://assistant-url.example.com",
            TRACE_TAG,
        ),
    ]
    result = recover_action_url(items, [])
    assert result is None


# ---------------------------------------------------------------------------
# extract_placeholder_template
# ---------------------------------------------------------------------------


def test_extract_placeholder_template_finds_token_in_tool_answer() -> None:
    """The placeholder token in a tool-record answer is found and returned."""
    transcript = [
        {"role": "system", "content": "You are a helpful agent."},
        {
            "role": "tool",
            "tool": "tool:fetch",
            "request": "",
            "answer": f"Important: {PLACEHOLDER_INSTR}",
        },
    ]
    step_text, system_prompt = extract_placeholder_template(transcript, (PLACEHOLDER_INSTR,))
    assert PLACEHOLDER_INSTR in step_text
    assert system_prompt == "You are a helpful agent."


def test_extract_placeholder_template_returns_most_recent_prior_system() -> None:
    """The system prompt returned is the one that immediately preceded the token."""
    transcript = [
        {"role": "system", "content": "First system."},
        {"role": "assistant", "content": "Thinking..."},
        {"role": "system", "content": "Second system."},
        {"role": "tool", "request": "", "answer": PLACEHOLDER_INSTR},
    ]
    _, system_prompt = extract_placeholder_template(transcript, (PLACEHOLDER_INSTR,))
    assert system_prompt == "Second system."


def test_extract_placeholder_template_no_prior_system_returns_none() -> None:
    """When no system message precedes the token, accompanying_system_prompt is None."""
    transcript = [
        {"role": "tool", "request": "", "answer": PLACEHOLDER_INSTR},
    ]
    _, system_prompt = extract_placeholder_template(transcript, (PLACEHOLDER_INSTR,))
    assert system_prompt is None


def test_extract_placeholder_template_alias_token_accepted() -> None:
    """PLACEHOLDER_ALIAS ('[PLACEHOLDER]') is also a valid search token."""
    transcript = [
        {"role": "tool", "request": "", "answer": f"do this: {PLACEHOLDER_ALIAS}"},
    ]
    step_text, _ = extract_placeholder_template(transcript, (PLACEHOLDER_ALIAS,))
    assert PLACEHOLDER_ALIAS in step_text


def test_extract_placeholder_template_finds_token_in_content() -> None:
    """Token embedded inside an assistant 'content' message is also found."""
    transcript = [
        {"role": "assistant", "content": f"Please {PLACEHOLDER_INSTR} now."},
    ]
    step_text, _ = extract_placeholder_template(transcript, (PLACEHOLDER_INSTR,))
    assert PLACEHOLDER_INSTR in step_text


def test_extract_placeholder_template_raises_when_absent() -> None:
    """PlaceholderNotFoundError is raised when no token appears anywhere."""
    transcript = [
        {"role": "system", "content": "Clean system prompt."},
        {"role": "tool", "request": "q", "answer": "no marker here"},
    ]
    with pytest.raises(PlaceholderNotFoundError):
        extract_placeholder_template(transcript, (PLACEHOLDER_INSTR,))


def test_extract_placeholder_template_raises_on_empty_transcript() -> None:
    """Empty transcript raises PlaceholderNotFoundError."""
    with pytest.raises(PlaceholderNotFoundError):
        extract_placeholder_template([], (PLACEHOLDER_INSTR,))
