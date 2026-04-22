"""Comprehensive tests for ChatbotTarget.

Tests cover: security domain structure, config/query/controllable/observable
interfaces, single-turn and multi-turn run flow, system prompt override,
cleanup/teardown lifecycle, and backwards-compatible alias.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.security_domain import SecurityDomainTag, scope_includes

from chatbot_target import (
    MODEL_TAG,
    RESPONSE_READABLE_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    USER_TAG,
    ChatbotTarget,
)

from conftest import make_litellm_response


# ======================================================================
# Security domain structure
# ======================================================================


class TestSecurityDomain:
    """Verify the two-tree forest and tag hierarchy."""

    def test_domain_has_two_roots(self, target: ChatbotTarget) -> None:
        roots = target.security_domain.roots
        root_names = sorted(t.name for t in roots)
        assert root_names == ["system", "user"]

    def test_system_prompt_parent_is_system(self) -> None:
        assert SYSTEM_PROMPT_TAG.parent is SYSTEM_TAG

    def test_system_prompt_readable_parent_is_system_prompt(self) -> None:
        assert SYSTEM_PROMPT_READABLE_TAG.parent is SYSTEM_PROMPT_TAG

    def test_model_parent_is_system(self) -> None:
        assert MODEL_TAG.parent is SYSTEM_TAG

    def test_response_readable_parent_is_model(self) -> None:
        assert RESPONSE_READABLE_TAG.parent is MODEL_TAG

    def test_user_is_independent_root(self) -> None:
        assert USER_TAG.parent is None

    def test_domain_has_six_tags(self, target: ChatbotTarget) -> None:
        combos = target.security_domain.distinct_combinations()
        # 2-tree forest: system tree has 5 tags, user tree has 1 tag
        # System antichains: 10 (empty + 9 non-empty subsets respecting hierarchy)
        # User antichains: 2 ({}, {user})
        # Total: 10 * 2 = 20
        assert len(combos) == 20

    def test_system_prompt_includes_readable(self) -> None:
        assert scope_includes(frozenset({SYSTEM_PROMPT_TAG}), SYSTEM_PROMPT_READABLE_TAG)

    def test_readable_does_not_include_system_prompt(self) -> None:
        assert not scope_includes(frozenset({SYSTEM_PROMPT_READABLE_TAG}), SYSTEM_PROMPT_TAG)

    def test_system_includes_all_children(self) -> None:
        assert scope_includes(frozenset({SYSTEM_TAG}), SYSTEM_PROMPT_TAG)
        assert scope_includes(frozenset({SYSTEM_TAG}), SYSTEM_PROMPT_READABLE_TAG)
        assert scope_includes(frozenset({SYSTEM_TAG}), MODEL_TAG)
        assert scope_includes(frozenset({SYSTEM_TAG}), RESPONSE_READABLE_TAG)

    def test_model_includes_response_readable(self) -> None:
        assert scope_includes(frozenset({MODEL_TAG}), RESPONSE_READABLE_TAG)

    def test_response_readable_does_not_include_model(self) -> None:
        assert not scope_includes(frozenset({RESPONSE_READABLE_TAG}), MODEL_TAG)

    def test_user_does_not_include_system_children(self) -> None:
        assert not scope_includes(frozenset({USER_TAG}), SYSTEM_PROMPT_TAG)
        assert not scope_includes(frozenset({USER_TAG}), MODEL_TAG)
        assert not scope_includes(frozenset({USER_TAG}), RESPONSE_READABLE_TAG)
        assert not scope_includes(frozenset({USER_TAG}), SYSTEM_PROMPT_READABLE_TAG)


# ======================================================================
# Config specs
# ======================================================================


class TestConfig:
    def test_has_system_prompt_config(self, target: ChatbotTarget) -> None:
        specs = target.config_specs
        assert len(specs) == 1
        assert specs[0].name == "system_prompt"
        assert specs[0].security_domain is SYSTEM_PROMPT_TAG

    def test_set_config_updates_system_prompt(self, target: ChatbotTarget) -> None:
        target.set_config("system_prompt", "Be evil.")
        # Observable should reflect the change
        obs = [o for o in target.get_observables() if o.observable.name == "system_prompt"]
        assert obs[0].content == "Be evil."

    def test_set_config_ignores_unknown_name(self, target: ChatbotTarget) -> None:
        target.set_config("unknown_config", "value")
        # No error, no effect
        obs = [o for o in target.get_observables() if o.observable.name == "system_prompt"]
        assert obs[0].content == "You are a helpful assistant."

    def test_default_system_prompt(self, target: ChatbotTarget) -> None:
        obs = [o for o in target.get_observables() if o.observable.name == "system_prompt"]
        assert obs[0].content == "You are a helpful assistant."


# ======================================================================
# Query specs
# ======================================================================


class TestQuery:
    def test_has_last_response_query(self, target: ChatbotTarget) -> None:
        specs = target.query_specs
        names = [s.name for s in specs]
        assert "last_response" in names

    def test_has_conversation_history_query(self, target: ChatbotTarget) -> None:
        specs = target.query_specs
        names = [s.name for s in specs]
        assert "conversation_history" in names

    def test_last_response_initially_empty(self, target: ChatbotTarget) -> None:
        assert target.query("last_response") == ""

    def test_conversation_history_initially_empty(self, target: ChatbotTarget) -> None:
        result = target.query("conversation_history")
        assert json.loads(result) == []

    def test_unknown_query_returns_empty(self, target: ChatbotTarget) -> None:
        assert target.query("nonexistent") == ""


# ======================================================================
# Controllables
# ======================================================================


class TestControllables:
    def test_has_three_controllables(self, target: ChatbotTarget) -> None:
        ctrls = target.get_controllables()
        assert len(ctrls) == 3

    def test_system_prompt_controllable(self, target: ChatbotTarget) -> None:
        ctrls = {c.name: c for c in target.get_controllables()}
        sp = ctrls["system_prompt"]
        assert sp.security_domain is SYSTEM_PROMPT_TAG

    def test_user_message_controllable(self, target: ChatbotTarget) -> None:
        ctrls = {c.name: c for c in target.get_controllables()}
        um = ctrls["user_message"]
        assert um.security_domain is USER_TAG

    def test_response_controllable(self, target: ChatbotTarget) -> None:
        ctrls = {c.name: c for c in target.get_controllables()}
        resp = ctrls["response"]
        assert resp.security_domain is MODEL_TAG


# ======================================================================
# Observables
# ======================================================================


class TestObservables:
    def test_model_observable_at_system_tag(self, target: ChatbotTarget) -> None:
        obs = {o.observable.name: o for o in target.get_observables()}
        assert obs["model"].observable.security_domain is SYSTEM_TAG
        assert obs["model"].content == "test-model"

    def test_system_prompt_observable_at_readable_tag(self, target: ChatbotTarget) -> None:
        obs = {o.observable.name: o for o in target.get_observables()}
        sp = obs["system_prompt"]
        assert sp.observable.security_domain is SYSTEM_PROMPT_READABLE_TAG
        assert sp.content == "You are a helpful assistant."


# ======================================================================
# Run: single turn
# ======================================================================


class TestRunSingleTurn:
    """Verify the event flow for a single-turn conversation."""

    async def test_single_turn_event_flow(self, target: ChatbotTarget) -> None:
        """One injection → LLM call → PostCallEvent → NoInjection → exit."""
        mock_resp = make_litellm_response("Hello from the LLM!")
        events_sent: list[object] = []

        async def mock_send_event(event):
            events_sent.append(event)
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                # First user_message: inject; second: no injection
                if sum(1 for e in events_sent
                       if isinstance(e, ControllablePreCallEvent)
                       and e.controllable.name == "user_message") == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="Hi there",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return None

        emitted: list[object] = []

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(emitted.append, mock_send_event)

        assert target.query("last_response") == "Hello from the LLM!"

    async def test_emits_observable_event_with_response_tag(self, target: ChatbotTarget) -> None:
        mock_resp = make_litellm_response("Response text")
        call_count = 0

        async def mock_send_event(event):
            nonlocal call_count
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                call_count += 1
                if call_count == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="test",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        emitted: list[object] = []

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(emitted.append, mock_send_event)

        obs_events = [e for e in emitted if isinstance(e, ObservableEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].observable.security_domain is RESPONSE_READABLE_TAG
        assert obs_events[0].content == "Response text"

    async def test_postcall_carries_request_and_answer(self, target: ChatbotTarget) -> None:
        mock_resp = make_litellm_response("LLM answer")
        postcalls: list[ControllablePostCallEvent] = []
        call_count = 0

        async def mock_send_event(event):
            nonlocal call_count
            if isinstance(event, ControllablePostCallEvent):
                postcalls.append(event)
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                call_count += 1
                if call_count == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="My question",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return None

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(lambda e: None, mock_send_event)

        assert len(postcalls) == 1
        assert postcalls[0].request == "My question"
        assert postcalls[0].answer == "LLM answer"
        assert postcalls[0].controllable.name == "response"


# ======================================================================
# Run: multi turn
# ======================================================================


class TestRunMultiTurn:
    async def test_multi_turn_accumulates_conversation(self, target: ChatbotTarget) -> None:
        responses = iter(["First reply", "Second reply"])
        user_call_count = 0

        async def mock_send_event(event):
            nonlocal user_call_count
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_call_count += 1
                if user_call_count <= 2:
                    return ControllableInjection(
                        event=event, controllable=event.controllable,
                        value=f"Turn {user_call_count}",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            return make_litellm_response(next(responses))

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        assert target.query("last_response") == "Second reply"
        history = json.loads(target.query("conversation_history"))
        assert len(history) == 5  # system + 2*(user + assistant)
        assert history[0]["role"] == "system"
        assert history[1] == {"role": "user", "content": "Turn 1"}
        assert history[2] == {"role": "assistant", "content": "First reply"}
        assert history[3] == {"role": "user", "content": "Turn 2"}
        assert history[4] == {"role": "assistant", "content": "Second reply"}

    async def test_llm_sees_full_history(self, target: ChatbotTarget) -> None:
        """Verify that each LLM call receives the accumulated conversation."""
        captured_messages: list[list[dict]] = []
        user_call_count = 0

        async def mock_send_event(event):
            nonlocal user_call_count
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_call_count += 1
                if user_call_count <= 2:
                    return ControllableInjection(
                        event=event, controllable=event.controllable,
                        value=f"msg{user_call_count}",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return make_litellm_response("ok")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        # Second LLM call should see full history
        assert len(captured_messages) == 2
        assert len(captured_messages[0]) == 2  # system + user1
        assert len(captured_messages[1]) == 4  # system + user1 + assistant1 + user2


# ======================================================================
# Run: response modification via PostCallEvent
# ======================================================================


class TestResponseModification:
    """Verify that injecting into the ControllablePostCallEvent modifies the response."""

    async def test_injection_overrides_llm_response(self, target: ChatbotTarget) -> None:
        """Optimizer injects into PostCallEvent, conversation uses the modified response."""
        mock_resp = make_litellm_response("Original LLM response")

        async def mock_send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                if not hasattr(mock_send_event, "_user_injected"):
                    mock_send_event._user_injected = True
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="Modified response",
                )
            return None

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(lambda e: None, mock_send_event)

        assert target.query("last_response") == "Modified response"
        history = json.loads(target.query("conversation_history"))
        assert history[-1] == {"role": "assistant", "content": "Modified response"}

    async def test_no_injection_keeps_original_response(self, target: ChatbotTarget) -> None:
        """NoInjection on PostCallEvent leaves the original LLM response."""
        mock_resp = make_litellm_response("Original response")

        async def mock_send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                if not hasattr(mock_send_event, "_user_injected"):
                    mock_send_event._user_injected = True
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return None

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(lambda e: None, mock_send_event)

        assert target.query("last_response") == "Original response"

    async def test_modified_response_in_observable(self, target: ChatbotTarget) -> None:
        """The ObservableEvent emits the modified response, not the original."""
        mock_resp = make_litellm_response("Original")
        emitted: list[object] = []

        async def mock_send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                if not hasattr(mock_send_event, "_user_injected"):
                    mock_send_event._user_injected = True
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="Replaced",
                )
            return None

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(emitted.append, mock_send_event)

        obs_events = [e for e in emitted if isinstance(e, ObservableEvent)]
        assert len(obs_events) == 1
        assert obs_events[0].content == "Replaced"

    async def test_modified_response_in_next_llm_call(self, target: ChatbotTarget) -> None:
        """In multi-turn, the modified response is what the LLM sees in history."""
        captured_messages: list[list[dict]] = []
        user_call_count = 0

        async def mock_send_event(event):
            nonlocal user_call_count
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_call_count += 1
                if user_call_count <= 2:
                    return ControllableInjection(
                        event=event, controllable=event.controllable,
                        value=f"msg{user_call_count}",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="MODIFIED",
                )
            return None

        async def mock_acompletion(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return make_litellm_response("original")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        # Second LLM call should see the modified assistant response in history
        assert len(captured_messages) == 2
        assistant_in_history = [
            m for m in captured_messages[1] if m["role"] == "assistant"
        ]
        assert assistant_in_history[0]["content"] == "MODIFIED"

    async def test_postcall_carries_original_answer(self, target: ChatbotTarget) -> None:
        """The PostCallEvent.answer carries the original LLM response, not a prior modification."""
        mock_resp = make_litellm_response("LLM original")
        postcalls: list[ControllablePostCallEvent] = []

        async def mock_send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                if not hasattr(mock_send_event, "_user_injected"):
                    mock_send_event._user_injected = True
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            if isinstance(event, ControllablePostCallEvent):
                postcalls.append(event)
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="Modified",
                )
            return None

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(lambda e: None, mock_send_event)

        assert postcalls[0].answer == "LLM original"


# ======================================================================
# Run: system prompt override
# ======================================================================


class TestSystemPromptOverride:
    async def test_override_replaces_system_prompt(self, target: ChatbotTarget) -> None:
        captured_messages: list[list[dict]] = []

        async def mock_send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableInjection(
                        event=event, controllable=event.controllable,
                        value="You are a pirate.",
                    )
                # One user turn then stop
                if not hasattr(mock_send_event, "_injected"):
                    mock_send_event._injected = True
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="Ahoy",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return make_litellm_response("Arr!")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        assert captured_messages[0][0] == {"role": "system", "content": "You are a pirate."}

    async def test_no_override_uses_task_configured_prompt(self, target: ChatbotTarget) -> None:
        target.set_config("system_prompt", "Be helpful.")
        captured_messages: list[list[dict]] = []
        user_calls = 0

        async def mock_send_event(event):
            nonlocal user_calls
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_calls += 1
                if user_calls == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return make_litellm_response("ok")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        assert captured_messages[0][0] == {"role": "system", "content": "Be helpful."}

    async def test_precall_request_carries_current_prompt(self, target: ChatbotTarget) -> None:
        """The system_prompt ControllablePreCallEvent.request should carry the current value."""
        target.set_config("system_prompt", "Custom prompt here.")
        sp_event: ControllablePreCallEvent | None = None

        async def mock_send_event(event):
            nonlocal sp_event
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    sp_event = event
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        await target.run(lambda e: None, mock_send_event)

        assert sp_event is not None
        assert sp_event.request == "Custom prompt here."


# ======================================================================
# Run: empty system prompt
# ======================================================================


class TestEmptySystemPrompt:
    async def test_empty_prompt_omits_system_message(self, target: ChatbotTarget) -> None:
        target.set_config("system_prompt", "")
        captured_messages: list[list[dict]] = []
        user_calls = 0

        async def mock_send_event(event):
            nonlocal user_calls
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_calls += 1
                if user_calls == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return make_litellm_response("ok")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        # No system message in conversation
        assert captured_messages[0][0]["role"] == "user"


# ======================================================================
# Run: immediate exit (no user messages)
# ======================================================================


class TestImmediateExit:
    async def test_no_injection_exits_immediately(self, target: ChatbotTarget) -> None:
        async def mock_send_event(event):
            return ControllableNoInjection(event=event, controllable=event.controllable)

        await target.run(lambda e: None, mock_send_event)

        assert target.query("last_response") == ""
        assert json.loads(target.query("conversation_history")) == []


# ======================================================================
# Run: api_base handling
# ======================================================================


class TestApiBase:
    async def test_api_base_passed_when_set(self, target_with_base: ChatbotTarget) -> None:
        captured_kwargs: list[dict] = []
        user_calls = 0

        async def mock_send_event(event):
            nonlocal user_calls
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_calls += 1
                if user_calls == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return make_litellm_response("ok")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target_with_base.run(lambda e: None, mock_send_event)

        assert captured_kwargs[0]["api_base"] == "http://localhost:8080"

    async def test_api_base_omitted_when_none(self, target: ChatbotTarget) -> None:
        captured_kwargs: list[dict] = []
        user_calls = 0

        async def mock_send_event(event):
            nonlocal user_calls
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_calls += 1
                if user_calls == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        async def mock_acompletion(**kwargs):
            captured_kwargs.append(kwargs)
            return make_litellm_response("ok")

        with patch("chatbot_target.target.acompletion", side_effect=mock_acompletion):
            await target.run(lambda e: None, mock_send_event)

        assert "api_base" not in captured_kwargs[0]


# ======================================================================
# Run: LLM response with None content
# ======================================================================


class TestNoneContent:
    async def test_none_content_becomes_empty_string(self, target: ChatbotTarget) -> None:
        mock_resp = make_litellm_response("ignored")
        mock_resp.choices[0].message.content = None
        user_calls = 0

        async def mock_send_event(event):
            nonlocal user_calls
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                user_calls += 1
                if user_calls == 1:
                    return ControllableInjection(
                        event=event, controllable=event.controllable, value="hi",
                    )
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableNoInjection(event=event, controllable=event.controllable)

        with patch("chatbot_target.target.acompletion", return_value=mock_resp):
            await target.run(lambda e: None, mock_send_event)

        assert target.query("last_response") == ""


# ======================================================================
# Cleanup and teardown
# ======================================================================


class TestLifecycle:
    async def test_cleanup_resets_last_response(self, target: ChatbotTarget) -> None:
        target._last_response = "some response"
        await target.cleanup()
        assert target.query("last_response") == ""

    async def test_cleanup_resets_conversation_history(self, target: ChatbotTarget) -> None:
        target._conversation_history = [{"role": "user", "content": "hi"}]
        await target.cleanup()
        assert json.loads(target.query("conversation_history")) == []

    async def test_teardown_is_noop(self, target: ChatbotTarget) -> None:
        await target.teardown()
        # No error, no state change
