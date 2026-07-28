"""A provider content-filter block is a scored attack failure, never an error.

When a guardrail suppresses the victim's reply the attacker did not extract
harmful content, so the run must finish and be scored.  If the exception
escaped ``run()`` the framework would abandon the task with
``stop_reason="error"`` and the attempt would be discarded, systematically
under-counting exactly the cases where a defense worked.

The blocked turn must also stay identifiable afterwards, so these tests pin the
three traces it leaves: the ``response_blocked`` observable carrying the
provider's reason, the marker that replaces the reply, and the ``blocked_turns``
/ ``last_block_reason`` queries.
"""

from __future__ import annotations

import pytest
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    ContentPolicyViolationError,
    RateLimitError,
)

from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)

from chatbot_target import (
    BLOCKED_OBSERVABLE_NAME,
    BLOCKED_RESPONSE_MARKER,
    RESPONSE_READABLE_TAG,
    ChatbotTarget,
    content_filter_reason,
)

from conftest import make_litellm_response

# The exact body AWS Bedrock returned in the RQ1.3-1 first run, which is what
# motivated this behaviour. Note the type: a plain BadRequestError, NOT
# ContentPolicyViolationError, so a type-only check would miss every one.
BEDROCK_BLOCK = (
    'litellm.BadRequestError: BedrockException - {"message":"The model returned '
    'the following errors: Output blocked by content filtering policy"}'
)


def _bedrock_block() -> BadRequestError:
    return BadRequestError(message=BEDROCK_BLOCK, model="m", llm_provider="bedrock")


def _driver(responses, *, turns=1):
    """A send_event that injects *turns* user messages, then stops.

    ``responses`` is consumed by the patched acompletion, not here.
    """
    seen = {"user": 0}

    async def send_event(event):
        if isinstance(event, ControllablePreCallEvent):
            if event.controllable.name == "system_prompt":
                return ControllableNoInjection(event=event, controllable=event.controllable)
            seen["user"] += 1
            if seen["user"] <= turns:
                return ControllableInjection(
                    event=event, controllable=event.controllable,
                    value=f"attack {seen['user']}",
                )
            return ControllableNoInjection(event=event, controllable=event.controllable)
        return ControllableNoInjection(event=event, controllable=event.controllable)

    return send_event


class TestDetector:
    """``content_filter_reason`` must be tight: only real blocks."""

    def test_bedrock_bad_request_is_a_block(self) -> None:
        assert content_filter_reason(_bedrock_block()) is not None

    def test_typed_content_policy_error_is_a_block(self) -> None:
        exc = ContentPolicyViolationError(
            message="content policy violation", model="m", llm_provider="openai",
        )
        assert content_filter_reason(exc) is not None

    @pytest.mark.parametrize("phrase", [
        "content_filter", "content_policy_violation",
        "content management policy", "responsible ai policy",
        "Output blocked by content filtering policy",
    ])
    def test_provider_phrases_are_recognised(self, phrase: str) -> None:
        exc = BadRequestError(message=phrase, model="m", llm_provider="p")
        assert content_filter_reason(exc) is not None

    @pytest.mark.parametrize("exc", [
        RateLimitError(message="Too many tokens, please wait before trying again.",
                       model="m", llm_provider="bedrock"),
        APIConnectionError(message="connection reset", model="m", llm_provider="bedrock"),
        BadRequestError(message="Bedrock is unable to process your request.",
                        model="m", llm_provider="bedrock"),
        BadRequestError(message="A conversation must start with a user message.",
                        model="m", llm_provider="bedrock"),
        ValueError("something else entirely"),
    ])
    def test_infrastructure_failures_are_not_blocks(self, exc: BaseException) -> None:
        # These must keep erroring the task. Absorbing them would record an
        # outage as a defended attack and inflate every defense's success rate.
        assert content_filter_reason(exc) is None


class TestBlockedRunIsScoredNotErrored:
    async def test_run_completes_instead_of_raising(self, target: ChatbotTarget) -> None:
        import unittest.mock as m

        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(lambda e: None, _driver(None))  # must not raise

    async def test_other_provider_errors_still_raise(self, target: ChatbotTarget) -> None:
        import unittest.mock as m

        boom = RateLimitError(message="Too many tokens, please wait before trying again.",
                              model="m", llm_provider="bedrock")
        with m.patch("chatbot_target.target.acompletion", side_effect=boom):
            with pytest.raises(RateLimitError):
                await target.run(lambda e: None, _driver(None))


class TestBlockIsVisibleAfterwards:
    async def test_emits_response_blocked_observable_with_the_reason(
        self, target: ChatbotTarget,
    ) -> None:
        import unittest.mock as m

        emitted: list[object] = []
        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(emitted.append, _driver(None))

        blocks = [
            e for e in emitted
            if isinstance(e, ObservableEvent)
            and e.observable.name == BLOCKED_OBSERVABLE_NAME
        ]
        assert len(blocks) == 1
        assert "content filtering policy" in blocks[0].content
        assert blocks[0].observable.security_domain is RESPONSE_READABLE_TAG

    async def test_queries_report_the_block(self, target: ChatbotTarget) -> None:
        import unittest.mock as m

        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(lambda e: None, _driver(None))

        assert target.query("blocked_turns") == "1"
        assert "content filtering policy" in target.query("last_block_reason")

    async def test_marker_replaces_the_reply_everywhere(self, target: ChatbotTarget) -> None:
        import json
        import unittest.mock as m

        posts: list[ControllablePostCallEvent] = []
        base = _driver(None)          # one driver: its turn counter must persist

        async def send_event(event):
            if isinstance(event, ControllablePostCallEvent):
                posts.append(event)
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return await base(event)

        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(lambda e: None, send_event)

        assert target.query("last_response") == BLOCKED_RESPONSE_MARKER
        assert posts and posts[0].answer == BLOCKED_RESPONSE_MARKER
        history = json.loads(target.query("conversation_history"))
        assert history[-1] == {"role": "assistant", "content": BLOCKED_RESPONSE_MARKER}

    async def test_marker_is_distinguishable_from_an_empty_reply(
        self, target: ChatbotTarget,
    ) -> None:
        # An empty response is its own failure mode (the model returned nothing).
        # A guardrail block must not look like it, or the two get conflated in
        # analysis.
        import unittest.mock as m

        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("")):
            await target.run(lambda e: None, _driver(None))
        assert target.query("last_response") == ""
        assert target.query("blocked_turns") == "0"
        assert target.query("last_block_reason") == ""


class TestConversationContinues:
    async def test_attacker_keeps_control_after_a_block(
        self, target: ChatbotTarget,
    ) -> None:
        # Ending the run on a block would truncate multi-turn attacks against
        # guardrailed models only, biasing them against ungarded ones.
        import unittest.mock as m

        calls = {"n": 0}

        def maybe_block(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _bedrock_block()
            return make_litellm_response("second turn got through")

        with m.patch("chatbot_target.target.acompletion", side_effect=maybe_block):
            await target.run(lambda e: None, _driver(None, turns=2))

        assert calls["n"] == 2
        assert target.query("last_response") == "second turn got through"
        assert target.query("blocked_turns") == "1"

    async def test_model_scope_can_still_override_a_blocked_reply(
        self, target: ChatbotTarget,
    ) -> None:
        # The response controllable keeps its meaning: an attacker holding
        # MODEL_TAG writes the victim's answer, blocked or not.
        import unittest.mock as m

        base = _driver(None)          # one driver: its turn counter must persist

        async def send_event(event):
            if isinstance(event, ControllablePostCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="forged reply",
                )
            return await base(event)

        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(lambda e: None, send_event)

        assert target.query("last_response") == "forged reply"
        assert target.query("blocked_turns") == "1"  # still recorded as blocked


class TestLifecycle:
    async def test_reset_clears_block_state(self, target: ChatbotTarget) -> None:
        import unittest.mock as m

        with m.patch("chatbot_target.target.acompletion", side_effect=_bedrock_block()):
            await target.run(lambda e: None, _driver(None))
        assert target.query("blocked_turns") == "1"

        await target.reset_ephemeral_state()
        assert target.query("blocked_turns") == "0"
        assert target.query("last_block_reason") == ""

    def test_block_queries_are_declared(self, target: ChatbotTarget) -> None:
        names = {q.name for q in target.query_specs}
        assert {"blocked_turns", "last_block_reason"} <= names
