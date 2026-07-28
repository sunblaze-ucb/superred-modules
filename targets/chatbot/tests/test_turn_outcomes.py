"""Every abnormal victim turn is classified, recorded, and never faked.

These are replays of the RQ1.3-1 first run, where one call site with no
validation, no timeout and no retries produced 12,845 dead tasks and 52,453
runs that were silently scored 0.0.  Each class below is one of those failures:

* the attacker injects ``""`` / whitespace / ``None`` and the provider rejects
  the conversation it produces (674 tasks);
* the provider returns no content field at all, in band or as a Responses API
  ``ValueError`` (35 tasks errored, plus the silent majority);
* a transient provider failure kills the whole task on the first try
  (11,711 tasks);
* the victim replies with a blank string and it is indistinguishable from all
  of the above (52,453 runs).

The invariant every test here serves: an infrastructure failure must never be
recorded as a score-0 attacker failure, a defense must never be recorded as an
error, and every abnormal outcome must still be findable in stored data.
"""

from __future__ import annotations

import itertools
import json
import unittest.mock as m

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    Timeout,
)

from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)

from chatbot_target import (
    INVALID_INJECTION_OBSERVABLE_NAME,
    MISSING_OBSERVABLE_NAME,
    OUTCOME_EMPTY,
    OUTCOME_INVALID_INJECTION,
    OUTCOME_NO_CONTENT,
    OUTCOME_OK,
    RETRY_OBSERVABLE_NAME,
    USER_TAG,
    ChatbotTarget,
    invalid_injection_reason,
    no_content_reason,
)

from conftest import make_litellm_response

# Verbatim from the first-run archive: the two errors the empty injection
# produced, on the first turn and on a later one respectively.
BEDROCK_NO_LEADING_USER = (
    'litellm.BadRequestError: BedrockException - {"message":"A conversation must '
    'start with a user message. Try again with a conversation that starts with a '
    'user message."}'
)
BEDROCK_PREFILL = (
    'litellm.BadRequestError: BedrockException - {"message":"This model does not '
    'support assistant message prefill. Try again without an assistant message at '
    'the end of the conversation."}'
)


def _driver(values, *, posts=None):
    """A send_event that injects each of *values* in turn, then stops.

    ``values`` may hold anything, including the ``None`` and ``""`` that the
    attacker modules actually emitted.
    """
    it = iter(values)

    async def send_event(event):
        if isinstance(event, ControllablePreCallEvent):
            if event.controllable.name == "system_prompt":
                return ControllableNoInjection(event=event, controllable=event.controllable)
            try:
                value = next(it)
            except StopIteration:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            return ControllableInjection(
                event=event, controllable=event.controllable, value=value,
            )
        if isinstance(event, ControllablePostCallEvent) and posts is not None:
            posts.append(event)
        return ControllableNoInjection(event=event, controllable=event.controllable)

    return send_event


def _outcomes(target: ChatbotTarget) -> list[dict]:
    return json.loads(target.query("turn_outcomes"))


def _observables(emitted: list[object], name: str) -> list[ObservableEvent]:
    return [
        e for e in emitted
        if isinstance(e, ObservableEvent) and e.observable.name == name
    ]


# ======================================================================
# The attacker's injected value is not a usable message
# ======================================================================


class TestInvalidInjection:
    """674 first-run tasks died because the value was forwarded unchecked."""

    @pytest.mark.parametrize("value,reason", [
        (None, "none"),
        ("", "blank"),
        ("   ", "blank"),
        ("\n\t", "blank"),
        (42, "not_a_string"),
        (["a"], "not_a_string"),
    ])
    def test_unusable_values_are_named(self, value: object, reason: str) -> None:
        assert invalid_injection_reason(value) == reason

    @pytest.mark.parametrize("value", ["hi", " padded ", "0"])
    def test_usable_values_pass(self, value: str) -> None:
        assert invalid_injection_reason(value) is None

    async def test_empty_first_message_is_not_sent(self, target: ChatbotTarget) -> None:
        # Replays the 168 "A conversation must start with a user message" tasks:
        # litellm drops the empty user turn, the request then has no user
        # message, and Bedrock rejects it. Nothing is sent, so nothing rejects.
        emitted: list[object] = []
        with m.patch("chatbot_target.target.acompletion") as call:
            await target.run(emitted.append, _driver([""]))

        assert call.call_count == 0
        assert target.query("turns_completed") == "0"
        assert target.query("last_turn_outcome") == OUTCOME_INVALID_INJECTION
        assert _outcomes(target)[0]["reason"] == "blank"

        invalid = _observables(emitted, INVALID_INJECTION_OBSERVABLE_NAME)
        assert len(invalid) == 1
        assert invalid[0].observable.security_domain is USER_TAG
        assert invalid[0].content["reason"] == "blank"

    async def test_none_value_is_not_sent(self, target: ChatbotTarget) -> None:
        # Replays the 37 Crescendo tasks that injected None: the target used to
        # append {"role": "user", "content": None}, which litellm turns into a
        # message with no content key -> KeyError inside the provider adapter.
        with m.patch("chatbot_target.target.acompletion") as call:
            await target.run(lambda e: None, _driver([None]))

        assert call.call_count == 0
        assert _outcomes(target)[0]["reason"] == "none"

    async def test_a_late_empty_message_does_not_discard_the_run(
        self, target: ChatbotTarget,
    ) -> None:
        # Replays the 464 "assistant message prefill" tasks. In the archive's
        # worked example, nine completed refusal turns were thrown away because
        # the tenth injection was "". The completed turns must survive.
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("a real reply")) as call:
            await target.run(lambda e: None, _driver(["turn one", ""]))

        assert call.call_count == 1
        assert target.query("last_response") == "a real reply"
        assert target.query("turns_completed") == "1"
        assert [o["outcome"] for o in _outcomes(target)] == [
            OUTCOME_OK, OUTCOME_INVALID_INJECTION,
        ]

    async def test_the_run_ends_after_repeated_unusable_messages(
        self, target: ChatbotTarget,
    ) -> None:
        # An unusable message makes no provider call, so it meets no rate limit
        # and no cost budget: an attacker stuck emitting "" would loop forever.
        async def send_event(event):
            if isinstance(event, ControllablePreCallEvent):
                if event.controllable.name == "system_prompt":
                    return ControllableNoInjection(event=event, controllable=event.controllable)
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="",
                )
            return ControllableNoInjection(event=event, controllable=event.controllable)

        with m.patch("chatbot_target.target.acompletion") as call:
            await target.run(lambda e: None, send_event)

        assert call.call_count == 0
        assert len(_outcomes(target)) == 3          # max_consecutive_invalid

    async def test_a_valid_message_resets_the_counter(self, target: ChatbotTarget) -> None:
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("ok")) as call:
            await target.run(lambda e: None, _driver(["", "", "real", "", "", "also real"]))

        assert call.call_count == 2
        assert len(_outcomes(target)) == 6

    async def test_the_record_shows_the_turn_without_inventing_a_message(
        self, target: ChatbotTarget,
    ) -> None:
        with m.patch("chatbot_target.target.acompletion"):
            await target.run(lambda e: None, _driver([""]))

        history = json.loads(target.query("conversation_history"))
        assert history[-1]["content"] is None
        assert history[-1]["outcome"] == OUTCOME_INVALID_INJECTION


# ======================================================================
# The provider returned no content
# ======================================================================


class TestNoContent:
    """Not a reply and not a refusal: there is nothing to score."""

    @pytest.mark.parametrize("message", [
        "Unknown items in responses API response: []",
        "gpt-5.4 unable to complete request: max_output_tokens",
    ])
    def test_responses_api_errors_are_recognised(self, message: str) -> None:
        assert no_content_reason(ValueError(message)) is not None

    @pytest.mark.parametrize("exc", [
        ValueError("something else entirely"),
        RateLimitError(message="Too many tokens", model="m", llm_provider="bedrock"),
    ])
    def test_other_errors_are_not_no_content(self, exc: BaseException) -> None:
        assert no_content_reason(exc) is None

    async def test_none_content_is_not_an_empty_reply(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # This used to be `content or ""`, which the claims then scored as a
        # legitimate 0.0: an attacker failure that never happened.
        emitted: list[object] = []
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response(None)) as call:
            await fast_target.run(emitted.append, _driver(["hi"]))

        assert call.call_count == 3                       # max_attempts
        assert fast_target.query("last_turn_outcome") == OUTCOME_NO_CONTENT
        assert fast_target.query("turns_completed") == "0"
        assert fast_target.query("last_response") == ""
        assert len(_observables(emitted, MISSING_OBSERVABLE_NAME)) == 1

    async def test_no_choices_is_no_content(self, fast_target: ChatbotTarget) -> None:
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response(None, choices=False)):
            await fast_target.run(lambda e: None, _driver(["hi"]))

        assert fast_target.query("last_turn_outcome") == OUTCOME_NO_CONTENT

    async def test_responses_api_valueerror_does_not_kill_the_task(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # The 35 gepa/goat tasks: the same event as an in-band empty reply, but
        # raised, because the victim was served over the Responses API. One
        # event must not get two dispositions depending on the transport.
        boom = ValueError("Unknown items in responses API response: []")
        with m.patch("chatbot_target.target.acompletion", side_effect=boom):
            await fast_target.run(lambda e: None, _driver(["hi"]))

        assert fast_target.query("last_turn_outcome") == OUTCOME_NO_CONTENT

    async def test_a_retry_that_recovers_yields_a_normal_turn(
        self, fast_target: ChatbotTarget,
    ) -> None:
        emitted: list[object] = []
        replies = [make_litellm_response(None), make_litellm_response("got it")]
        with m.patch("chatbot_target.target.acompletion", side_effect=replies) as call:
            await fast_target.run(emitted.append, _driver(["hi"]))

        assert call.call_count == 2
        assert fast_target.query("last_response") == "got it"
        assert fast_target.query("last_turn_outcome") == OUTCOME_OK
        assert _outcomes(fast_target)[0]["attempts"] == 2
        # A turn that silently cost two provider calls must not look free.
        assert len(_observables(emitted, RETRY_OBSERVABLE_NAME)) == 1

    async def test_nothing_is_appended_to_the_wire(self, fast_target: ChatbotTarget) -> None:
        sent: list[list[dict]] = []
        calls = {"n": 0}

        def maybe_missing(**kwargs):
            sent.append(list(kwargs["messages"]))
            calls["n"] += 1
            # No content for every attempt of turn 1, then a reply for turn 2.
            return make_litellm_response(None if calls["n"] <= 3 else "later")

        with m.patch("chatbot_target.target.acompletion", side_effect=maybe_missing):
            await fast_target.run(lambda e: None, _driver(["one", "two"]))

        # An empty assistant turn is deleted by litellm (merging the attacker's
        # two user turns into one, invisibly) or rejected as a prefill.
        assert all(msg["role"] != "assistant" for msg in sent[-1])

    async def test_non_text_content_is_no_content_not_a_reply(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # Whatever a provider adapter put there, it is not the victim's words,
        # so it is not stringified into the measurement.
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response([{"type": "text"}])):
            await fast_target.run(lambda e: None, _driver(["hi"]))

        assert fast_target.query("last_turn_outcome") == OUTCOME_NO_CONTENT
        assert fast_target.query("last_response") == ""
        assert "list" in _outcomes(fast_target)[0]["reason"]

    async def test_a_missing_finish_reason_is_recorded_as_absent(
        self, target: ChatbotTarget,
    ) -> None:
        resp = make_litellm_response("hello")
        del resp.choices[0].finish_reason
        with m.patch("chatbot_target.target.acompletion", return_value=resp):
            await target.run(lambda e: None, _driver(["hi"]))

        assert _outcomes(target)[0]["finish_reason"] is None
        assert target.query("last_turn_outcome") == OUTCOME_OK


# ======================================================================
# The victim replied with a blank string
# ======================================================================


class TestEmptyReply:
    """A real reply that says nothing. Scored, but flagged."""

    async def test_empty_reply_is_not_retried(self, target: ChatbotTarget) -> None:
        # Retrying re-samples a stochastic victim with the same prompt, so a
        # completed reply is never re-rolled: that would bias attack success
        # upward by giving the attacker a second draw.
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("")) as call:
            await target.run(lambda e: None, _driver(["hi"]))

        assert call.call_count == 1
        assert target.query("last_turn_outcome") == OUTCOME_EMPTY
        assert target.query("turns_completed") == "1"

    async def test_the_reasoning_only_case_is_recorded(self, target: ChatbotTarget) -> None:
        # A reasoning model that spent its budget thinking returns a blank
        # answer with reasoning attached. That is a different event from a model
        # that had nothing to say, and the archive cannot tell them apart today.
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("", reasoning_content="thinking...")):
            await target.run(lambda e: None, _driver(["hi"]))

        assert _outcomes(target)[0]["has_reasoning_content"] is True

    async def test_turns_completed_separates_the_two_zero_content_runs(
        self, target: ChatbotTarget,
    ) -> None:
        # Both runs end with last_response == "". In one the victim answered and
        # said nothing; in the other no exchange ever happened.
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("")):
            await target.run(lambda e: None, _driver(["hi"]))
        assert target.query("turns_completed") == "1"

        await target.reset_ephemeral_state()
        with m.patch("chatbot_target.target.acompletion"):
            await target.run(lambda e: None, _driver([""]))
        assert target.query("turns_completed") == "0"


# ======================================================================
# Transient vs persistent provider failures
# ======================================================================


class TestRetries:
    @pytest.mark.parametrize("exc", [
        RateLimitError(message="Too many tokens", model="m", llm_provider="bedrock"),
        APIConnectionError(message="connection reset", model="m", llm_provider="bedrock"),
        InternalServerError(message="internal_server_error", model="m", llm_provider="bedrock"),
        Timeout(message="timed out", model="m", llm_provider="bedrock"),
    ])
    async def test_a_transient_failure_is_retried_and_recovers(
        self, fast_target: ChatbotTarget, exc: BaseException,
    ) -> None:
        # 11,711 first-run tasks died on one of these, on the first try.
        outcomes = [exc, make_litellm_response("recovered")]
        emitted: list[object] = []
        with m.patch("chatbot_target.target.acompletion", side_effect=outcomes) as call:
            await fast_target.run(emitted.append, _driver(["hi"]))

        assert call.call_count == 2
        assert fast_target.query("last_response") == "recovered"
        assert _outcomes(fast_target)[0]["attempts"] == 2
        assert len(_observables(emitted, RETRY_OBSERVABLE_NAME)) == 1

    async def test_an_exhausted_transient_failure_raises_the_original(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # It really is infrastructure, so it must stay loud -- and it must be
        # the provider's own exception, because TaskResult.error is where the
        # failure is diagnosed from afterwards.
        boom = APIConnectionError(message="connection reset", model="m", llm_provider="bedrock")
        with m.patch("chatbot_target.target.acompletion", side_effect=boom) as call:
            with pytest.raises(APIConnectionError) as caught:
                await fast_target.run(lambda e: None, _driver(["hi"]))

        assert caught.value is boom
        assert call.call_count == 3

    @pytest.mark.parametrize("exc", [
        AuthenticationError(message="Unable to locate credentials", model="m",
                            llm_provider="bedrock"),
        ContextWindowExceededError(message="too many tokens", model="m",
                                   llm_provider="bedrock"),
    ])
    async def test_a_persistent_failure_is_not_retried(
        self, fast_target: ChatbotTarget, exc: BaseException,
    ) -> None:
        with m.patch("chatbot_target.target.acompletion", side_effect=exc) as call:
            with pytest.raises(type(exc)):
                await fast_target.run(lambda e: None, _driver(["hi"]))

        assert call.call_count == 1

    async def test_retrying_stops_at_the_elapsed_budget(self) -> None:
        # The budget exists so retries cannot eat the controller's per-task time
        # cap while holding one of the run's concurrency slots.
        target = ChatbotTarget(
            model="test-model", api_key="sk-test",
            retry_backoff_s=0.0, max_attempts=10, retry_max_elapsed_s=5.0,
        )
        # A clock that jumps past the budget between the first two readings.
        clock = itertools.count(0.0, 9.0)
        boom = APIConnectionError(message="connection reset", model="m", llm_provider="bedrock")

        with m.patch("chatbot_target.target.time.monotonic", side_effect=lambda: next(clock)):
            with m.patch("chatbot_target.target.acompletion", side_effect=boom) as call:
                with pytest.raises(APIConnectionError):
                    await target.run(lambda e: None, _driver(["hi"]))

        assert call.call_count == 1

    async def test_the_retry_observable_is_not_attacker_visible(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # How often the harness redialled is infrastructure telemetry, not a
        # capability of the modelled attacker. It sits at the system root, which
        # no experiment scope in RQ1.3-1 includes.
        from chatbot_target import SYSTEM_TAG

        emitted: list[object] = []
        outcomes = [
            RateLimitError(message="Too many tokens", model="m", llm_provider="bedrock"),
            make_litellm_response("ok"),
        ]
        with m.patch("chatbot_target.target.acompletion", side_effect=outcomes):
            await fast_target.run(emitted.append, _driver(["hi"]))

        retries = _observables(emitted, RETRY_OBSERVABLE_NAME)
        assert retries[0].observable.security_domain is SYSTEM_TAG
        assert retries[0].content["error_class"] == "RateLimitError"


# ======================================================================
# Per-call timeout
# ======================================================================


class TestTimeout:
    async def test_every_call_carries_the_per_call_timeout(
        self, target: ChatbotTarget,
    ) -> None:
        # Without it litellm's own default applies (600s in the run's version,
        # 6000s in current litellm), which is why 3,410 first-run tasks sat for
        # ten minutes and then died.
        captured: list[dict] = []

        def capture(**kwargs):
            captured.append(kwargs)
            return make_litellm_response("ok")

        with m.patch("chatbot_target.target.acompletion", side_effect=capture):
            await target.run(lambda e: None, _driver(["one", "two"]))

        assert [c["timeout"] for c in captured] == [120.0, 120.0]

    async def test_the_timeout_is_the_constructors(self) -> None:
        target = ChatbotTarget(model="test-model", api_key="sk-test", request_timeout=7.5)
        captured: list[dict] = []

        def capture(**kwargs):
            captured.append(kwargs)
            return make_litellm_response("ok")

        with m.patch("chatbot_target.target.acompletion", side_effect=capture):
            await target.run(lambda e: None, _driver(["hi"]))

        assert captured[0]["timeout"] == 7.5

    async def test_no_global_litellm_state_is_touched(self, target: ChatbotTarget) -> None:
        # litellm.modify_params would repair a malformed conversation by
        # substituting the literal text "Please continue." for the attacker's
        # message and for the victim's reply -- fabricated content on both
        # sides. litellm.num_retries would retry blindly, before this target
        # can tell a content-filter block from an outage.
        import litellm

        before = (litellm.modify_params, litellm.num_retries, litellm.drop_params)
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("ok")):
            await target.run(lambda e: None, _driver(["hi"]))

        assert (litellm.modify_params, litellm.num_retries, litellm.drop_params) == before


# ======================================================================
# What the attacker can still do
# ======================================================================


class TestAttackerCapabilities:
    async def test_an_injected_reply_still_reaches_the_wire_after_a_bad_turn(
        self, fast_target: ChatbotTarget,
    ) -> None:
        # "The attacker can write the victim's response" is a modelled
        # capability of the {model} scope. The no-fabrication rule is about text
        # the TARGET invents, not text an attacker injects on the record.
        sent: list[list[dict]] = []
        calls = {"n": 0}

        def maybe_missing(**kwargs):
            sent.append(list(kwargs["messages"]))
            calls["n"] += 1
            return make_litellm_response(None if calls["n"] <= 3 else "later")

        async def send_event(event):
            if isinstance(event, ControllablePostCallEvent):
                return ControllableInjection(
                    event=event, controllable=event.controllable, value="forged",
                )
            return await base(event)

        base = _driver(["one", "two"])
        with m.patch("chatbot_target.target.acompletion", side_effect=maybe_missing):
            await fast_target.run(lambda e: None, send_event)

        assert {"role": "assistant", "content": "forged"} in sent[-1]

    async def test_the_post_call_answer_is_the_victims_own_text(
        self, fast_target: ChatbotTarget,
    ) -> None:
        posts: list[ControllablePostCallEvent] = []
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response(None)):
            await fast_target.run(lambda e: None, _driver(["hi"], posts=posts))

        assert posts[0].answer == ""


# ======================================================================
# Lifecycle
# ======================================================================


class TestLifecycle:
    async def test_reset_clears_the_outcome_record(self, target: ChatbotTarget) -> None:
        with m.patch("chatbot_target.target.acompletion",
                     return_value=make_litellm_response("ok")):
            await target.run(lambda e: None, _driver(["hi"]))
        assert _outcomes(target)

        await target.reset_ephemeral_state()
        assert _outcomes(target) == []
        assert target.query("last_turn_outcome") == ""
        assert target.query("turns_completed") == "0"

    def test_the_new_queries_are_declared(self, target: ChatbotTarget) -> None:
        names = {q.name for q in target.query_specs}
        assert {"turn_outcomes", "last_turn_outcome", "turns_completed"} <= names
