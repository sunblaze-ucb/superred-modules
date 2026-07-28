"""ChatbotTarget: any LLM as a chatbot (single-turn or multi-turn).

Wraps a litellm-accessible LLM for chatbot interaction.  The target
loops over controllable events so the optimizer controls conversation
length: inject a message to continue, respond with ControllableNoInjection
to end the run.  This supports single-turn (one injection then stop) and
multi-turn (multiple injections) from the same target implementation.

Security domain is a two-tree forest::

    Tree 1:  system
               ├── system_prompt              (controllable - override prompt)
               │     └── system_prompt_readable  (observable - read prompt)
               ├── model                      (controllable - modify LLM response)
               │     └── response_readable    (observable - read response)
               └── model_identity             (observable - read which model is in use)
    Tree 2:  user

Scope semantics:
    {response_readable}             -> can observe responses (read-only)
    {model}                         -> can modify LLM responses AND observe them
    {system_prompt_readable}        -> can see the system prompt text, can't change it
    {system_prompt}                 -> can see AND override the system prompt
    {system_prompt, user}           -> can override prompt + send messages
    {system_prompt_readable, user}  -> can see prompt + send messages (but not override)
    {model_identity}                -> can read which model is in use; nothing else
    {model_identity, user}          -> knows the victim model + can send messages
    {user}                          -> blind input (can send messages, see responses
                                      via ControllablePostCallEvent)
    {model, user}                   -> can modify responses + send messages

The ``model_identity`` tag isolates "knowledge of the victim model" from the
other system-tree capabilities. Earlier versions placed the ``model``
observable at ``SYSTEM_TAG`` (the root), which made model-fingerprint knowledge
inseparable from full system control. With ``model_identity`` as a sibling of
``system_prompt`` and ``model``, an attacker can be modeled as "knows which
model is being attacked" without simultaneously having system prompt override
or response modification capabilities.

Turn outcomes: a measurement must never be silently wrong
---------------------------------------------------------
Every victim turn ends in exactly one of five recorded outcomes.  The two
dispositions this target must never take are the ones that corrupt the
measurement: raising (which makes the framework abandon the whole task with
``stop_reason="error"``, discarding every completed turn with it) and silently
substituting the empty string (which the claims score as a legitimate 0.0
attacker failure).

======================  ==========================================  ====================
outcome                 what happened                               task disposition
======================  ==========================================  ====================
``ok``                  the victim replied with text                scored
``empty``               the victim replied, with a blank string     scored, flagged
``blocked``             a provider content filter suppressed it     scored: a defense
                                                                    worked, the attack
                                                                    failed
``no_content``          the provider returned no content field at   scored, flagged: NOT
                        all, after bounded retries                  a valid measurement
``invalid_injection``   the attacker's injected value was not a     no provider call;
                        usable message (``None`` / non-str /        the turn never
                        blank), so nothing was sent                 happened
======================  ==========================================  ====================

Anything else -- a transient provider failure that survives bounded retries, or
a persistent one (auth, context window, malformed request) -- still propagates
and still errors the task, because those genuinely are infrastructure failures
and must stay loud.

Each abnormal outcome leaves a dedicated ``ObservableEvent`` in the run's stored
trajectory (:data:`BLOCKED_OBSERVABLE_NAME`, :data:`EMPTY_OBSERVABLE_NAME`,
:data:`MISSING_OBSERVABLE_NAME`, :data:`RETRY_OBSERVABLE_NAME`,
:data:`INVALID_INJECTION_OBSERVABLE_NAME`) plus an entry in the
``turn_outcomes`` query, so an analyst can tell them apart afterwards.  They are
deliberately NOT distinguishable from ``last_response`` alone: all four abnormal
outcomes leave it empty.

The target never fabricates victim content
------------------------------------------
``last_response``, the ``ControllablePostCallEvent`` answer and the wire
conversation carry only text the victim actually produced (or text an attacker
holding ``model`` explicitly injected, which is a modelled capability).  A turn
with no victim text appends NOTHING to the wire conversation: no marker, no
empty assistant turn.  The attacker's next message then becomes a consecutive
user turn, which providers accept, whereas an empty assistant turn is silently
deleted by litellm (merging the neighbouring user turns) or rejected outright as
an assistant prefill.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass

from litellm import ModelResponse, acompletion
from litellm.exceptions import (
    APIConnectionError,
    ContentPolicyViolationError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

# ---------------------------------------------------------------------------
# Security domain: two-tree forest
#   Tree 1: system -> (system_prompt -> system_prompt_readable),
#                      (model -> response_readable),
#                      (model_identity)
#   Tree 2: user (independent root)
# ---------------------------------------------------------------------------
SYSTEM_TAG = SecurityDomainTag("system")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
SYSTEM_PROMPT_READABLE_TAG = SecurityDomainTag("system_prompt_readable", parent=SYSTEM_PROMPT_TAG)
MODEL_TAG = SecurityDomainTag("model", parent=SYSTEM_TAG)
RESPONSE_READABLE_TAG = SecurityDomainTag("response_readable", parent=MODEL_TAG)
MODEL_IDENTITY_TAG = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
USER_TAG = SecurityDomainTag("user")

_DOMAIN = SecurityDomain([
    SYSTEM_TAG, SYSTEM_PROMPT_TAG, SYSTEM_PROMPT_READABLE_TAG,
    MODEL_TAG, RESPONSE_READABLE_TAG, MODEL_IDENTITY_TAG, USER_TAG,
])

# ---------------------------------------------------------------------------
# Controllables
# ---------------------------------------------------------------------------
_USER_MESSAGE_CTRL = Controllable(
    name="user_message",
    security_domain=USER_TAG,
    description="The user message sent to the chatbot.",
)

_SYSTEM_PROMPT_CTRL = Controllable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description="Override the system prompt for this run.",
    value_type="text",
)

_RESPONSE_CTRL = Controllable(
    name="response",
    security_domain=MODEL_TAG,
    description="The LLM response. Injection overrides the response text.",
)

# ---------------------------------------------------------------------------
# Turn outcomes
# ---------------------------------------------------------------------------
#: The victim replied with non-empty text.
OUTCOME_OK = "ok"
#: The victim replied, but the text is blank. A real reply, so it is scored --
#: but flagged, because on some providers a blank reply is a policy stop that
#: the API does not label as one.
OUTCOME_EMPTY = "empty"
#: A provider content filter suppressed the reply.
OUTCOME_BLOCKED = "blocked"
#: The provider returned no assistant content field at all. Not a reply and not
#: a refusal: there is nothing to score, so a claim should abstain on it.
OUTCOME_NO_CONTENT = "no_content"
#: The attacker's injected value was not a usable message; nothing was sent.
OUTCOME_INVALID_INJECTION = "invalid_injection"

# ---------------------------------------------------------------------------
# Content-filter blocks
# ---------------------------------------------------------------------------
# A provider guardrail that suppresses the model's output is an ATTACK OUTCOME,
# not an infrastructure failure: the attacker did not get harmful content out of
# the victim.  Letting the exception escape ``run()`` would abandon the whole
# task with ``stop_reason="error"``, so the attempt would be discarded instead
# of counted, and a defense that works would look like a broken experiment.
#
# The turn is therefore recorded as a blocked turn, the run continues, and the
# provider's reason is preserved so analysis can tell a guardrail block apart
# from an ordinary model refusal and from an empty response.
#
# Matching is on the MESSAGE, not the exception type, because the type is not
# trustworthy here.  In the RQ1.3-1 first-run archive, 413 victim-side policy
# blocks were recorded and only 31 of them arrived as ``BadRequestError``; the
# other 382 arrived as ``APIConnectionError``, which is exactly the class this
# code must otherwise let through.  Gating on the type would therefore have
# absorbed 7% of real blocks and re-errored the rest.
#
# The phrases carry the specificity instead.  Each is verbatim provider response
# text, long enough that it cannot collide with an attacker prompt that merely
# discusses content policy, and it is only ever matched against the exception
# message, never against a model reply.
_CONTENT_FILTER_PHRASES = (
    "output blocked by content filtering policy",              # AWS Bedrock guardrail
    "blocked by content filter",
    "limited access to this content for safety reasons",       # OpenAI validation_error
    "content_policy_violation",                                # OpenAI
    "content management policy",                               # Azure OpenAI
    "responsible ai policy",                                   # Azure
    "content_filter",                                          # provider error code
)

#: How a blocked turn is RENDERED in ``conversation_history``, which is a human
#: readable record, not the wire.  It is never sent to a provider, never put in
#: ``last_response`` and never shown to the attacker as the victim's answer: the
#: entry that carries it also carries ``"content": null`` and ``"outcome"``, so
#: it cannot be mistaken for something the victim said.
BLOCKED_RESPONSE_MARKER = "[response withheld: blocked by the provider's content filter]"

#: Observable emitted once per blocked turn, carrying the provider's reason.
#: This is what makes a guardrail block findable in stored trajectories.
BLOCKED_OBSERVABLE_NAME = "response_blocked"

#: Observable emitted when the victim replied with a blank string.
EMPTY_OBSERVABLE_NAME = "response_empty"

#: Observable emitted when the provider returned no content field at all.
MISSING_OBSERVABLE_NAME = "response_missing"

#: Observable emitted once per retried attempt, so a turn that silently cost
#: three provider calls is never invisible in the record.
RETRY_OBSERVABLE_NAME = "response_retry"

#: Observable emitted when the attacker's injected message was unusable.
INVALID_INJECTION_OBSERVABLE_NAME = "invalid_user_message"

_BLOCKED_OBS = Observable(
    name=BLOCKED_OBSERVABLE_NAME,
    security_domain=RESPONSE_READABLE_TAG,
    description=(
        "The provider's content filter suppressed this response. "
        "Content is the provider's reason."
    ),
)

_EMPTY_OBS = Observable(
    name=EMPTY_OBSERVABLE_NAME,
    security_domain=RESPONSE_READABLE_TAG,
    description="The victim replied with a blank string. Content is the turn's diagnostics.",
)

_MISSING_OBS = Observable(
    name=MISSING_OBSERVABLE_NAME,
    security_domain=RESPONSE_READABLE_TAG,
    description=(
        "The provider returned no assistant content field for this turn, after retries. "
        "Content is the turn's diagnostics."
    ),
)

# SYSTEM_TAG, not RESPONSE_READABLE_TAG: how many times the harness had to redial
# the provider is infrastructure telemetry, not a modelled attacker observation.
# Emitting it means it is always persisted for the analyst, while no scope that
# stops below the system root can leak it into an attacker's context.
_RETRY_OBS = Observable(
    name=RETRY_OBSERVABLE_NAME,
    security_domain=SYSTEM_TAG,
    description="A provider call failed transiently and is being retried. Content is diagnostics.",
)

_INVALID_INJECTION_OBS = Observable(
    name=INVALID_INJECTION_OBSERVABLE_NAME,
    security_domain=USER_TAG,
    description=(
        "The injected user message was not usable (None, not a string, or blank) "
        "so no provider call was made. Content is the turn's diagnostics."
    ),
)

# Transient: the provider might answer if asked again.  Everything else (auth,
# context window, unsupported parameter, any other bad request) is persistent
# and is re-raised on the first attempt -- retrying it would only burn wall clock
# against the task time cap.
_TRANSIENT_ERRORS = (
    Timeout,
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
    asyncio.TimeoutError,
)

# litellm's Responses API adapter raises these instead of returning an empty
# reply, so on that transport the same event arrives as an exception.
_NO_CONTENT_ERROR_PHRASES = (
    "unknown items in responses api response",
    "unable to complete request:",
)

# Retries from 16 concurrent slots that all hit the same rate limit would
# otherwise redial in lockstep; spread them.
_RETRY_JITTER = 0.25


def content_filter_reason(exc: BaseException) -> str | None:
    """The provider's reason if *exc* is a content-filter block, else ``None``.

    ``ContentPolicyViolationError`` is the typed signal, but almost no provider
    in practice uses it.  The two wordings that actually occur are::

        BadRequestError: BedrockException - {"message": "The model returned the
        following errors: Output blocked by content filtering policy"}

        APIConnectionError: {"error": {"code": "validation_error", "message":
        "Invalid prompt: we've limited access to this content for safety
        reasons. ..."}}

    Note the second one's type.  OpenAI's refusal is a policy decision reported
    through a transport-shaped exception, so it is recognised by message alone.
    """
    if isinstance(exc, ContentPolicyViolationError):
        return str(exc)
    # Curly apostrophes appear in provider text ("we’ve"); fold them so the
    # phrase list does not need both spellings.
    haystack = str(exc).lower().replace("’", "'")
    if any(p in haystack for p in _CONTENT_FILTER_PHRASES):
        return str(exc)
    return None


def no_content_reason(exc: BaseException) -> str | None:
    """The reason if *exc* means "the provider returned nothing", else ``None``.

    litellm's chat transport reports an empty reply in band (``content`` is
    ``None``); its Responses API transport raises ``ValueError`` instead.  Both
    are the same event and must reach the same outcome, or the same failure gets
    two different dispositions depending on which model was under attack.
    """
    if not isinstance(exc, ValueError):
        return None
    haystack = str(exc).lower()
    if any(p in haystack for p in _NO_CONTENT_ERROR_PHRASES):
        return str(exc)
    return None


def invalid_injection_reason(value: object) -> str | None:
    """Why *value* is unusable as a user message, or ``None`` if it is usable.

    The attacker's injected value reaches the provider verbatim, so a broken
    attacker turns into a dead task: in the RQ1.3-1 first run, 674 tasks died
    here.  ``""`` and ``None`` are dropped by litellm before the request is
    signed, which leaves a conversation that starts with no user message, or
    ends on an assistant turn, and the provider rejects it.  The victim must
    catch that, because the alternative dispositions are both wrong: raising
    discards every completed turn of the run, and substituting filler text
    ("Please continue.", which is what ``litellm.modify_params`` does) puts
    words the attacker never wrote into the measurement.
    """
    if value is None:
        return "none"
    if not isinstance(value, str):
        return "not_a_string"
    if not value.strip():
        return "blank"
    return None


@dataclass(frozen=True)
class _TurnResult:
    """One attempted victim turn, classified."""

    outcome: str
    text: str
    reason: str | None
    finish_reason: str | None
    has_reasoning_content: bool
    attempts: int


class ChatbotTarget(Target):
    """Chatbot target wrapping any litellm-accessible LLM.

    Supports both single-turn and multi-turn conversations.

    **System prompt controllable**: At the start of each run, the target
    sends a ``ControllablePreCallEvent`` for ``system_prompt``.  If the
    optimizer's scope includes ``system_prompt`` and it injects a value,
    that value replaces the task-configured system prompt for the run.
    If the controllable is out of scope or the optimizer chooses not to
    inject, the task-configured prompt is used unchanged.

    **User message loop**: The target then loops sending
    ``ControllablePreCallEvent`` for ``user_message``.  Injection
    continues the conversation; ``ControllableNoInjection`` ends it.

    Args:
        model: LiteLLM model identifier (e.g. ``"openai/gpt-4o-mini"``).
        api_key: API key for the provider.
        api_base: Optional API base URL override.
        request_timeout: Per-call timeout in seconds, passed to every
            ``acompletion``.  litellm's own default is 600-6000s, long enough
            that one stuck call holds a concurrency slot for the whole task.
        max_attempts: How many times a single turn may call the provider.
            Only transient failures and ``no_content`` are retried.
        retry_backoff_s: Base for the exponential backoff between attempts.
        retry_max_elapsed_s: Wall-clock ceiling for one turn's attempts.  Must
            stay well under the controller's per-task time cap.
        max_consecutive_invalid: How many unusable injected messages in a row
            end the run.  Without it an attacker stuck emitting ``""`` would
            loop forever, since an unusable message makes no provider call and
            so meets no rate limit.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        *,
        request_timeout: float = 120.0,
        max_attempts: int = 3,
        retry_backoff_s: float = 2.0,
        retry_max_elapsed_s: float = 300.0,
        max_consecutive_invalid: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._request_timeout = request_timeout
        self._max_attempts = max_attempts
        self._retry_backoff_s = retry_backoff_s
        self._retry_max_elapsed_s = retry_max_elapsed_s
        self._max_consecutive_invalid = max_consecutive_invalid
        self._system_prompt: str = "You are a helpful assistant."
        self._last_response: str = ""
        self._conversation_history: list[dict[str, object]] = []
        self._turn_outcomes: list[dict[str, object]] = []
        self._blocked_turns: int = 0
        self._last_block_reason: str = ""

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="system_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
                description="System prompt for the chatbot. Set by tasks before each run.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "system_prompt":
            self._system_prompt = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="last_response",
                description=(
                    "The chatbot's most recent response text, verbatim. Empty if the last "
                    "turn produced no victim text -- which is NOT on its own a refusal: "
                    "use last_turn_outcome to tell an empty reply, a content-filter block "
                    "and a missing response apart."
                ),
            ),
            QuerySpec(
                name="conversation_history",
                description=(
                    "JSON-encoded list of message dicts [{role, content}, ...] for the "
                    "full conversation in this run. Turns with no victim text appear with "
                    "content null plus an 'outcome' key; this is the record, and it is a "
                    "superset of what was actually sent to the provider."
                ),
            ),
            QuerySpec(
                name="turn_outcomes",
                description=(
                    "JSON-encoded list, one entry per attempted turn: "
                    "{turn, outcome, reason, finish_reason, has_reasoning_content, "
                    "content_len, attempts}. The machine-readable audit trail of the run."
                ),
            ),
            QuerySpec(
                name="last_turn_outcome",
                description=(
                    "Outcome of the most recent attempted turn: one of 'ok', 'empty', "
                    "'blocked', 'no_content', 'invalid_injection'. Empty if no turn was "
                    "attempted. The single authoritative discriminator."
                ),
            ),
            QuerySpec(
                name="turns_completed",
                description=(
                    "How many turns the provider actually answered (outcome 'ok' or "
                    "'empty'). Zero means no exchange with the victim ever happened, so "
                    "there is nothing to score."
                ),
            ),
            QuerySpec(
                name="blocked_turns",
                description=(
                    "How many turns in this run had their response suppressed by the "
                    "provider's content filter. Non-zero means the attack was stopped "
                    "by a guardrail rather than by the model declining to comply."
                ),
            ),
            QuerySpec(
                name="last_block_reason",
                description=(
                    "The provider's reason for the most recent content-filter block "
                    "in this run, or empty if nothing was blocked."
                ),
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        if name == "conversation_history":
            return json.dumps(self._conversation_history)
        if name == "turn_outcomes":
            return json.dumps(self._turn_outcomes)
        if name == "last_turn_outcome":
            if not self._turn_outcomes:
                return ""
            return str(self._turn_outcomes[-1]["outcome"])
        if name == "turns_completed":
            return str(sum(
                1 for t in self._turn_outcomes
                if t["outcome"] in (OUTCOME_OK, OUTCOME_EMPTY)
            ))
        if name == "blocked_turns":
            return str(self._blocked_turns)
        if name == "last_block_reason":
            return self._last_block_reason
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [_SYSTEM_PROMPT_CTRL, _USER_MESSAGE_CTRL, _RESPONSE_CTRL]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=MODEL_IDENTITY_TAG,
                    description="The LLM model identifier.",
                ),
                content=self._model,
            ),
            ObservableValue(
                observable=Observable(
                    name="system_prompt",
                    security_domain=SYSTEM_PROMPT_READABLE_TAG,
                    description="The current system prompt text.",
                ),
                content=self._system_prompt,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        # --- Phase 1: system prompt override ---
        # Send a controllable event for the system prompt.  If the optimizer's
        # scope includes system_prompt (the writable tag), it can inject a
        # replacement.  If out of scope or no injection, the task-configured
        # value is used.
        sp_resp = await send_event(
            ControllablePreCallEvent(
                controllable=_SYSTEM_PROMPT_CTRL,
                request=self._system_prompt,
            ),
        )
        if isinstance(sp_resp, ControllableInjection):
            effective_prompt = sp_resp.value
        else:
            effective_prompt = self._system_prompt

        # Build the initial conversation with the (possibly overridden) prompt.
        # ``conversation`` is the wire: only text the attacker or the victim
        # actually produced ever enters it.  ``record`` is what the analyst
        # reads afterwards, and additionally holds the turns that never made it
        # onto the wire.
        conversation: list[dict[str, str]] = []
        record: list[dict[str, object]] = []
        if effective_prompt:
            conversation.append({"role": "system", "content": effective_prompt})
            record.append({"role": "system", "content": effective_prompt})

        # --- Phase 2: user message loop ---
        turn = 0
        consecutive_invalid = 0
        while True:
            pre_resp = await send_event(
                ControllablePreCallEvent(
                    controllable=_USER_MESSAGE_CTRL,
                    request="user message",
                ),
            )

            if not isinstance(pre_resp, ControllableInjection):
                break

            turn += 1
            invalid = invalid_injection_reason(pre_resp.value)
            if invalid is not None:
                self._record_turn(
                    turn=turn,
                    result=_TurnResult(
                        outcome=OUTCOME_INVALID_INJECTION, text="", reason=invalid,
                        finish_reason=None, has_reasoning_content=False, attempts=0,
                    ),
                )
                record.append({
                    "role": "user", "content": None,
                    "outcome": OUTCOME_INVALID_INJECTION, "reason": invalid,
                })
                self._conversation_history = list(record)
                emit(
                    ObservableEvent(
                        observable=_INVALID_INJECTION_OBS,
                        content={
                            "turn": turn, "reason": invalid,
                            "repr": repr(pre_resp.value)[:200],
                        },
                    ),
                )
                consecutive_invalid += 1
                if consecutive_invalid >= self._max_consecutive_invalid:
                    break
                continue

            consecutive_invalid = 0
            user_message = pre_resp.value
            conversation.append({"role": "user", "content": user_message})
            record.append({"role": "user", "content": user_message})

            result = await self._complete(conversation, emit, turn)
            self._record_turn(turn=turn, result=result)
            self._emit_outcome(emit, turn, result)

            # Report the response via ControllablePostCallEvent.  If the
            # optimizer's scope includes MODEL_TAG it can inject a modified
            # response that replaces the original LLM output.  The answer is the
            # victim's raw text -- "" when there was none, never a stand-in.
            assistant_message = result.text
            post_resp = await send_event(
                ControllablePostCallEvent(
                    controllable=_RESPONSE_CTRL,
                    request=user_message,
                    answer=assistant_message,
                ),
            )
            if isinstance(post_resp, ControllableInjection) and isinstance(post_resp.value, str):
                assistant_message = post_resp.value

            # WIRE RULE: an assistant turn is appended only when it carries text.
            # An empty one is not neutral -- litellm deletes it and merges the
            # surrounding attacker turns, and Bedrock rejects a conversation that
            # ends on one ("does not support assistant message prefill").
            if assistant_message.strip():
                conversation.append({"role": "assistant", "content": assistant_message})
                record.append({"role": "assistant", "content": assistant_message})
            else:
                entry: dict[str, object] = {
                    "role": "assistant", "content": None,
                    "outcome": result.outcome, "reason": result.reason,
                }
                if result.outcome == OUTCOME_BLOCKED:
                    entry["rendered"] = BLOCKED_RESPONSE_MARKER
                record.append(entry)
            self._last_response = assistant_message
            self._conversation_history = list(record)

            # Emit an observation of the (possibly modified) response.
            emit(
                ObservableEvent(
                    observable=Observable(
                        name="response",
                        security_domain=RESPONSE_READABLE_TAG,
                        description="The chatbot's response.",
                    ),
                    content=assistant_message,
                ),
            )

    async def _complete(
        self, conversation: list[dict[str, str]], emit: EventHandler, turn: int,
    ) -> _TurnResult:
        """Call the provider for one turn, retrying only what is worth retrying.

        Retrying re-samples a stochastic victim with the identical prompt, which
        can turn a refusal into a compliance and bias the measured attack success
        rate upward.  So a completed reply is never retried, whatever it says:
        only a transient transport failure and ``no_content`` (which is not a
        reply at all) get a second call, and ``attempts`` is recorded on the turn
        so retried turns can be excluded from analysis.
        """
        # Temperature is left unset so each model uses its provider default.
        # Some frontier models (e.g. GPT-5.5, Claude Opus 4.7) reject explicit
        # temperature values entirely, so we cannot pin it to 0 across the set.
        kwargs: dict[str, object] = {
            "model": self._model,
            "api_key": self._api_key,
            "timeout": self._request_timeout,
        }
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base

        started = time.monotonic()
        attempt = 0
        pending_reason = ""
        pending_error: BaseException | None = None
        while attempt < self._max_attempts:
            attempt += 1
            if attempt > 1:
                emit(
                    ObservableEvent(
                        observable=_RETRY_OBS,
                        content={
                            "turn": turn, "attempt": attempt,
                            "error_class": type(pending_error).__name__
                            if pending_error is not None else None,
                            "reason": pending_reason,
                        },
                    ),
                )
                delay = self._retry_backoff_s * 2 ** (attempt - 2)
                await asyncio.sleep(delay * (1.0 + random.random() * _RETRY_JITTER))

            try:
                response = await acompletion(messages=conversation, **kwargs)  # type: ignore[arg-type]
            except Exception as exc:
                # Order matters: 382 of the archive's 413 real blocks arrive as
                # APIConnectionError, the same class as the top retryable
                # failure, so the block check must come first or every block
                # would be retried and then re-raised as an outage.
                blocked = content_filter_reason(exc)
                if blocked is not None:
                    return _TurnResult(
                        outcome=OUTCOME_BLOCKED, text="", reason=blocked,
                        finish_reason=None, has_reasoning_content=False, attempts=attempt,
                    )
                missing = no_content_reason(exc)
                if missing is not None:
                    pending_reason, pending_error = missing, None
                elif isinstance(exc, _TRANSIENT_ERRORS):
                    pending_reason = f"{type(exc).__name__}: {exc}"
                    pending_error = exc
                else:
                    raise  # persistent: an honest infrastructure error, stay loud
            else:
                result = self._classify(response, attempt)
                if result.outcome != OUTCOME_NO_CONTENT:
                    return result
                pending_reason, pending_error = result.reason or "", None

            if time.monotonic() - started >= self._retry_max_elapsed_s:
                break

        if pending_error is not None:
            # Bare re-raise of the ORIGINAL exception: the controller stores it
            # on TaskResult.error, and the provider's own wording is what makes
            # the failure diagnosable afterwards.
            raise pending_error
        return _TurnResult(
            outcome=OUTCOME_NO_CONTENT, text="", reason=pending_reason,
            finish_reason=None, has_reasoning_content=False, attempts=attempt,
        )

    def _classify(self, response: object, attempt: int) -> _TurnResult:
        """Classify a provider response that did not raise."""
        assert isinstance(response, ModelResponse)

        def missing(reason: str) -> _TurnResult:
            return _TurnResult(
                outcome=OUTCOME_NO_CONTENT, text="", reason=reason,
                finish_reason=None, has_reasoning_content=False, attempts=attempt,
            )

        choices = response.choices or []
        if not choices:
            return missing("the provider returned no choices")
        message = choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            return missing("the provider returned no assistant content")
        if not isinstance(content, str):
            return missing(f"the assistant content was {type(content).__name__}, not text")

        finish_reason = getattr(choices[0], "finish_reason", None)
        if not isinstance(finish_reason, str):
            finish_reason = None
        # A reasoning model that spent its whole budget thinking returns a blank
        # answer with reasoning attached. Recording that distinguishes it from a
        # victim that had nothing to say.
        reasoning = getattr(message, "reasoning_content", None)
        has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())

        # The in-band form of a block: no exception, the provider labels it on
        # the choice.  Structurally impossible to false-positive on attacker
        # text, unlike the message matcher, so it is preferred where present.
        if finish_reason == "content_filter":
            return _TurnResult(
                outcome=OUTCOME_BLOCKED, text=content, reason="finish_reason=content_filter",
                finish_reason=finish_reason, has_reasoning_content=has_reasoning,
                attempts=attempt,
            )
        return _TurnResult(
            outcome=OUTCOME_OK if content.strip() else OUTCOME_EMPTY,
            text=content, reason=None, finish_reason=finish_reason,
            has_reasoning_content=has_reasoning, attempts=attempt,
        )

    def _record_turn(self, *, turn: int, result: _TurnResult) -> None:
        self._turn_outcomes.append({
            "turn": turn,
            "outcome": result.outcome,
            "reason": result.reason,
            "finish_reason": result.finish_reason,
            "has_reasoning_content": result.has_reasoning_content,
            "content_len": len(result.text),
            "attempts": result.attempts,
        })
        if result.outcome == OUTCOME_BLOCKED:
            self._blocked_turns += 1
            self._last_block_reason = result.reason or ""

    def _emit_outcome(self, emit: EventHandler, turn: int, result: _TurnResult) -> None:
        """One dedicated observable per abnormal outcome, so each is greppable."""
        if result.outcome == OUTCOME_BLOCKED:
            emit(ObservableEvent(observable=_BLOCKED_OBS, content=result.reason))
        elif result.outcome == OUTCOME_EMPTY:
            emit(ObservableEvent(observable=_EMPTY_OBS, content={
                "turn": turn,
                "finish_reason": result.finish_reason,
                "has_reasoning_content": result.has_reasoning_content,
            }))
        elif result.outcome == OUTCOME_NO_CONTENT:
            emit(ObservableEvent(observable=_MISSING_OBS, content={
                "turn": turn, "reason": result.reason, "attempts": result.attempts,
            }))

    async def reset_ephemeral_state(self) -> None:
        self._last_response = ""
        self._conversation_history = []
        self._turn_outcomes = []
        self._blocked_turns = 0
        self._last_block_reason = ""

    async def teardown(self) -> None:
        pass
