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

Content-filter blocks are a defense, not an outage
--------------------------------------------------
When the provider's guardrail suppresses the victim's reply, the attacker
failed to extract harmful content.  That is a measurement, so the target does
not raise: it records the turn as blocked and lets the run finish and be scored.
Raising would make the framework abandon the whole task with
``stop_reason="error"``, discarding the attempt, which systematically
under-counts exactly the cases where a defense worked.

Every blocked turn leaves three traces, so the outcome is distinguishable
afterwards from an ordinary refusal and from an empty reply:

* an ``ObservableEvent`` named ``response_blocked``
  (:data:`BLOCKED_OBSERVABLE_NAME`) whose content is the provider's reason, in
  the run's stored trajectory;
* :data:`BLOCKED_RESPONSE_MARKER` in place of the reply, in the conversation,
  in ``last_response`` and in the ``ControllablePostCallEvent`` answer;
* the ``blocked_turns`` and ``last_block_reason`` queries, for tasks that want
  to score or label the outcome.

Only genuine content-filter blocks are absorbed.  Every other provider error
still propagates and still errors the task, because a rate limit or a dropped
connection is an infrastructure failure and must not be recorded as a defended
attack.
"""

from __future__ import annotations

from litellm import ModelResponse, acompletion
from litellm.exceptions import ContentPolicyViolationError

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

#: Stands in for the suppressed reply in the conversation, in ``last_response``
#: and in the ``ControllablePostCallEvent`` answer, so all three agree.  A
#: distinct marker rather than an empty string: an empty assistant turn is
#: rejected outright by some providers, and it would be indistinguishable from
#: a model that legitimately returned nothing.
BLOCKED_RESPONSE_MARKER = "[response withheld: blocked by the provider's content filter]"

#: Observable emitted once per blocked turn, carrying the provider's reason.
#: This is what makes a guardrail block findable in stored trajectories.
BLOCKED_OBSERVABLE_NAME = "response_blocked"


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
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._system_prompt: str = "You are a helpful assistant."
        self._last_response: str = ""
        self._conversation_history: list[dict[str, str]] = []
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
                description="The chatbot's most recent response text.",
            ),
            QuerySpec(
                name="conversation_history",
                description=(
                    "JSON-encoded list of message dicts [{role, content}, ...] "
                    "for the full conversation in this run."
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
            import json
            return json.dumps(self._conversation_history)
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
        conversation: list[dict[str, str]] = []
        if effective_prompt:
            conversation.append({"role": "system", "content": effective_prompt})

        # --- Phase 2: user message loop ---
        while True:
            pre_resp = await send_event(
                ControllablePreCallEvent(
                    controllable=_USER_MESSAGE_CTRL,
                    request="user message",
                ),
            )

            if not isinstance(pre_resp, ControllableInjection):
                break

            user_message = pre_resp.value
            conversation.append({"role": "user", "content": user_message})

            # Call the LLM. Temperature is left unset so each model uses its
            # provider default. Some frontier models (e.g. GPT-5.5, Claude
            # Opus 4.7) reject explicit temperature values entirely, so we
            # cannot pin it to 0 across the model set.
            kwargs: dict[str, str | None] = {
                "model": self._model,
                "api_key": self._api_key,
            }
            if self._api_base is not None:
                kwargs["api_base"] = self._api_base

            try:
                response = await acompletion(messages=conversation, **kwargs)  # type: ignore[arg-type]
            except Exception as exc:
                reason = content_filter_reason(exc)
                if reason is None:
                    raise  # a real failure: let the framework error the task
                # The victim's guardrail suppressed the reply. Record it as a
                # blocked turn and keep going, so the attacker keeps control of
                # the conversation and the task is still scored.
                self._blocked_turns += 1
                self._last_block_reason = reason
                assistant_message = BLOCKED_RESPONSE_MARKER
                emit(
                    ObservableEvent(
                        observable=Observable(
                            name=BLOCKED_OBSERVABLE_NAME,
                            security_domain=RESPONSE_READABLE_TAG,
                            description=(
                                "The provider's content filter suppressed this response. "
                                "Content is the provider's reason."
                            ),
                        ),
                        content=reason,
                    ),
                )
            else:
                assert isinstance(response, ModelResponse)
                assistant_message = response.choices[0].message.content or ""

            # Report the response via ControllablePostCallEvent.  If the
            # optimizer's scope includes MODEL_TAG it can inject a modified
            # response that replaces the original LLM output.
            post_resp = await send_event(
                ControllablePostCallEvent(
                    controllable=_RESPONSE_CTRL,
                    request=user_message,
                    answer=assistant_message,
                ),
            )
            if isinstance(post_resp, ControllableInjection):
                assistant_message = post_resp.value

            conversation.append({"role": "assistant", "content": assistant_message})
            self._last_response = assistant_message
            self._conversation_history = list(conversation)

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

    async def reset_ephemeral_state(self) -> None:
        self._last_response = ""
        self._conversation_history = []
        self._blocked_turns = 0
        self._last_block_reason = ""

    async def teardown(self) -> None:
        pass
