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
               └── model                      (controllable - modify LLM response)
                     └── response_readable    (observable - read response)
    Tree 2:  user

Scope semantics:
    {response_readable}             -> can observe responses (read-only)
    {model}                         -> can modify LLM responses AND observe them
    {system_prompt_readable}        -> can see the system prompt text, can't change it
    {system_prompt}                 -> can see AND override the system prompt
    {system_prompt, user}           -> can override prompt + send messages
    {system_prompt_readable, user}  -> can see prompt + send messages (but not override)
    {user}                          -> blind input (can send messages, see responses
                                      via ControllablePostCallEvent)
    {model, user}                   -> can modify responses + send messages
"""

from __future__ import annotations

from litellm import ModelResponse, acompletion

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
#                      (model -> response_readable)
#   Tree 2: user (independent root)
# ---------------------------------------------------------------------------
SYSTEM_TAG = SecurityDomainTag("system")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
SYSTEM_PROMPT_READABLE_TAG = SecurityDomainTag("system_prompt_readable", parent=SYSTEM_PROMPT_TAG)
MODEL_TAG = SecurityDomainTag("model", parent=SYSTEM_TAG)
RESPONSE_READABLE_TAG = SecurityDomainTag("response_readable", parent=MODEL_TAG)
USER_TAG = SecurityDomainTag("user")

_DOMAIN = SecurityDomain([
    SYSTEM_TAG, SYSTEM_PROMPT_TAG, SYSTEM_PROMPT_READABLE_TAG,
    MODEL_TAG, RESPONSE_READABLE_TAG, USER_TAG,
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
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        if name == "conversation_history":
            import json
            return json.dumps(self._conversation_history)
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
                    security_domain=SYSTEM_TAG,
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

            # Call the LLM. Temperature is fixed at 0 for reproducibility:
            # red-team experiments need deterministic target behavior so that
            # repeated runs with the same inputs produce comparable outputs.
            kwargs: dict[str, str | int | None] = {
                "model": self._model,
                "api_key": self._api_key,
                "temperature": 0,
            }
            if self._api_base is not None:
                kwargs["api_base"] = self._api_base

            response = await acompletion(messages=conversation, **kwargs)  # type: ignore[arg-type]
            assert isinstance(response, ModelResponse)
            assistant_message: str = response.choices[0].message.content or ""

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

    async def cleanup(self) -> None:
        self._last_response = ""
        self._conversation_history = []

    async def teardown(self) -> None:
        pass
