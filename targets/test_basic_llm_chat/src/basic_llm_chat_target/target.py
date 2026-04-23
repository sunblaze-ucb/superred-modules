"""BasicLLMChatTarget: a single-turn LLM chat system.

The simplest possible real target — sends a user message to an LLM
and returns the response. One controllable (the user input), one
config slot (system prompt), one query (last response).

Uses litellm for LLM access, so any model litellm supports works.
"""

from __future__ import annotations

from litellm import ModelResponse, acompletion

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

# Security domain: system root with user_input child
SYSTEM_TAG = SecurityDomainTag("system")
USER_INPUT_TAG = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
_DOMAIN = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG])


class BasicLLMChatTarget(Target):
    """Single-turn LLM chat target.

    The target accepts a system prompt (via config) and a user message
    (via controllable injection). It calls the LLM and stores the
    response for post-run evaluation.

    Args:
        model: LiteLLM model identifier (e.g. ``"gpt-4o-mini"``).
        api_base: LiteLLM-compatible API base URL.
        api_key: API key for the LiteLLM proxy.
    """

    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._system_prompt: str = "You are a helpful assistant."
        self._last_response: str = ""

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="system_prompt",
                security_domain=SYSTEM_TAG,
                description="System prompt for the LLM.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "system_prompt":
            self._system_prompt = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [QuerySpec(name="last_response", description="The LLM's last response.")]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name="user_input",
                security_domain=USER_INPUT_TAG,
                description="The user message sent to the LLM.",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        obs = Observable(
            name="model",
            security_domain=SYSTEM_TAG,
            description="The LLM model identifier.",
        )
        return [ObservableValue(observable=obs, content=self._model)]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        # 1. Get user input from optimizer
        ctrl = Controllable(
            name="user_input", security_domain=USER_INPUT_TAG,
        )
        resp = await send_event(
            ControllablePreCallEvent(controllable=ctrl, request="Enter user message:"),
        )
        user_message = resp.value if isinstance(resp, ControllableInjection) else "Hello"

        emit(ObservableEvent(
            observable=Observable(
                name="model_request", security_domain=USER_INPUT_TAG,
                description="User message sent to the LLM.",
            ),
            content=user_message,
        ))

        # 2. Call LLM
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await acompletion(
            model=self._model,
            messages=messages,
            api_base=self._api_base,
            api_key=self._api_key,
        )
        assert isinstance(response, ModelResponse)
        llm_response = response.choices[0].message.content or ""
        self._last_response = llm_response

        emit(ObservableEvent(
            observable=Observable(
                name="model_response", security_domain=SYSTEM_TAG,
                description="LLM response.",
            ),
            content=llm_response,
        ))

    async def cleanup(self) -> None:
        self._last_response = ""

    async def teardown(self) -> None:
        pass
