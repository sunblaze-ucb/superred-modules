"""HttpEndpointTarget: an arbitrary HTTP LLM/chat endpoint as a superred Target.

Points superred at *your own deployed* LLM application (or any HTTP API): the
attacker controls the prompt, the target renders it into a configurable JSON
request body, POSTs it to your endpoint, and extracts the model's reply via a
configurable JSON path. This is the "bring your own endpoint" target archetype —
distinct from provider-SDK targets (e.g. ``minimal_llm_chat``) — so any chatbot
claim / optimizer can drive a real HTTP service under test.

The prompt is placed JSON-safely: the configured ``body_template`` is a parsed
JSON structure, and every value equal to the sentinel ``{{prompt}}`` is replaced
by the prompt *as a value* (not string-substituted), so a prompt containing
quotes or braces can never corrupt the request body.

Auth headers (an API key / bearer token) are held privately and never emitted as
an observable, returned from a query, or written into any rationale.

Live runs hit your real endpoint; offline tests inject an ``httpx.MockTransport``.
Only red-team an endpoint you are authorized to test.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Final

import httpx
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

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG])

# Sentinel value marking where the prompt goes in the JSON body template. Any value
# in the template exactly equal to this is replaced by the prompt (as a JSON value).
PROMPT_PLACEHOLDER: Final = "{{prompt}}"


def _render(template: Any, prompt: str) -> Any:
    """Deep-copy ``template``, replacing every ``PROMPT_PLACEHOLDER`` value with
    ``prompt`` as a value (JSON-safe — no string interpolation)."""
    if isinstance(template, dict):
        return {k: _render(v, prompt) for k, v in template.items()}
    if isinstance(template, list):
        return [_render(v, prompt) for v in template]
    if template == PROMPT_PLACEHOLDER:
        return prompt
    return template


def _extract(data: Any, path: str) -> str:
    """Extract the reply text from a JSON response via a dot path (e.g.
    ``choices.0.message.content``). Empty path returns the whole body as text.
    A path that does not resolve returns ''."""
    if not path:
        return data if isinstance(data, str) else json.dumps(data)
    cur: Any = data
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return ""
        elif isinstance(cur, dict):
            if part not in cur:
                return ""
            cur = cur[part]
        else:
            return ""
    return cur if isinstance(cur, str) else json.dumps(cur)


class HttpEndpointTarget(Target):
    """An arbitrary HTTP LLM endpoint as a superred target.

    Args:
        url: the endpoint URL to POST to.
        method: HTTP method (default ``POST``).
        headers: request headers, may include an auth key/token — held privately,
            never emitted.
        body_template: the JSON request body as a Python structure; every value
            equal to ``{{prompt}}`` is replaced by the prompt. Defaults to
            ``{"prompt": "{{prompt}}"}``.
        response_path: dot path to the reply text in the JSON response (e.g.
            ``choices.0.message.content``); empty returns the whole body as text.
        timeout: per-request timeout (seconds).
        transport: optional ``httpx`` transport for offline tests
            (``httpx.MockTransport``); ``None`` uses the real network.
    """

    def __init__(
        self,
        *,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        body_template: Any = None,
        response_path: str = "",
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
    ) -> None:
        self._url = url
        self._method = method.upper()
        self._headers = dict(headers or {})  # PRIVATE: never emitted/queried/logged
        self._body_template: Any = body_template if body_template is not None else {
            "prompt": PROMPT_PLACEHOLDER
        }
        self._response_path = response_path
        self._timeout = timeout
        self._transport = transport
        self._max_retries = max(1, max_retries)
        self._retry_backoff_base = retry_backoff_base

        self._benign_prompt = "Hello, can you help me?"
        self._reset_state()

    def _reset_state(self) -> None:
        self._last_response: str = ""
        self._raw_response: str = ""
        self._http_status: int | None = None
        self._error: str = ""
        self._sent_prompt: str = ""

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="benign_prompt",
                security_domain=SYSTEM_TAG,
                description="The non-attack prompt used when the prompt is not injected.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "benign_prompt":
            self._benign_prompt = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="last_response",
                description="The reply text extracted from the endpoint.",
            ),
            QuerySpec(name="raw_response", description="The raw response body."),
            QuerySpec(name="http_status", description="HTTP status code of the call, or ''."),
            QuerySpec(name="error", description="Error detail if the call failed, else ''."),
            QuerySpec(
                name="sent_prompt",
                description="The effective prompt sent (after injection).",
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        if name == "raw_response":
            return self._raw_response
        if name == "http_status":
            return "" if self._http_status is None else str(self._http_status)
        if name == "error":
            return self._error
        if name == "sent_prompt":
            return self._sent_prompt
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name="prompt",
                security_domain=USER_INPUT_TAG,
                description="The prompt sent to the HTTP endpoint.",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="endpoint",
                    security_domain=SYSTEM_TAG,
                    description="The target endpoint (method + host only; never headers/auth).",
                ),
                content=f"{self._method} {self._host()}",
            ),
        ]

    def _host(self) -> str:
        # scheme+host+path only — drop any query string (which could carry a token).
        no_scheme = self._url.split("://", 1)[-1]
        return no_scheme.split("?", 1)[0]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()

        ctrl = self.get_controllables()[0]
        resp = await send_event(
            ControllablePreCallEvent(controllable=ctrl, request="Enter prompt:")
        )
        prompt = resp.value if isinstance(resp, ControllableInjection) else self._benign_prompt
        self._sent_prompt = prompt

        emit(
            ObservableEvent(
                observable=Observable(
                    name="endpoint_input",
                    security_domain=USER_INPUT_TAG,
                    description="The prompt sent to the endpoint.",
                ),
                content=prompt,
            )
        )

        await self._call(prompt)

        emit(
            ObservableEvent(
                observable=Observable(
                    name="endpoint_response",
                    security_domain=SYSTEM_TAG,
                    description="The endpoint's extracted reply.",
                ),
                content=self._last_response[:500],
            )
        )

    async def _call(self, prompt: str) -> None:
        body = _render(self._body_template, prompt)
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    transport=self._transport, timeout=self._timeout
                ) as client:
                    resp = await client.request(
                        self._method, self._url, json=body, headers=self._headers
                    )
                self._http_status = resp.status_code
                self._raw_response = resp.text
                if resp.status_code == 429 and attempt < self._max_retries:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                if resp.status_code >= 500 and attempt < self._max_retries:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                if resp.status_code >= 400:
                    self._error = f"HTTP {resp.status_code}"
                    return
                self._error = ""  # a prior transient attempt, if any, recovered
                try:
                    data = resp.json()
                except ValueError:
                    # a 2xx with a non-JSON body: use the raw text as the reply.
                    self._last_response = resp.text
                    return
                self._last_response = _extract(data, self._response_path)
                return
            except httpx.HTTPError as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_retries:
                    await self._backoff(attempt, None)
                    continue
                return

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = self._retry_backoff_base * (2.0 ** (attempt - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        if delay > 0:
            await asyncio.sleep(delay)

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = ["HttpEndpointTarget", "PROMPT_PLACEHOLDER", "SYSTEM_TAG", "USER_INPUT_TAG"]
