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
import re
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
    A path that does not resolve, or resolves to JSON ``null``, returns ''."""
    cur: Any = data
    for part in path.split(".") if path else []:
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
    # A resolved JSON null (e.g. OpenAI's `content: null` when the model emits only
    # a tool call) is "no text", not the literal string "null".
    if cur is None:
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
        max_attempts: total number of HTTP attempts per call *including the first*
            (not retries beyond it) — ``max_attempts=3`` calls the endpoint up to
            three times, retrying only transient ``429`` / ``5xx`` / transport
            failures. Floored at 1 (always at least one attempt). Default 3.
        retry_backoff_base: base seconds for exponential backoff between attempts
            (delay = ``base * 2 ** (attempt - 1)``). Default 0.5.
        max_retry_delay: hard cap (seconds) on any backoff sleep, including a
            server-supplied ``Retry-After`` — so an untrusted endpoint cannot stall
            the run with a huge ``Retry-After``. Default 60.
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
        max_attempts: int = 3,
        retry_backoff_base: float = 0.5,
        max_retry_delay: float = 60.0,
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
        self._max_attempts = max(1, max_attempts)  # always at least one attempt
        self._retry_backoff_base = retry_backoff_base
        self._max_retry_delay = max_retry_delay
        # One client per target instance, reused across attempts and run() calls
        # (connection pooling; the injected transport is NOT closed per attempt).
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout)

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
                    description="The target endpoint: method + host(:port) only "
                    "(never userinfo, path, query, or headers/auth).",
                ),
                content=f"{self._method} {self._host()}",
            ),
        ]

    # A DNS name / IPv4 literal: letters, digits, '.', '-'. No credential-bearing
    # characters ('@', ':', '/', '%') can appear, so a match cannot carry userinfo.
    _DNS_OR_IPV4: Final = re.compile(r"[A-Za-z0-9.\-]+")
    # An IPv6 literal as httpx returns u.host (unbracketed): hex digits and ':'.
    _IPV6: Final = re.compile(r"[0-9A-Fa-f:.]+")

    def _host(self) -> str:
        # Emit ONLY a validated bare host(:port) — never userinfo, path, query, or
        # fragment. The path/query/fragment are the credential-smear surface: httpx's
        # lenient authority parse stops at the first of '/', '?' or '#', so a
        # malformed "user:pass@host" whose password contains one of those — or a
        # percent-encoded '@' ("%40") — spills credential material into
        # host/port/path/query WITHOUT raising (e.g. "user:12/34@host" parses to
        # host="user", port=12). Six prior leak variants all rode the emitted path;
        # rather than blocklist each one, we DROP the path and positively ALLOWLIST
        # the output as a hostname/IP literal plus a numeric port. Redact otherwise.
        # The ENTIRE body is guarded: httpx.URL() can reject a URL outright, and
        # accessing u.host can itself raise (e.g. IDNA InvalidCodepoint on a
        # malformed "xn--" punycode host that CONSTRUCTS but fails to decode). This
        # runs from get_observables(), so it must be total — never crash the sweep.
        try:
            u = httpx.URL(self._url)
            host = u.host  # httpx has already stripped any RECOGNIZED userinfo
            if not host:
                return "(unparsable url)"
            # A URL carries credentials only through an '@' userinfo delimiter, so an
            # unaccounted '@' is the tell that credentials spilled past the authority
            # into host/port. Count every '@' that can materialize as a literal '@'
            # in a client-decoded URL: a raw '@' OR a percent-encoded "%40" (httpx
            # decodes %40 -> '@'). httpx recognizes exactly one '@' as userinfo iff
            # u.username/u.password is set; any other '@' (raw or encoded) means a
            # smeared credential separator — redact rather than emit the misparse.
            at_total = self._url.count("@") + self._url.count("%40")
            expected_at = 1 if (u.username or u.password) else 0
            if at_total != expected_at:
                return "(unparsable url)"
            # Positively allowlist the host: a DNS name / IPv4, else an IPv6 literal
            # (bracketed so the ':port' is unambiguous). This rejects any residual
            # smear a count check could miss (stray delimiters, unicode). Port, if
            # present, is already an int parsed by httpx.
            if self._DNS_OR_IPV4.fullmatch(host):
                hostpart = host
            elif self._IPV6.fullmatch(host):
                hostpart = f"[{host}]"
            else:
                return "(unparsable url)"
            return hostpart if u.port is None else f"{hostpart}:{u.port}"
        except Exception:  # noqa: BLE001 - malformed URL, or a raising property access
            return "(unparsable url)"

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
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = await self._client.request(
                    self._method, self._url, json=body, headers=self._headers
                )
                self._http_status = resp.status_code
                self._raw_response = resp.text
                if resp.status_code == 429 and attempt < self._max_attempts:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                if resp.status_code >= 500 and attempt < self._max_attempts:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                # >= 300 (not just >= 400): redirects are NOT followed
                # (follow_redirects defaults to False), so a 3xx (HTTP->HTTPS,
                # trailing-slash normalization, an auth/SSO redirect) is not the
                # app's reply — treat it as an error so the claim abstains rather
                # than scoring an empty/redirect body as the response.
                if resp.status_code >= 300:
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
                # Clear status/body so they never pair a prior attempt's response
                # with this attempt's error. Record only the exception TYPE, never
                # str(exc): an httpx error message can echo the URL (incl. userinfo
                # credentials), and `error` is a caller-visible query.
                self._http_status = None
                self._raw_response = ""
                self._error = type(exc).__name__
                if attempt < self._max_attempts:
                    await self._backoff(attempt, None)
                    continue
                return
            except Exception as exc:  # noqa: BLE001 - e.g. httpx.InvalidURL (NOT an HTTPError)
                # A malformed URL (bad port, control char, unsubstituted template)
                # raises synchronously from client.request and is not an
                # httpx.HTTPError — record it and stop (retrying can't fix the URL)
                # so it never propagates out of run() and crashes the sweep. Type
                # only: an InvalidURL message embeds the URL (userinfo credentials).
                self._http_status = None
                self._raw_response = ""
                self._error = type(exc).__name__
                return

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = self._retry_backoff_base * (2.0 ** (attempt - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        # Clamp to max_retry_delay: this target points at an arbitrary/untrusted
        # endpoint, so a hostile server-controlled Retry-After (e.g. 86400) must not
        # be able to stall the run — the request timeout does not bound this sleep.
        delay = min(delay, self._max_retry_delay)
        if delay > 0:
            await asyncio.sleep(delay)

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        # Close the client only if we own its transport (transport=None -> httpx
        # built a private one). An injected transport is the caller's to manage
        # (and may be shared across instances / concurrency), so we never close it.
        if self._transport is None:
            await self._client.aclose()


__all__ = ["HttpEndpointTarget", "PROMPT_PLACEHOLDER", "SYSTEM_TAG", "USER_INPUT_TAG"]
