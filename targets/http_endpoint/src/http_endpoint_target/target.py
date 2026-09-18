"""HttpEndpointTarget: an arbitrary HTTP LLM/chat endpoint as a superred Target.

Points superred at *your own deployed* LLM application (or any HTTP API): the
attacker controls the values placed into a configurable JSON request body, the
target POSTs it to your endpoint, and extracts the model's reply via a
configurable JSON path. This is the "bring your own endpoint" target archetype —
distinct from provider-SDK targets (e.g. ``minimal_llm_chat``) — so any chatbot
claim / optimizer can drive a real HTTP service under test.

Values are placed JSON-safely: the configured ``body_template`` is a parsed JSON
structure, and every value equal to a placeholder such as ``{{prompt}}`` is
replaced by that slot's value *as a value* (not string-substituted), so a value
containing quotes or braces can never corrupt the request body.

Security domains are declared by the caller, because only the caller knows what
each part of the request carries. Every ``{{name}}`` placeholder is a
:class:`Slot` naming the security domain its value arrives through (the user's
message, a retrieved document, a system prompt, ...), and each slot becomes one
controllable at that domain; ``slots`` is therefore required whenever a
``body_template`` is given. The extracted reply and the endpoint's identity are
observables at ``response_domain`` and ``endpoint_domain``. The defaults model a
plain chat endpoint: one ``{{prompt}}`` slot on an independent ``user_input``
root, with the reply and the endpoint identity as leaves of a ``system`` root
(see ``README.md``).

Auth headers (an API key / bearer token) are held privately and never emitted as
an observable, returned from a query, or written into any rationale.

Live runs hit your real endpoint; offline tests inject an ``httpx.MockTransport``.
Only red-team an endpoint you are authorized to test.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
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

# Default forest, per the superred security-domain guide: the user's channel is
# independent of the deployed system, so it is its own root (principle 5), and
# the reply and the endpoint's identity are separate leaves of the system root,
# so "reads the replies" and "knows the host" can each be granted (normally via
# the Controller's ``read_only``) without the other and without any write
# capability (principles 3 and 4).
SYSTEM_TAG: Final = SecurityDomainTag("system")
RESPONSE_TAG: Final = SecurityDomainTag("response", parent=SYSTEM_TAG)
ENDPOINT_TAG: Final = SecurityDomainTag("endpoint", parent=SYSTEM_TAG)
USER_INPUT_TAG: Final = SecurityDomainTag("user_input")

_DEFAULT_TAGS: Final = (SYSTEM_TAG, RESPONSE_TAG, ENDPOINT_TAG, USER_INPUT_TAG)

# Placeholder of the default body template and its default slot.
PROMPT_PLACEHOLDER: Final = "{{prompt}}"
_DEFAULT_BENIGN_PROMPT: Final = "Hello, can you help me?"
_DEFAULT_PROMPT_DESCRIPTION: Final = "The prompt sent to the HTTP endpoint."

# A placeholder is a complete JSON string value "{{name}}", name an identifier.
_PLACEHOLDER: Final = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
# Anything that merely looks like one, so a near miss can be rejected instead of
# being sent to the endpoint verbatim.
_PLACEHOLDER_ISH: Final = re.compile(r"\{\{.*?\}\}")
_SLOT_NAME: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Observable names this target emits itself. A read-only slot is re-presented to
# the optimizer as an observable named after the slot, so a slot may not shadow
# one of these (or another slot's ``sent_`` observable).
_RESERVED_SLOT_NAMES: Final = frozenset({"endpoint", "endpoint_response"})
_SENT_PREFIX: Final = "sent_"


def _sentinel(name: str) -> str:
    return "{{" + name + "}}"


@dataclass(frozen=True)
class Slot:
    """One ``{{name}}`` placeholder in the request body.

    Attributes:
        domain: the security domain the value arrives through. The target
            exposes the slot as a controllable at exactly this tag object, so
            build each tag once and pass the same objects to the Controller's
            ``scope`` / ``read_only`` (tags match by identity).
        default: the value sent when the optimizer does not inject.
        description: what this part of the request carries, as shown to the
            optimizer. Defaults to a line naming the placeholder.
    """

    domain: SecurityDomainTag
    default: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain, SecurityDomainTag):
            raise TypeError(
                f"Slot.domain must be a SecurityDomainTag, not {type(self.domain).__name__}"
            )
        if not isinstance(self.default, str):
            raise TypeError(f"Slot.default must be a str, not {type(self.default).__name__}")
        if not isinstance(self.description, str):
            raise TypeError(
                f"Slot.description must be a str, not {type(self.description).__name__}"
            )


class ResponseTooLargeError(Exception):
    """Raised internally to abort reading a response body that exceeds the byte
    cap (an untrusted endpoint must not be able to exhaust memory)."""


def _check_tag_identity(tag: SecurityDomainTag, what: str) -> None:
    """Reject a look-alike of one of this module's exported tags.

    superred matches tags by IDENTITY (``SecurityDomainTag.includes`` compares
    with ``is``) while ``==`` compares name and parent. So a freshly built
    ``SecurityDomainTag("user_input")`` prints, compares and set-tests as equal
    to :data:`USER_INPUT_TAG`, yet a Controller scope built from the exported tag
    would cover nothing this target exposes: the run would inject nowhere and
    report no error at all. Fail at construction instead."""
    cur: SecurityDomainTag | None = tag
    while cur is not None:
        for default in _DEFAULT_TAGS:
            if cur == default and cur is not default:
                raise ValueError(
                    f"{what} is a new SecurityDomainTag equal to this module's "
                    f"{default.name!r} tag; import and reuse the exported object, "
                    "because a Controller scope matches tags by identity and would "
                    "silently cover nothing here"
                )
        cur = cur.parent


def _render(template: Any, values: Mapping[str, str]) -> Any:
    """Deep-copy ``template``, replacing every complete-value ``{{name}}``
    placeholder with ``values[name]`` as a value (JSON-safe — no string
    interpolation)."""
    if isinstance(template, dict):
        return {k: _render(v, values) for k, v in template.items()}
    if isinstance(template, (list, tuple)):
        return [_render(v, values) for v in template]
    if isinstance(template, str) and (m := _PLACEHOLDER.fullmatch(template)):
        return values[m.group(1)]
    return template


def _scan_template(template: Any) -> set[str]:
    """The names of the placeholders that appear as complete values in the template.

    Raises ``ValueError`` for anything that only looks like a placeholder:
    embedded in a longer string (``"ask: {{prompt}}"``), spaced
    (``"{{ prompt }}"``), a non-identifier name, or sitting in a dict KEY. Only a
    complete string value is substituted — string interpolation would let a value
    with quotes or braces corrupt the JSON body — so every other form would be
    sent to the endpoint verbatim, which is a silent misconfiguration rather than
    an attack."""
    found: set[str] = set()
    if isinstance(template, dict):
        for key, value in template.items():
            if isinstance(key, str) and _PLACEHOLDER_ISH.search(key):
                raise ValueError(
                    f"body_template key {key!r} looks like a placeholder, but only a "
                    "complete string VALUE is ever substituted, never a key"
                )
            found |= _scan_template(value)
    elif isinstance(template, (list, tuple)):
        for value in template:
            found |= _scan_template(value)
    elif isinstance(template, str):
        if m := _PLACEHOLDER.fullmatch(template):
            found.add(m.group(1))
        elif _PLACEHOLDER_ISH.search(template):
            raise ValueError(
                f"body_template value {template!r} contains something that looks "
                "like a placeholder but is not exactly one, so it would be sent "
                "verbatim. A placeholder is a whole string value whose name is an "
                "identifier, such as " + repr(_sentinel("prompt")) + ": no text "
                "around it, and no spaces inside the braces."
            )
    return found


def _closure(tags: list[SecurityDomainTag]) -> list[SecurityDomainTag]:
    """``tags`` plus every ancestor, each tag OBJECT once (roots first).

    Deduplicated by identity, not name: two distinct tag objects that share a
    name both land in the list, so :class:`SecurityDomain` rejects the duplicate
    instead of a scope built from one object silently missing the other."""
    ordered: dict[int, SecurityDomainTag] = {}
    for tag in tags:
        chain: list[SecurityDomainTag] = []
        cur: SecurityDomainTag | None = tag
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        for t in reversed(chain):
            ordered.setdefault(id(t), t)
    return list(ordered.values())


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
            never emitted. Note ``Accept-Encoding`` is always forced to ``identity``
            (compression is refused for bomb-safety — see ``max_response_bytes``), so
            any ``Accept-Encoding`` given here is ignored.
        body_template: the JSON request body as a Python structure; every value
            equal to a declared slot's ``{{name}}`` is replaced by that slot's value.
            Defaults to ``{"prompt": "{{prompt}}"}``.
        slots: ``{name: Slot(domain, default)}`` for every placeholder in
            ``body_template``, in the order their controllables are offered. Each
            becomes a controllable named ``name`` at ``domain``. **Required
            whenever ``body_template`` is given**, so nothing is silently labelled
            as user input; with the default template it defaults to one ``prompt``
            slot at :data:`USER_INPUT_TAG`. Construction raises ``ValueError`` if a
            declared slot's placeholder is missing from the template, or the
            template holds a placeholder no slot declares.
        response_path: dot path to the reply text in the JSON response (e.g.
            ``choices.0.message.content``); empty returns the whole body as text.
        response_domain: security domain of the ``endpoint_response`` observable
            (the extracted reply). Default :data:`RESPONSE_TAG`.
        endpoint_domain: security domain of the static ``endpoint`` observable
            (method + host). Default :data:`ENDPOINT_TAG`.
        timeout: httpx per-operation timeout (connect/read/write/pool, seconds).
            This bounds the wait *between* chunks, NOT total response time — see
            ``max_response_time`` for the total bound.
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
        max_response_time: total wall-clock cap (seconds) per attempt on the whole
            request + body read. Unlike ``timeout`` (per-op), this bounds a hostile
            server that trickles bytes slowly to keep the connection open; on
            exceed the attempt aborts (``TimeoutError``, not retried). Default 60.
        max_response_bytes: cap on the response bytes read into memory. The body is
            streamed and the read aborts past this cap (``ResponseTooLargeError``,
            not retried); responses are requested uncompressed
            (``Accept-Encoding: identity``) and a compressed response is refused
            rather than decompressed, so an untrusted endpoint cannot exhaust memory
            with a huge or compression-bomb response (see :meth:`_request_capped`).
            Default 1_000_000 (1 MB).
    """

    def __init__(
        self,
        *,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        body_template: Any = None,
        slots: Mapping[str, Slot] | None = None,
        response_path: str = "",
        response_domain: SecurityDomainTag = RESPONSE_TAG,
        endpoint_domain: SecurityDomainTag = ENDPOINT_TAG,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_attempts: int = 3,
        retry_backoff_base: float = 0.5,
        max_retry_delay: float = 60.0,
        max_response_time: float = 60.0,
        max_response_bytes: int = 1_000_000,
    ) -> None:
        self._url = url
        self._method = method.upper()
        self._headers = dict(headers or {})  # PRIVATE: never emitted/queried/logged
        default_template = body_template is None
        self._body_template: Any = (
            {"prompt": PROMPT_PLACEHOLDER} if default_template else body_template
        )
        if slots is None:
            if not default_template:
                raise ValueError(
                    "slots is required with a custom body_template: declare every "
                    "{{name}} placeholder as Slot(domain, default) so every value "
                    "the endpoint receives has a security domain behind it. Only the "
                    "default template's {{prompt}} is assumed to be user input."
                )
            slots = {
                "prompt": Slot(
                    USER_INPUT_TAG, _DEFAULT_BENIGN_PROMPT, _DEFAULT_PROMPT_DESCRIPTION
                )
            }
        self._slots: dict[str, Slot] = dict(slots)
        if not self._slots:
            raise ValueError("slots must declare at least one placeholder to inject into")
        for name, slot in self._slots.items():
            if not isinstance(name, str) or not _SLOT_NAME.fullmatch(name):
                raise ValueError(
                    f"slot name {name!r} must be an identifier (letters, digits, '_')"
                )
            if name in _RESERVED_SLOT_NAMES or name.startswith(_SENT_PREFIX):
                raise ValueError(
                    f"slot name {name!r} is reserved: this target emits observables "
                    f"named {sorted(_RESERVED_SLOT_NAMES)} and {_SENT_PREFIX}<slot>, and a "
                    "read-only slot is re-presented to the optimizer as an observable "
                    "named after the slot, so the names would collide"
                )
            if not isinstance(slot, Slot):
                raise TypeError(f"slots[{name!r}] must be a Slot, not {type(slot).__name__}")
        for arg, tag in (
            ("response_domain", response_domain),
            ("endpoint_domain", endpoint_domain),
        ):
            if not isinstance(tag, SecurityDomainTag):
                raise TypeError(f"{arg} must be a SecurityDomainTag, not {type(tag).__name__}")
            _check_tag_identity(tag, arg)
        for name, slot in self._slots.items():
            _check_tag_identity(slot.domain, f"slots[{name!r}].domain")
        # Fail fast on a template/slot mismatch. _render only substitutes a value
        # EXACTLY equal to a declared placeholder, so a slot whose placeholder is
        # missing / embedded in a string / mistyped would silently send an
        # un-injected body on every call — a systematic false-negative with no
        # observable signal — and a placeholder no slot declares would be sent
        # literally, with no security domain behind it. Reject both at construction
        # rather than run a hollow attack.
        found = _scan_template(self._body_template)
        missing = [name for name in self._slots if name not in found]
        if missing:
            raise ValueError(
                "body_template must contain every declared slot's placeholder as a "
                "complete value (not embedded in a larger string) so its value is "
                f"actually injected; missing: {', '.join(map(_sentinel, missing))}. "
                "The same un-injected body would be sent on every call and the "
                "endpoint never attacked."
            )
        undeclared = sorted(found - set(self._slots))
        if undeclared:
            raise ValueError(
                f"body_template has placeholder(s) {', '.join(map(_sentinel, undeclared))} "
                "with no slot; declare each in `slots` with the security domain its "
                "value arrives through (otherwise it is sent literally)."
            )
        self._response_domain = response_domain
        self._endpoint_domain = endpoint_domain
        # The forest is exactly the declared surfaces plus their ancestors, so a
        # custom declaration never drags in the default tags.
        self._domain = SecurityDomain(
            _closure(
                [slot.domain for slot in self._slots.values()]
                + [response_domain, endpoint_domain]
            )
        )
        self._controllables: dict[str, Controllable] = {
            name: Controllable(
                name=name,
                security_domain=slot.domain,
                description=slot.description
                or f"The value sent at {_sentinel(name)} in the request body.",
            )
            for name, slot in self._slots.items()
        }
        self._benign: dict[str, str] = {name: slot.default for name, slot in self._slots.items()}
        self._response_path = response_path
        self._timeout = timeout
        self._transport = transport
        self._max_attempts = max(1, max_attempts)  # always at least one attempt
        self._retry_backoff_base = retry_backoff_base
        self._max_retry_delay = max_retry_delay
        self._max_response_time = max_response_time
        self._max_response_bytes = max_response_bytes
        # One client per target instance, reused across attempts and run() calls
        # (connection pooling; the injected transport is NOT closed per attempt).
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout)

        self._reset_state()

    def _reset_state(self) -> None:
        self._last_response: str = ""
        self._raw_response: str = ""
        self._http_status: int | None = None
        self._error: str = ""
        self._sent: dict[str, str] = {}

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name=f"benign_{name}",
                security_domain=slot.domain,
                description=f"The value sent at {_sentinel(name)} when it is not injected.",
            )
            for name, slot in self._slots.items()
        ]

    def set_config(self, name: str, value: str) -> None:
        slot = name.removeprefix("benign_")
        if slot != name and slot in self._benign:
            self._benign[slot] = value

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
                description="The effective value sent at {{prompt}} (after injection), "
                "or '' when there is no prompt slot.",
            ),
            QuerySpec(
                name="sent_inputs",
                description="JSON object of the effective value sent at every slot "
                "(after injection).",
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
            return self._sent.get("prompt", "")
        if name == "sent_inputs":
            return json.dumps(self._sent)
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return self._domain

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return list(self._controllables.values())

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="endpoint",
                    security_domain=self._endpoint_domain,
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

        for name, ctrl in self._controllables.items():
            resp = await send_event(
                ControllablePreCallEvent(controllable=ctrl, request=f"Enter {name}:")
            )
            self._sent[name] = (
                resp.value if isinstance(resp, ControllableInjection) else self._benign[name]
            )

        # Each value is observable at its own slot's domain, so whoever holds that
        # surface (read & write, or read_only) sees what was actually sent there.
        for name, value in self._sent.items():
            slot = self._slots[name]
            emit(
                ObservableEvent(
                    observable=Observable(
                        name=f"{_SENT_PREFIX}{name}",
                        security_domain=slot.domain,
                        description=slot.description
                        or f"The value sent at {_sentinel(name)}.",
                    ),
                    content=value,
                )
            )

        await self._call(self._sent)

        emit(
            ObservableEvent(
                observable=Observable(
                    name="endpoint_response",
                    security_domain=self._response_domain,
                    description="The endpoint's extracted reply.",
                ),
                content=self._last_response[:500],
            )
        )

    async def _call(self, values: Mapping[str, str]) -> None:
        body = _render(self._body_template, values)
        for attempt in range(1, self._max_attempts + 1):
            try:
                status, retry_after, raw = await self._request_capped(body)
                self._http_status = status
                self._raw_response = raw
                if status == 429 and attempt < self._max_attempts:
                    await self._backoff(attempt, retry_after)
                    continue
                if status >= 500 and attempt < self._max_attempts:
                    await self._backoff(attempt, retry_after)
                    continue
                # >= 300 (not just >= 400): redirects are NOT followed
                # (follow_redirects defaults to False), so a 3xx (HTTP->HTTPS,
                # trailing-slash normalization, an auth/SSO redirect) is not the
                # app's reply — treat it as an error so the claim abstains rather
                # than scoring an empty/redirect body as the response.
                if status >= 300:
                    self._error = f"HTTP {status}"
                    return
                self._error = ""  # a prior transient attempt, if any, recovered
                try:
                    data = json.loads(raw)
                except ValueError:
                    # a 2xx with a non-JSON body: use the raw text as the reply.
                    self._last_response = raw
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
            except Exception as exc:  # noqa: BLE001 - InvalidURL / TimeoutError / ResponseTooLargeError
                # Not an httpx.HTTPError, so retrying won't help / would prolong a
                # hostile stall — record the type and stop. Covers: a malformed URL
                # (httpx.InvalidURL, raised synchronously); the total-time budget
                # exceeded (asyncio.timeout -> TimeoutError, e.g. a slow-trickle
                # server); and an over-cap body (ResponseTooLargeError). Type only: an
                # InvalidURL message embeds the URL (userinfo credentials). Never
                # propagates out of run() and crashes the sweep.
                self._http_status = None
                self._raw_response = ""
                self._error = type(exc).__name__
                return

    async def _request_capped(self, body: Any) -> tuple[int, str | None, str]:
        """POST and read the response under two bounds an untrusted endpoint cannot
        exceed: a total wall-clock cap (``asyncio.timeout`` — httpx's per-op read
        timeout does NOT bound a slow byte-trickle) and a response byte cap.

        Compression-bomb-safe: httpx decompresses a ``Content-Encoding`` response
        with no output limit, so a tiny gzip chunk that inflates to GBs would blow
        past a cap measured on the *decoded* stream before the check runs. We
        request ``Accept-Encoding: identity`` so a compliant endpoint sends
        plaintext, and if the endpoint compresses anyway we refuse to read the body
        at all — so httpx never decompresses. A plaintext body is then read via
        ``aiter_bytes`` (== the wire bytes, nothing to decode) under the byte cap.
        Returns ``(status_code, Retry-After, decoded_body)``."""
        # Force Accept-Encoding: identity UNCONDITIONALLY (stripping any operator
        # value, case-insensitively, so no duplicate header is sent): the bomb
        # defense below depends on never REQUESTING compression, so a caller-supplied
        # Accept-Encoding must not re-enable it and then trip the refusal on every
        # call (misreported as a size error).
        req_headers = {
            k: v for k, v in self._headers.items() if k.lower() != "accept-encoding"
        }
        req_headers["Accept-Encoding"] = "identity"
        async with asyncio.timeout(self._max_response_time):
            async with self._client.stream(
                self._method, self._url, json=body, headers=req_headers
            ) as resp:
                retry_after = resp.headers.get("Retry-After")
                enc = resp.headers.get("content-encoding", "").strip().lower()
                if enc and enc != "identity":
                    # Endpoint compressed despite Accept-Encoding: identity — reading
                    # the decoded stream would let httpx inflate a possible bomb past
                    # the cap, so refuse without ever decompressing.
                    raise ResponseTooLargeError
                total = 0
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_response_bytes:
                        raise ResponseTooLargeError
                    chunks.append(chunk)
                raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
                return resp.status_code, retry_after, raw

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


__all__ = [
    "ENDPOINT_TAG",
    "PROMPT_PLACEHOLDER",
    "RESPONSE_TAG",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "HttpEndpointTarget",
    "Slot",
]
