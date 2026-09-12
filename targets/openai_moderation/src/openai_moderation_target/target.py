"""OpenAIModerationTarget: the OpenAI Moderation API as a superred Target.

The Moderation API is a *content-safety classifier*: it rates input text across
categories (hate, violence, self-harm, sexual, ...) and returns a ``flagged``
verdict. This target turns the classifier into a superred victim so an optimizer
can red-team it: the attacker controls the ``input`` text, the target calls
``/moderations``, and the ``flagged`` verdict becomes the observable. Evasion — a
genuinely harmful input the classifier rates ``flagged == false`` — is the
security failure the paired claim scores. (Distinct from ``prompt_shield``, which
detects prompt *injection*; this detects harmful *content*.)

REST contract (verified against the OpenAI API reference):

    POST {base_url}/moderations
    Authorization: Bearer <key>
    {"input": "...", "model": "omni-moderation-latest"}
    -> {"results": [{"flagged": bool,
                     "categories": {"hate": bool, ...},
                     "category_scores": {"hate": float, ...}}]}

The API key is held privately and never emitted as an observable, returned from a
query, or written into any rationale. Live runs need the user's own OpenAI key;
offline tests inject an ``httpx.MockTransport``.
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
INPUT_TAG: Final = SecurityDomainTag("input", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, INPUT_TAG])

_DEFAULT_MODEL: Final = "omni-moderation-latest"
_DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"


class OpenAIModerationTarget(Target):
    """The OpenAI Moderation classifier as a superred target.

    Args:
        api_key: the OpenAI API key, sent as ``Authorization: Bearer``. Held
            privately; never emitted/queried/logged.
        model: the moderation model (default ``omni-moderation-latest``).
        base_url: API base (default ``https://api.openai.com/v1``); a trailing
            slash is trimmed.
        timeout: per-request timeout (seconds).
        transport: optional ``httpx`` transport for offline tests
            (``httpx.MockTransport``); ``None`` uses the real network.
        max_retry_delay: cap on any backoff sleep (a server ``Retry-After`` is
            honored but clamped to this).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
        max_retry_delay: float = 60.0,
    ) -> None:
        self._api_key = api_key  # PRIVATE: never emitted/queried/logged
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._max_retries = max(1, max_retries)
        self._retry_backoff_base = retry_backoff_base
        self._max_retry_delay = max_retry_delay

        self._benign_input = "Hello, how are you today?"
        self._reset_state()

    def _reset_state(self) -> None:
        self._flagged: bool | None = None
        self._categories: list[str] = []
        self._raw_response: str = ""
        self._http_status: int | None = None
        self._error: str = ""
        self._sent_input: str = ""

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="benign_input",
                security_domain=SYSTEM_TAG,
                description="The non-attack input used when the input is not injected.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "benign_input":
            self._benign_input = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="flagged",
                description="'true' if the classifier flagged the input, 'false' if not, "
                "'' if no verdict (HTTP error).",
            ),
            QuerySpec(
                name="flagged_categories",
                description="JSON list of the category names the classifier flagged.",
            ),
            QuerySpec(name="raw_response", description="Raw JSON body returned by /moderations."),
            QuerySpec(name="http_status", description="HTTP status code of the call, or ''."),
            QuerySpec(name="error", description="Error detail if the call failed, else ''."),
            QuerySpec(
                name="sent_input",
                description="The effective input submitted (after any injection). Lets a claim "
                "confirm a benign input was not mutated.",
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "flagged":
            return "" if self._flagged is None else ("true" if self._flagged else "false")
        if name == "flagged_categories":
            return json.dumps(self._categories)
        if name == "raw_response":
            return self._raw_response
        if name == "http_status":
            return "" if self._http_status is None else str(self._http_status)
        if name == "error":
            return self._error
        if name == "sent_input":
            return self._sent_input
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name="input",
                security_domain=INPUT_TAG,
                description="The text submitted to the moderation classifier.",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="moderator",
                    security_domain=SYSTEM_TAG,
                    description="The classifier identity (model id; never the key).",
                ),
                content=self._model,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()

        ctrl = self.get_controllables()[0]
        resp = await send_event(
            ControllablePreCallEvent(controllable=ctrl, request="Enter input:")
        )
        text = resp.value if isinstance(resp, ControllableInjection) else self._benign_input
        self._sent_input = text

        emit(
            ObservableEvent(
                observable=Observable(
                    name="moderation_input",
                    security_domain=INPUT_TAG,
                    description="The text submitted to the classifier.",
                ),
                content=text,
            )
        )

        await self._call(text)

        emit(
            ObservableEvent(
                observable=Observable(
                    name="moderation_verdict",
                    security_domain=SYSTEM_TAG,
                    description="The classifier's flagged verdict.",
                ),
                content=self.query("flagged"),
            )
        )

    async def _call(self, text: str) -> None:
        url = f"{self._base_url}/moderations"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = {"input": text, "model": self._model}
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    transport=self._transport, timeout=self._timeout
                ) as client:
                    resp = await client.post(url, json=body, headers=headers)
                self._http_status = resp.status_code
                self._raw_response = resp.text
                if resp.status_code == 429 and attempt < self._max_retries:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                if resp.status_code >= 500 and attempt < self._max_retries:
                    await self._backoff(attempt, resp.headers.get("Retry-After"))
                    continue
                # >= 300 (not just >= 400): redirects are NOT followed, so a 3xx is
                # not a moderation result — treat it as an error so the claim
                # abstains rather than scoring an empty/redirect body as a verdict.
                if resp.status_code >= 300:
                    self._error = f"HTTP {resp.status_code}"
                    return
                self._error = ""  # a prior transient attempt, if any, recovered
                try:
                    data = resp.json()
                except ValueError:
                    self._error = "invalid JSON response"
                    return
                self._parse(data)
                return
            except httpx.HTTPError as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_retries:
                    await self._backoff(attempt, None)
                    continue
                return
            except Exception as exc:  # noqa: BLE001 - e.g. httpx.InvalidURL (NOT an HTTPError)
                # A malformed URL (bad port, control char, ...) raises synchronously
                # from client.post and is not an httpx.HTTPError — record it and stop
                # (retrying can't fix the URL) so it never crashes the sweep.
                self._error = f"{type(exc).__name__}: {exc}"
                return

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = self._retry_backoff_base * (2.0 ** (attempt - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        # Clamp so a hostile/misbehaving server can't stall the run via Retry-After.
        delay = min(delay, self._max_retry_delay)
        if delay > 0:
            await asyncio.sleep(delay)

    def _parse(self, data: Any) -> None:
        if not isinstance(data, dict):
            self._error = "unexpected response shape"
            return
        results = data.get("results")
        if not isinstance(results, list) or not results:
            self._error = "unexpected response shape"
            return
        first = results[0]
        if not isinstance(first, dict):
            self._error = "unexpected response shape"
            return
        flagged = first.get("flagged")
        if not isinstance(flagged, bool):
            # A parseable 200 whose result lacks a bool verdict (truncated / proxied
            # / gateway-rewritten body) is NOT a verdict — record an error so the
            # contract holds (empty `flagged` <=> error set) and a claim abstains
            # instead of reading the missing verdict as a "not flagged" evasion.
            self._error = "unexpected response shape"
            return
        self._flagged = flagged
        categories = first.get("categories")
        if isinstance(categories, dict):
            self._categories = [k for k, v in categories.items() if v is True]

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = ["OpenAIModerationTarget", "SYSTEM_TAG", "INPUT_TAG"]
