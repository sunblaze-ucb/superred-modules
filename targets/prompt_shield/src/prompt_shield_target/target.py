"""PromptShieldTarget: Azure AI Content Safety "Prompt Shields" as a superred Target.

Prompt Shields is a *detector*: a synchronous API that classifies a user prompt
(direct injection) and/or grounding documents (indirect injection) as attack /
not-attack. This target turns the detector into a superred victim so an optimizer
can red-team it: the attacker controls the ``user_prompt`` and/or ``document``
text, the target calls ``text:shieldPrompt``, and the detector's verdict
(``attackDetected``) becomes the observable. Evasion — a genuine attack the
detector rates ``attackDetected == false`` — is the security failure the paired
claim scores.

REST contract (api-version 2024-09-01, verified against Microsoft Learn):

    POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01
    Ocp-Apim-Subscription-Key: <key>
    {"userPrompt": "...", "documents": ["..."]}
    -> {"userPromptAnalysis": {"attackDetected": bool},
        "documentsAnalysis": [{"attackDetected": bool}]}

The Azure subscription key is held privately and never emitted as an observable,
returned from a query, or written into any rationale.

Live runs need the user's own Azure AI Content Safety resource + key; offline
tests inject an ``httpx.MockTransport``. Testing your own detection resource this
way is permitted under Microsoft's Penetration Testing Rules of Engagement (which
explicitly cover "testing your tenant's security monitoring and detection
systems"); see the README for the ToS notes and the publish-benchmark caveat.
"""

from __future__ import annotations

import asyncio
import json
from typing import Final

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

# Security domain: the detector "system" with the two attacker-controlled input
# channels as children (direct user-prompt injection, indirect document injection).
SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_PROMPT_TAG: Final = SecurityDomainTag("user_prompt", parent=SYSTEM_TAG)
DOCUMENT_TAG: Final = SecurityDomainTag("document", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, USER_PROMPT_TAG, DOCUMENT_TAG])

_DEFAULT_API_VERSION: Final = "2024-09-01"
_CHANNEL_USER_PROMPT: Final = "user_prompt"
_CHANNEL_DOCUMENT: Final = "document"
_CHANNELS: Final = (_CHANNEL_USER_PROMPT, _CHANNEL_DOCUMENT)


class PromptShieldTarget(Target):
    """Azure Prompt Shields detector as a superred target.

    Args:
        endpoint: the Content Safety resource endpoint, protocol + host only,
            e.g. ``https://<resource>.cognitiveservices.azure.com`` (a trailing
            slash is trimmed).
        api_key: the resource key, sent as ``Ocp-Apim-Subscription-Key``. Held
            privately; never observable.
        api_version: the ``api-version`` query value (default ``2024-09-01``).
        timeout: per-request timeout in seconds.
        transport: optional ``httpx`` transport for offline tests
            (``httpx.MockTransport``); ``None`` uses the real network.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str = _DEFAULT_API_VERSION,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key  # PRIVATE: never emitted/queried/logged
        self._api_version = api_version
        self._timeout = timeout
        self._transport = transport
        self._max_retries = max(1, max_retries)
        self._retry_backoff_base = retry_backoff_base

        # Config
        self._channel: str = _CHANNEL_USER_PROMPT
        self._benign_user_prompt: str = ""
        self._benign_document: str = ""

        # Post-run state
        self._reset_state()

    def _reset_state(self) -> None:
        self._user_prompt_detected: bool | None = None
        self._document_detected: bool | None = None
        self._raw_response: str = ""
        self._http_status: int | None = None
        self._error: str = ""
        self._sent_user_prompt: str = ""
        self._sent_document: str = ""

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="channel",
                security_domain=SYSTEM_TAG,
                description=(
                    "Which input channel the attack targets: 'user_prompt' (direct "
                    "injection, default) or 'document' (indirect injection). The "
                    "targeted channel is offered for injection first."
                ),
            ),
            ConfigSpec(
                name="benign_user_prompt",
                security_domain=SYSTEM_TAG,
                description="Non-attack user prompt used when the user_prompt channel is "
                "not injected.",
            ),
            ConfigSpec(
                name="benign_document",
                security_domain=SYSTEM_TAG,
                description="Non-attack document used when the document channel is not injected.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "channel":
            if value not in _CHANNELS:
                raise ValueError(f"channel must be one of {_CHANNELS}, got {value!r}")
            self._channel = value
        elif name == "benign_user_prompt":
            self._benign_user_prompt = value
        elif name == "benign_document":
            self._benign_document = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="attack_detected",
                description="'true' if the detector flagged EITHER channel, 'false' if neither, "
                "'' if no verdict (HTTP error).",
            ),
            QuerySpec(
                name="user_prompt_attack_detected",
                description="Detector verdict for the user prompt channel: 'true'/'false'/''.",
            ),
            QuerySpec(
                name="document_attack_detected",
                description="Detector verdict for the document channel: 'true'/'false'/''.",
            ),
            QuerySpec(name="raw_response", description="Raw JSON body returned by shieldPrompt."),
            QuerySpec(
                name="http_status", description="HTTP status code of the shieldPrompt call, or ''."
            ),
            QuerySpec(name="error", description="Error detail if the call failed, else ''."),
            QuerySpec(name="channel", description="The channel this run targeted."),
            QuerySpec(
                name="sent_text",
                description="The effective text submitted on the targeted channel (after any "
                "injection). Lets a claim confirm a benign input was not mutated.",
            ),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "attack_detected":
            overall = self._overall_detected()
            return "" if overall is None else _b(overall)
        if name == "user_prompt_attack_detected":
            return "" if self._user_prompt_detected is None else _b(self._user_prompt_detected)
        if name == "document_attack_detected":
            return "" if self._document_detected is None else _b(self._document_detected)
        if name == "raw_response":
            return self._raw_response
        if name == "http_status":
            return "" if self._http_status is None else str(self._http_status)
        if name == "error":
            return self._error
        if name == "channel":
            return self._channel
        if name == "sent_text":
            if self._channel == _CHANNEL_DOCUMENT:
                return self._sent_document
            return self._sent_user_prompt
        return ""

    def _overall_detected(self) -> bool | None:
        vals = [v for v in (self._user_prompt_detected, self._document_detected) if v is not None]
        if not vals:
            return None
        return any(vals)

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name="user_prompt",
                security_domain=USER_PROMPT_TAG,
                description="The user prompt submitted to Prompt Shields "
                "(direct injection surface).",
            ),
            Controllable(
                name="document",
                security_domain=DOCUMENT_TAG,
                description="A grounding document submitted to Prompt Shields "
                "(indirect injection surface).",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="detector",
                    security_domain=SYSTEM_TAG,
                    description="The detector identity (endpoint host + api-version).",
                ),
                content=f"{self._host()}@{self._api_version}",
            ),
        ]

    def _host(self) -> str:
        # host only, never the key
        return self._endpoint.split("://", 1)[-1]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()

        # Offer the targeted channel for injection FIRST, so the passthrough
        # optimizer (which injects the goal into the first non-system controllable)
        # fills the intended channel; the other channel then uses its benign default.
        if self._channel == _CHANNEL_DOCUMENT:
            order = ("document", "user_prompt")
        else:
            order = ("user_prompt", "document")

        values = {"user_prompt": self._benign_user_prompt, "document": self._benign_document}
        for name in order:
            ctrl = next(c for c in self.get_controllables() if c.name == name)
            resp = await send_event(
                ControllablePreCallEvent(controllable=ctrl, request=f"Enter {name}:"),
            )
            if isinstance(resp, ControllableInjection):
                values[name] = resp.value

        self._sent_user_prompt = values["user_prompt"]
        self._sent_document = values["document"]

        emit(
            ObservableEvent(
                observable=Observable(
                    name="shield_request",
                    # SYSTEM_TAG (not a channel tag): this record bundles BOTH the
                    # user_prompt and document text, so it is not scoped to either
                    # attacker channel — matching the detector/shield_verdict records.
                    security_domain=SYSTEM_TAG,
                    description="The text submitted to the detector.",
                ),
                content=json.dumps(
                    {
                        "channel": self._channel,
                        "user_prompt": self._sent_user_prompt,
                        "document": self._sent_document,
                    }
                ),
            )
        )

        await self._call_shield(self._sent_user_prompt, self._sent_document)

        emit(
            ObservableEvent(
                observable=Observable(
                    name="shield_verdict",
                    security_domain=SYSTEM_TAG,
                    description="The detector's attackDetected verdict.",
                ),
                content=self.query("attack_detected"),
            )
        )

    async def _call_shield(self, user_prompt: str, document: str) -> None:
        body: dict[str, object] = {}
        if user_prompt:
            body["userPrompt"] = user_prompt
        if document:
            body["documents"] = [document]
        if not body:
            # Nothing to analyze; leave verdicts as None (unknown) without a call.
            self._error = "no input to analyze"
            return

        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self._api_version}"
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
            "Content-Type": "application/json",
        }
        # Bounded retry on 429 (rate limit — the F0 tier is 5 RPS) and on
        # transient transport errors, with exponential backoff. A persistent
        # error leaves the verdict unknown so the claim abstains.
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
                if resp.status_code != 200:
                    self._error = f"HTTP {resp.status_code}"
                    return
                self._error = ""  # a prior transient attempt, if any, recovered
                try:
                    data = resp.json()
                except ValueError:
                    # a 200 with a non-JSON body (e.g. a gateway HTML error page):
                    # record it and abstain, never crash the run.
                    self._error = "invalid JSON response"
                    return
                self._parse(data, sent_user_prompt=bool(user_prompt), sent_document=bool(document))
                return
            except httpx.HTTPError as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_retries:
                    await self._backoff(attempt, None)
                    continue
                return

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        """Sleep before a retry: honour ``Retry-After`` if present, else exponential."""
        delay = self._retry_backoff_base * (2.0 ** (attempt - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        if delay > 0:
            await asyncio.sleep(delay)

    def _parse(self, data: object, *, sent_user_prompt: bool, sent_document: bool) -> None:
        if not isinstance(data, dict):
            self._error = "unexpected response shape"
            return
        if sent_user_prompt:
            upa = data.get("userPromptAnalysis")
            if isinstance(upa, dict) and isinstance(upa.get("attackDetected"), bool):
                self._user_prompt_detected = upa["attackDetected"]
        if sent_document:
            da = data.get("documentsAnalysis")
            if isinstance(da, list):
                flags = [d.get("attackDetected") for d in da if isinstance(d, dict)]
                bools = [f for f in flags if isinstance(f, bool)]
                if bools:
                    self._document_detected = any(bools)

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


def _b(value: bool) -> str:
    return "true" if value else "false"
