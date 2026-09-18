"""HTTP-endpoint target for superred.

Points superred at an arbitrary HTTP LLM/chat endpoint (your own deployed app):
the attacker controls the values placed JSON-safely into a configurable request
body, one :class:`Slot` per ``{{name}}`` placeholder, each declared with the
security domain its value arrives through; the reply is extracted from the JSON
response via a dot path. Auth headers are held privately and never emitted.
Offline-testable via ``httpx.MockTransport``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
from superred.core.controller import TargetFactory
from superred.core.types.security_domain import SecurityDomainTag

from http_endpoint_target.target import (
    ENDPOINT_TAG,
    PROMPT_PLACEHOLDER,
    RESPONSE_TAG,
    SYSTEM_TAG,
    USER_INPUT_TAG,
    HttpEndpointTarget,
    Slot,
)


def http_endpoint_target_factory(
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
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`HttpEndpointTarget` instances.

    Every instance shares the same ``slots`` and domain tag objects, so a
    Controller scope built from those tags applies to all of them.
    """
    return TargetFactory(
        create=lambda: HttpEndpointTarget(
            url=url,
            method=method,
            headers=headers,
            body_template=body_template,
            slots=slots,
            response_path=response_path,
            response_domain=response_domain,
            endpoint_domain=endpoint_domain,
            timeout=timeout,
            transport=transport,
            max_attempts=max_attempts,
            retry_backoff_base=retry_backoff_base,
            max_retry_delay=max_retry_delay,
            max_response_time=max_response_time,
            max_response_bytes=max_response_bytes,
        ),
        concurrency=concurrency,
    )


__all__ = [
    "ENDPOINT_TAG",
    "PROMPT_PLACEHOLDER",
    "RESPONSE_TAG",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "HttpEndpointTarget",
    "Slot",
    "http_endpoint_target_factory",
]
