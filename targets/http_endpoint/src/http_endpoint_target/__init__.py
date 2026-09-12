"""HTTP-endpoint target for superred.

Points superred at an arbitrary HTTP LLM/chat endpoint (your own deployed app):
the attacker controls the prompt, which is placed JSON-safely into a configurable
request body; the reply is extracted from the JSON response via a dot path. Auth
headers are held privately and never emitted. Offline-testable via
``httpx.MockTransport``.
"""

from __future__ import annotations

from typing import Any

import httpx
from superred.core.controller import TargetFactory

from http_endpoint_target.target import (
    PROMPT_PLACEHOLDER,
    SYSTEM_TAG,
    USER_INPUT_TAG,
    HttpEndpointTarget,
)


def http_endpoint_target_factory(
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
    max_response_time: float = 60.0,
    max_response_bytes: int = 1_000_000,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`HttpEndpointTarget` instances."""
    return TargetFactory(
        create=lambda: HttpEndpointTarget(
            url=url,
            method=method,
            headers=headers,
            body_template=body_template,
            response_path=response_path,
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
    "HttpEndpointTarget",
    "http_endpoint_target_factory",
    "PROMPT_PLACEHOLDER",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
]
