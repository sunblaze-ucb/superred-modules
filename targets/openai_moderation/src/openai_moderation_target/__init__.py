"""OpenAI Moderation target for superred.

Turns the OpenAI Moderation content-safety classifier into a superred victim: the
attacker controls the input text, the target calls ``/moderations``, and the
``flagged`` verdict is the observable (evasion — harmful input rated
``flagged == false`` — is the failure). The API key is held privately and never
emitted. Offline-testable via ``httpx.MockTransport``.
"""

from __future__ import annotations

import httpx
from superred.core.controller import TargetFactory

from openai_moderation_target.target import (
    INPUT_TAG,
    SYSTEM_TAG,
    OpenAIModerationTarget,
)


def openai_moderation_target_factory(
    *,
    api_key: str,
    model: str = "omni-moderation-latest",
    base_url: str = "https://api.openai.com/v1",
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
    max_retries: int = 3,
    retry_backoff_base: float = 0.5,
    max_retry_delay: float = 60.0,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`OpenAIModerationTarget` instances."""
    return TargetFactory(
        create=lambda: OpenAIModerationTarget(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            max_retry_delay=max_retry_delay,
        ),
        concurrency=concurrency,
    )


__all__ = [
    "OpenAIModerationTarget",
    "openai_moderation_target_factory",
    "SYSTEM_TAG",
    "INPUT_TAG",
]
