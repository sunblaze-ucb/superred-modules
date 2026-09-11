"""Azure AI Content Safety "Prompt Shields" as a superred target.

Red-teams the *detector*: the attacker controls the ``user_prompt`` (direct
injection) and/or ``document`` (indirect injection) text, the target calls
``text:shieldPrompt``, and the detector's ``attackDetected`` verdict is the
observable. A genuine attack the detector rates ``attackDetected == false`` is an
evasion — the failure the paired ``prompt_shield`` claim scores.

Live runs need the user's own Azure Content Safety resource + key; offline tests
inject an ``httpx.MockTransport``.
"""

from __future__ import annotations

import httpx
from superred.core.controller import TargetFactory

from prompt_shield_target.target import (
    DOCUMENT_TAG,
    SYSTEM_TAG,
    USER_PROMPT_TAG,
    PromptShieldTarget,
)

_DEFAULT_API_VERSION = "2024-09-01"


def prompt_shield_target_factory(
    *,
    endpoint: str,
    api_key: str,
    api_version: str = _DEFAULT_API_VERSION,
    timeout: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
    concurrency: int = 1,
) -> TargetFactory:
    """Build a :class:`TargetFactory` of fresh :class:`PromptShieldTarget` instances.

    ``concurrency`` defaults to 1 because the free (F0) Content Safety tier is
    rate-limited; raise it for a standard-tier resource.
    """
    return TargetFactory(
        create=lambda: PromptShieldTarget(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
            transport=transport,
        ),
        concurrency=concurrency,
    )


__all__ = [
    "PromptShieldTarget",
    "prompt_shield_target_factory",
    "SYSTEM_TAG",
    "USER_PROMPT_TAG",
    "DOCUMENT_TAG",
]
