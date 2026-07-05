"""Opt-in real Gemini provider helpers (requires ``GEMINI_API_KEY``)."""

from __future__ import annotations

import os
from typing import Any

from openclaw_target import OpenClawTarget

DEFAULT_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/v1"
DEFAULT_GEMINI_MODEL = "google/gemini-2.5-flash"
DEFAULT_PROVIDER_TIMEOUT_S = 180


def gemini_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def provider_model() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_MODEL", DEFAULT_GEMINI_MODEL)


def provider_base_url() -> str:
    return os.environ.get("OPENCLAW_PROVIDER_BASE_URL", DEFAULT_GEMINI_BASE)


def gemini_target(
    *,
    timeout_s: float = DEFAULT_PROVIDER_TIMEOUT_S,
    **kwargs: Any,
) -> OpenClawTarget:
    key = gemini_api_key()
    assert key is not None
    return OpenClawTarget(
        managed=True,
        model_id=provider_model(),
        provider_base_url=provider_base_url(),
        provider_api_key=key,
        agent_timeout_s=timeout_s,
        **kwargs,
    )


def docker_gemini_target(
    *,
    timeout_s: float = DEFAULT_PROVIDER_TIMEOUT_S,
    **kwargs: Any,
) -> OpenClawTarget:
    from test_support.docker import docker_target

    key = gemini_api_key()
    assert key is not None
    return docker_target(
        model_id=provider_model(),
        provider_base_url=provider_base_url(),
        provider_api_key=key,
        agent_timeout_s=timeout_s,
        **kwargs,
    )
