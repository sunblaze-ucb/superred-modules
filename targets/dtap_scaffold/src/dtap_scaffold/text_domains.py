"""The text-only DTAP domain allowlist.

DTAP ships 14 domains; this port covers only the TEXT-based ones. Three are
excluded because their environment is vision/GUI driven and cannot be observed
or controlled as text:

- ``browser``  -- the browser MCP server drives a headless Chromium and feeds
  full-page screenshots (image tokens), not text.
- ``macos``    -- a QEMU macOS desktop automated via screen control.
- ``windows``  -- a QEMU Windows desktop automated via screen control.

The Task selects which environments are active per run (see
``config_specs.ACTIVE_MCP_SERVERS``); the scaffolding validates that selection
against the active environments' domains here, rejecting any excluded domain.
The mapping from an MCP server name to its domain lives in the vendored
``mcp.yaml`` and is resolved by ``docker/env_registry.py`` (added with the
Docker lifecycle); this module is the single source of truth for *which domains
are in scope at all*.
"""

from __future__ import annotations

TEXT_ONLY_DOMAINS: frozenset[str] = frozenset(
    {
        "code",
        "crm",
        "customer-service",
        "finance",
        "legal",
        "medical",
        "os-filesystem",
        "research",
        "telecom",
        "travel",
        "workflow",
    }
)
"""The 11 text-based DTAP domains this port supports."""

EXCLUDED_DOMAINS: frozenset[str] = frozenset({"browser", "macos", "windows"})
"""Vision/GUI DTAP domains deliberately out of scope for this text-only port."""


def is_text_only_domain(domain: str) -> bool:
    """Whether *domain* is one of the supported text-only DTAP domains."""
    return domain in TEXT_ONLY_DOMAINS


def require_text_only_domain(domain: str) -> str:
    """Return *domain* if supported, else raise ``ValueError``.

    ``windows`` reports its DTAP domain as ``os``; callers should map it before
    validation. This function rejects any domain not in
    :data:`TEXT_ONLY_DOMAINS` (including the three excluded GUI domains).
    """
    if domain not in TEXT_ONLY_DOMAINS:
        if domain in EXCLUDED_DOMAINS:
            raise ValueError(
                f"DTAP domain {domain!r} is vision/GUI and excluded from this "
                "text-only port (supported: "
                f"{', '.join(sorted(TEXT_ONLY_DOMAINS))})."
            )
        raise ValueError(
            f"Unknown DTAP domain {domain!r}; supported text-only domains: "
            f"{', '.join(sorted(TEXT_ONLY_DOMAINS))}."
        )
    return domain


__all__ = [
    "TEXT_ONLY_DOMAINS",
    "EXCLUDED_DOMAINS",
    "is_text_only_domain",
    "require_text_only_domain",
]
