"""Static "detailed system specification" content for the DTAP agent target.

A leaked, free-text design and threat-intelligence brief on this target system,
loaded from the sibling ``system_specification.md`` and exposed once as the
``detailed_system_specification`` static observable (see
:mod:`dtap_scaffold.observables`). It describes the system's purpose,
architecture, runtime logic, and the injection wiring, each at abstract /
specific / examples depth, so an attacker who obtained it can understand how the
target works and operates and craft stronger attacks. The text is static (the
invariant design of the Dockerized-agent + MCP-proxy + env-stack setup); the
configured model, the active environment list, and the per-tool catalogue have
their own observables (``model_identity`` / ``active_environments`` /
``tool_catalogue``), so this brief describes HOW those surfaces work, not their
current per-run VALUES.
"""

from __future__ import annotations

from importlib import resources

DETAILED_SYSTEM_SPECIFICATION: str = (
    resources.files("dtap_scaffold")
    .joinpath("system_specification.md")
    .read_text(encoding="utf-8")
)
"""The leaked system-specification brief (markdown), read once at import time."""
