"""Static "detailed system specification" content for the inspect-agent target.

A leaked, free-text design and threat-intelligence brief on this target system,
loaded from the sibling ``system_specification.md`` and exposed once as the
``detailed_system_specification`` static observable (see
:mod:`inspect_agent_target.observables`).  It describes the system's purpose,
architecture, runtime logic, and hardcoded prompts/code, each at abstract /
specific / examples depth, so an attacker who obtained it can understand how the
system works and craft stronger attacks.  The text is static (the invariant
design); the configured model, prompts, and tool catalogue have their own
observables.
"""

from __future__ import annotations

from importlib import resources

DETAILED_SYSTEM_SPECIFICATION: str = (
    resources.files("inspect_agent_target")
    .joinpath("system_specification.md")
    .read_text(encoding="utf-8")
)
"""The leaked system-specification brief (markdown), read once at import time."""
