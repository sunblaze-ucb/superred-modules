"""asb_target: Agent Security Bench (ASB) agent as a superred Target.

Runs ASB's real vendored plan-then-execute agent loop (pinned to upstream
``ASB_UPSTREAM_SHA``) against a litellm proxy. It is a bare runtime that
exposes the four ASB injection surfaces (DPI/OPI/PoT/MP) as superred
Controllables on a trust-boundary forest (roots: user, system, tools,
memory), restores ASB's durable memory store, and performs no injection by
default. The benchmark-specific data, grading, and tasks live in the paired
``security_claim_asb`` package.
"""

from __future__ import annotations

from asb_target._vendor import ASB_UPSTREAM_SHA, ensure_vendor_on_path

ensure_vendor_on_path()

from asb_target.security_tags import (  # noqa: E402
    AGENT_TRACE_TAG,
    DOMAIN,
    MEMORY_TAG,
    NORMAL_TOOL_NAMES,
    SCENARIO_TOOL_TAGS,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_OBSERVATION_TAGS,
    TOOLS_TAG,
    USER_TAG,
)
from asb_target.target import AsbTarget  # noqa: E402

__all__ = [
    "AsbTarget",
    "ASB_UPSTREAM_SHA",
    "DOMAIN",
    # the trust-boundary tags (scope building blocks)
    "USER_TAG",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "AGENT_TRACE_TAG",
    "TOOLS_TAG",
    "SCENARIO_TOOL_TAGS",
    "TOOL_OBSERVATION_TAGS",
    "NORMAL_TOOL_NAMES",
    "MEMORY_TAG",
]
