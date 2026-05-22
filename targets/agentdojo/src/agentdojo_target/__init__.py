"""agentdojo_target: composite AgentDojo target for superred.

Exposes a single :class:`AgentDojoTarget` instance unioning all four
AgentDojo suites (banking, workspace, slack, travel) behind a wrapped
function runtime that converts every read into an on-demand controllable
event and exposes the tool catalogue as a separate controllable surface.
Benchmark version pinned via :data:`agentdojo_target.BENCHMARK_VERSION`
(currently the latest released, ``v1.2.2``).

See ``README.md`` for usage and ``ASSUMPTIONS.md`` for divergences from
AgentDojo upstream.

v0.1.0 is alpha; the surface may change without notice.
"""

from __future__ import annotations

from agentdojo_target.controllables import (
    CONTROLLABLES,
    READ_CTRLS,
    SYSTEM_PROMPT_CTRL,
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    USER_PROMPT_CTRL,
)
from agentdojo_target.env import CompositeEnvironment, sync_initial_fields
from agentdojo_target.observables import (
    COMPOSITE_ENV_SNAPSHOT_OBS,
    MODEL_IDENTITY_OBS,
    STATIC_OBSERVABLE_SPECS,
    SYSTEM_PROMPT_OBS,
    TOOL_CATALOG_LISTING_OBS,
)
from agentdojo_target.seed_loader import BENCHMARK_VERSION
from agentdojo_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    CONTENT_1P_DATA_1P_TAG,
    CONTENT_1P_DATA_3P_TAG,
    CONTENT_3P_DATA_1P_TAG,
    CONTENT_3P_DATA_3P_TAG,
    DOMAIN,
    MODEL_IDENTITY_TAG,
    PROMPT_READABLE_TAG,
    PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    USER_TAG,
)
from agentdojo_target.system_prompt import default_system_prompt
from agentdojo_target.target import AgentDojoTarget

__version__ = "0.1.0"

__all__ = [
    # Target
    "AgentDojoTarget",
    "BENCHMARK_VERSION",
    "CompositeEnvironment",
    "sync_initial_fields",
    # Security domain forest
    "DOMAIN",
    "SYSTEM_TAG",
    "PROMPT_TAG", "PROMPT_READABLE_TAG",
    "TOOL_CATALOGUE_TAG", "TOOL_CATALOGUE_READABLE_TAG", "TOOL_CATALOGUE_ADDABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "AGENT_TRACE_TAG", "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG", "AGENT_TRACE_TOOL_RESPONSES_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "CONTENT_1P_DATA_1P_TAG", "CONTENT_1P_DATA_3P_TAG",
    "CONTENT_3P_DATA_1P_TAG", "CONTENT_3P_DATA_3P_TAG",
    # Controllables
    "CONTROLLABLES",
    "READ_CTRLS",
    "SYSTEM_PROMPT_CTRL",
    "USER_PROMPT_CTRL",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    # Observables
    "STATIC_OBSERVABLE_SPECS",
    "MODEL_IDENTITY_OBS",
    "SYSTEM_PROMPT_OBS",
    "TOOL_CATALOG_LISTING_OBS",
    "COMPOSITE_ENV_SNAPSHOT_OBS",
    # System prompt
    "default_system_prompt",
]
