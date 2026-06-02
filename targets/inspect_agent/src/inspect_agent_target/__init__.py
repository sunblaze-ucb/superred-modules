"""inspect_agent_target: a general inspect-ai tool-calling agent target for superred.

Benchmark-agnostic.  A SecurityClaim supplies the tool implementations (via a
``tool_resolver``) and per-task config (prompts, tool names); this target runs
the inspect tool-calling loop and exposes the resulting message trace.  It also
exposes an AgentDojo-style tool-catalogue Controllable surface (register /
replace / unregister / rewrite-description) fired every turn, so attacker-scoped
optimizers from other claims can poison the tool registry.
"""

from __future__ import annotations

from inspect_agent_target.controllables import (
    CONTROLLABLES,
    SYSTEM_PROMPT_CTRL,
    TOOL_CATALOG_CTRLS,
    TOOL_CATALOG_REGISTER_CTRL,
    TOOL_CATALOG_REPLACE_CTRL,
    TOOL_CATALOG_REWRITE_DOC_CTRL,
    TOOL_CATALOG_UNREGISTER_CTRL,
    TOOL_OUTPUT_CTRL,
    USER_PROMPT_CTRL,
)
from inspect_agent_target.observables import (
    TOOL_CATALOG_LISTING_OBS,
    agent_tool_response_observable,
)
from inspect_agent_target.rollout import run_rollout, static_tools_provider
from inspect_agent_target.security_tags import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    AGENT_TRACE_TOOL_RESPONSES_TAG,
    DOMAIN,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_READABLE_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_READABLE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOL_OUTPUT_TAG,
    USER_TAG,
)
from inspect_agent_target.target import InspectAgentTarget, ToolResolver
from inspect_agent_target.tool_catalog import ToolCatalog

__version__ = "0.1.0"

__all__ = [
    "InspectAgentTarget",
    "ToolResolver",
    "ToolCatalog",
    "run_rollout",
    "static_tools_provider",
    "DOMAIN",
    "USER_TAG",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_PROMPT_READABLE_TAG",
    "MODEL_IDENTITY_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_READABLE_TAG",
    "TOOL_CATALOGUE_ADDABLE_TAG",
    "TOOL_OUTPUT_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "AGENT_TRACE_TOOL_RESPONSES_TAG",
    "SYSTEM_PROMPT_CTRL",
    "USER_PROMPT_CTRL",
    "TOOL_CATALOG_REGISTER_CTRL",
    "TOOL_CATALOG_REPLACE_CTRL",
    "TOOL_CATALOG_UNREGISTER_CTRL",
    "TOOL_CATALOG_REWRITE_DOC_CTRL",
    "TOOL_CATALOG_CTRLS",
    "TOOL_OUTPUT_CTRL",
    "TOOL_CATALOG_LISTING_OBS",
    "agent_tool_response_observable",
    "CONTROLLABLES",
    "__version__",
]
