"""inspect_agent_target: a general inspect-ai tool-calling agent target for superred.

Benchmark-agnostic.  A SecurityClaim supplies the tool implementations (via a
``tool_resolver``), per-task config (prompts, tool names), and optionally a
per-tool trust-boundary scope map (``tool_scopes``); this target runs the inspect
tool-calling loop and exposes the resulting message trace.  Attacker surfaces:
the ``system_prompt`` / ``user_prompt`` controllables, an AgentDojo-style
tool-catalogue surface (register / replace / unregister / rewrite-description,
fired once at run start) for poisoning the tool registry, and one
``tool:<name>`` output-injection controllable per tool (scoped to that tool's
trust boundary) for poisoning what a tool returns.
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
    USER_PROMPT_CTRL,
    tool_output_controllable,
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
    MESSAGE_LIMIT_TAG,
    MODEL_IDENTITY_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADDABLE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    USER_TAG,
    build_domain,
    tool_tag_closure,
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
    "build_domain",
    "tool_tag_closure",
    "USER_TAG",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "MODEL_IDENTITY_TAG",
    "MESSAGE_LIMIT_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_ADDABLE_TAG",
    "TOOLS_TAG",
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
    "tool_output_controllable",
    "TOOL_CATALOG_LISTING_OBS",
    "agent_tool_response_observable",
    "CONTROLLABLES",
    "__version__",
]
