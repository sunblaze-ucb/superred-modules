"""Shared scaffolding for the DecodingTrust-Agent (DTAP) superred targets.

This package holds everything agent-agnostic: the security-domain forest, the
controllables/observables, the pre-run/post-run specs, the text-only domain
allowlist, and (added incrementally) the env/MCP/Docker lifecycle, the host MCP
proxy, the env-injection bridge, the byte-faithful judge runner, the dataset
loader, and the agent-agnostic ``DtapAgentTarget`` base class. The two concrete
targets (Claude Code, OpenClaw) subclass the base and implement only a handful
of agent-specific hooks; the DTAP-BENCH claim drives either through the Target
ABC.
"""

from __future__ import annotations

from dtap_scaffold.config_specs import CONFIG_SPECS
from dtap_scaffold.controllables import (
    FIXED_CONTROLLABLES,
    SKILL_CTRL,
    SYSTEM_PROMPT_CTRL,
    TOOL_DESCRIPTION_OVERRIDE_CTRL,
    TOOL_DESCRIPTION_SUFFIX_CTRL,
    USER_PROMPT_CTRL,
    env_inject_controllable,
    env_tool_output_controllable,
)
from dtap_scaffold.forest import (
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    DOMAIN,
    ENVIRONMENT_TAG,
    FIXED_TAGS,
    MAX_TURNS_TAG,
    MODEL_IDENTITY_TAG,
    SKILL_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    TOOL_CATALOGUE_TAG,
    TOOLS_TAG,
    USER_TAG,
    build_domain,
    env_server_tag,
    tools_server_tag,
)
from dtap_scaffold.observables import (
    STATIC_OBSERVABLE_SPECS,
    agent_message_observable,
    native_tool_observable,
)
from dtap_scaffold.query_specs import QUERY_SPECS
from dtap_scaffold.text_domains import (
    EXCLUDED_DOMAINS,
    TEXT_ONLY_DOMAINS,
    is_text_only_domain,
    require_text_only_domain,
)

__all__ = [
    # forest
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "SKILL_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_ADD_TAG",
    "TOOL_CATALOGUE_EDIT_TAG",
    "TOOL_CATALOGUE_REMOVE_TAG",
    "MODEL_IDENTITY_TAG",
    "DETAILED_SYSTEM_SPECIFICATION_TAG",
    "MAX_TURNS_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "ENVIRONMENT_TAG",
    "FIXED_TAGS",
    "DOMAIN",
    "build_domain",
    "tools_server_tag",
    "env_server_tag",
    # controllables
    "USER_PROMPT_CTRL",
    "SYSTEM_PROMPT_CTRL",
    "SKILL_CTRL",
    "TOOL_DESCRIPTION_OVERRIDE_CTRL",
    "TOOL_DESCRIPTION_SUFFIX_CTRL",
    "FIXED_CONTROLLABLES",
    "env_tool_output_controllable",
    "env_inject_controllable",
    # observables
    "STATIC_OBSERVABLE_SPECS",
    "agent_message_observable",
    "native_tool_observable",
    # specs
    "CONFIG_SPECS",
    "QUERY_SPECS",
    # text-only domains
    "TEXT_ONLY_DOMAINS",
    "EXCLUDED_DOMAINS",
    "is_text_only_domain",
    "require_text_only_domain",
]
