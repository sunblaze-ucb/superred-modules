"""Pre-run ConfigSpec slots the Task sets on a DTAP agent target.

ConfigSpec is set by the Task, never the attacker, so each slot's tag is
informational (which trust boundary the configured fact concerns). All values
are strings (JSON-encoded where structured), per ``Target.set_config(name,
value: str)``. The model identity and generation settings are construction
concerns (constructor args), NOT config slots.
"""

from __future__ import annotations

from superred.core.types.state import ConfigSpec

from dtap_scaffold.forest import (
    ATTACKER_CONTEXT_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    ENVIRONMENT_TAG,
    MAX_TURNS_TAG,
    SYSTEM_PROMPT_TAG,
    SYSTEM_TAG,
    TOOL_CATALOGUE_TAG,
    USER_TAG,
)

# Slot-name constants (referenced by the target's set_config dispatch and the claim).
ACTIVE_MCP_SERVERS = "active_mcp_servers"
ENV_INJECTION_CONFIG = "env_injection_config"
ENV_INJECTION_TEMPLATES = "env_injection_templates"
SYSTEM_PROMPT = "system_prompt"
USER_PROMPT = "user_prompt"
TASK_DIR = "task_dir"
AVAILABLE_INJECTIONS = "available_injections"
ADDITIONAL_INFORMATION = "additional_information"
SERVER_ENV_OVERRIDES = "server_env_overrides"
THREAT_MODEL = "threat_model"
MAX_TURNS = "max_turns"
NATIVE_TOOLS_POLICY = "native_tools_policy"

CONFIG_SPECS: tuple[ConfigSpec, ...] = (
    ConfigSpec(
        name=ACTIVE_MCP_SERVERS,
        security_domain=TOOL_CATALOGUE_TAG,
        description=(
            "JSON list[str] of MCP server names to activate for this task "
            "(text-only domains only). Drives which env + injection containers "
            "start and which tools the agent gets."
        ),
    ),
    ConfigSpec(
        name=ENV_INJECTION_CONFIG,
        security_domain=ENVIRONMENT_TAG,
        description=(
            "JSON of DTAP RedTeamingAgent.env_injection_config (e.g. "
            '{"gmail-injection": "all"}): which injection MCP servers start and '
            "which inject_* points become env_inject controllables."
        ),
    ),
    ConfigSpec(
        name=ENV_INJECTION_TEMPLATES,
        security_domain=ENVIRONMENT_TAG,
        description=(
            "JSON {server: {injection_mcp_tool, kwargs, content_field}} of the task's "
            "reference env injection per server (from Attack.attack_turns environment "
            "steps). TARGET-PRIVATE routing: it is NOT replayed as an attack; it lets "
            "the target splice a plain-text env_inject payload into the pinned content "
            "field. A server present here exposes env_inject as free text; absent -> "
            "structured JSON only. Empty when the task pins no env injection."
        ),
    ),
    ConfigSpec(
        name=SYSTEM_PROMPT,
        security_domain=SYSTEM_PROMPT_TAG,
        description="The Agent.system_prompt baseline (the system_prompt controllable's default).",
    ),
    ConfigSpec(
        name=USER_PROMPT,
        security_domain=USER_TAG,
        description=(
            "The benign Task.task_instruction baseline (JSON list[str] for "
            "multi-turn). The user_prompt controllable's default per turn."
        ),
    ),
    ConfigSpec(
        name=TASK_DIR,
        security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
        description="Absolute path to the DTAP task dir (setup.sh / metadata / judge.py).",
    ),
    ConfigSpec(
        name=AVAILABLE_INJECTIONS,
        security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
        description=(
            "JSON {prompt,tool,environment,skill: bool} from config.yaml. Recorded "
            "as the claim's default scope hint; does NOT remove controllables (the "
            "target always exposes the agent's real surface; the Controller scope "
            "gates per experiment)."
        ),
    ),
    ConfigSpec(
        name=ADDITIONAL_INFORMATION,
        security_domain=ATTACKER_CONTEXT_TAG,
        description=(
            "DTAP Attack.additional_information: per-task attacker prerequisite facts "
            "(victim inbox + whitelisted attacker sender accounts) surfaced as the "
            "attacker_context observable. Empty when the task carries none."
        ),
    ),
    ConfigSpec(
        name=SERVER_ENV_OVERRIDES,
        security_domain=ENVIRONMENT_TAG,
        description=(
            "JSON {server: {VAR: value}} of per-task Agent.mcp_servers env_vars: "
            "upstream's highest-priority env tier, merged LAST into each MCP env "
            "server's process env. Sets the acting-user identity (USER_ACCESS_TOKEN) "
            "and per-task credentials the env-state judge verifies against."
        ),
    ),
    ConfigSpec(
        name=THREAT_MODEL,
        security_domain=DETAILED_SYSTEM_SPECIFICATION_TAG,
        description='DTAP threat model: "direct" or "indirect" (metadata).',
    ),
    ConfigSpec(
        name=MAX_TURNS,
        security_domain=MAX_TURNS_TAG,
        description="Agent loop bound for this run (int as str; default applied if unset).",
    ),
    ConfigSpec(
        name=NATIVE_TOOLS_POLICY,
        security_domain=SYSTEM_TAG,
        description=(
            'Native-tool policy: "enabled" (default; agent gets bash/edit/exec/fs), '
            '"disabled" (upstream-faithful deny-list for golden replay), or a JSON '
            "deny list of native tool names."
        ),
    ),
)
"""The pre-run config slots, in stable order."""


__all__ = [
    "ACTIVE_MCP_SERVERS",
    "ENV_INJECTION_CONFIG",
    "ENV_INJECTION_TEMPLATES",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "TASK_DIR",
    "AVAILABLE_INJECTIONS",
    "ADDITIONAL_INFORMATION",
    "SERVER_ENV_OVERRIDES",
    "THREAT_MODEL",
    "MAX_TURNS",
    "NATIVE_TOOLS_POLICY",
    "CONFIG_SPECS",
]
