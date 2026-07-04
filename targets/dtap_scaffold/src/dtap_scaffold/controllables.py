"""Controllables exposed by the DTAP agent targets (shared by both agents).

These are the attacker-facing injection points. The target ALWAYS fires the
corresponding event; the Controller's ``security_domain_filter`` decides, per the
experiment's scope, whether an optimizer may actually inject. A passthrough
optimizer (declines everything) leaves every value genuine, reproducing DTAP's
unattacked run -- the clean baseline. The byte-identical upstream attack payloads
are NOT defaults here; they live only in the claim's replay-based faithfulness
test.

The four DTAP injection vectors map to these controllables:

- prompt      -> ``user_prompt``                       (USER, PreCall)
- skill       -> ``skill``                             (SKILL, PreCall)
- tool        -> ``tool_description_override`` /       (TOOL_CATALOGUE_EDIT, PreCall;
                 ``tool_description_suffix``             applied in the proxy list_tools)
- environment -> ``env_inject:<server>``               (environment.<server>, PostCall,
                 (one per injection server)             one per injection server; writes
                                                        attacker data to the live backend)

Plus two superred-afforded surfaces DTAP does not itself enumerate as vectors:

- ``system_prompt``                                    (SYSTEM_PROMPT, PreCall; overrides
                                                        the agent's system message)
- ``env_tool:<server>``                                (tools.<server>, PostCall, one per
                                                        active env server; replaces what a
                                                        tool on that server returns)

Native agent tools (bash/edit/exec/fs) run inside the agent container and are
observable-only (see ``observables.AGENT_TRACE_TOOL_CALLS_TAG``); they are not
one of DTAP's external vectors.
"""

from __future__ import annotations

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

from dtap_scaffold.forest import (
    SKILL_TAG,
    SYSTEM_PROMPT_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    USER_TAG,
)

# ---------------------------------------------------------------------------
# Fixed singletons (PreCall: value consumed before / at the start of the run)
# ---------------------------------------------------------------------------

USER_PROMPT_CTRL: Controllable = Controllable(
    name="user_prompt",
    security_domain=USER_TAG,
    description=(
        "DTAP prompt vector. Override the user instruction the Task set (the "
        "direct-threat-model / jailbreak surface). PreCall; request carries the "
        "benign task_instruction, the injected value replaces it for the run."
    ),
    value_type="text",
)

SYSTEM_PROMPT_CTRL: Controllable = Controllable(
    name="system_prompt",
    security_domain=SYSTEM_PROMPT_TAG,
    description="Override the agent's system prompt for this run. PreCall.",
    value_type="text",
)

SKILL_CTRL: Controllable = Controllable(
    name="skill",
    security_domain=SKILL_TAG,
    description=(
        "DTAP skill vector. Create/extend a SKILL.md the agent loads before the "
        'run. JSON value: {"name": str, "content": str, '
        '"mode": "create"|"append"|"insert", "row"?: int}. PreCall.'
    ),
    value_type="json",
)

TOOL_DESCRIPTION_OVERRIDE_CTRL: Controllable = Controllable(
    name="tool_description_override",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "DTAP tool vector (override mode). Replace an existing MCP tool's "
        'description the model sees. JSON value: {"server": str, "tool": str, '
        '"description": str}. PreCall; applied in the proxy list_tools handler.'
    ),
    value_type="json",
)

TOOL_DESCRIPTION_SUFFIX_CTRL: Controllable = Controllable(
    name="tool_description_suffix",
    security_domain=TOOL_CATALOGUE_EDIT_TAG,
    description=(
        "DTAP tool vector (suffix mode). Append attacker text to an existing MCP "
        'tool\'s description. JSON value: {"server": str, "tool": str, '
        '"suffix": str}. PreCall; applied in the proxy list_tools handler.'
    ),
    value_type="json",
)

FIXED_CONTROLLABLES: list[Controllable] = [
    USER_PROMPT_CTRL,
    SYSTEM_PROMPT_CTRL,
    SKILL_CTRL,
    TOOL_DESCRIPTION_OVERRIDE_CTRL,
    TOOL_DESCRIPTION_SUFFIX_CTRL,
]
"""Controllables the target always exposes, in stable order. The per-server
``env_tool:`` and ``env_inject:`` controllables are appended by the target from
the active environment set (built once, with cached tags -- see forest identity
note)."""


# ---------------------------------------------------------------------------
# Per-server builders (the content / indirect-injection surfaces, PostCall)
# ---------------------------------------------------------------------------


def env_tool_output_controllable(server: str, security_domain: SecurityDomainTag) -> Controllable:
    """Per-MCP-SERVER return-tampering controllable (the proxy content surface).

    One per active env server (enumerable pre-run from ``active_mcp_servers``,
    before any container starts). Fired as a ``ControllablePostCallEvent`` each
    time a tool on *server* returns: the event's ``request`` carries
    ``{tool, params}`` and its ``answer`` carries the genuine return; a
    ``ControllableInjection`` replaces what the agent sees. *security_domain* is
    the (cached) ``tools.<server>`` leaf -- the SAME instance must be reused in
    the domain, the controllable, and the event (identity-based scope checks).
    """
    return Controllable(
        name=f"env_tool:{server}",
        security_domain=security_domain,
        description=(
            f"Replace the value any {server} MCP tool returns to the agent "
            "(indirect prompt injection). PostCall, once per tool call; the event's "
            "request carries {tool, params} and its answer the genuine return."
        ),
        value_type="text",
    )


def env_inject_controllable(server: str, security_domain: SecurityDomainTag) -> Controllable:
    """DTAP environment-vector controllable for one injection SERVER.

    One per ``<server>-injection`` server in ``env_injection_config`` (enumerable
    pre-run). Fired as a ``ControllablePostCallEvent`` before the agent reads: the
    event's ``answer`` carries the genuine current backend content; a
    ``ControllableInjection`` value (JSON ``{injection_mcp_tool, kwargs}`` or text)
    is written into the LIVE backend via the injection MCP tool (DTAP
    ``inject_*``), so the agent later reads attacker data through its normal tools
    (or, for FS domains, via native bash on the shared volume). *security_domain*
    is the (cached) ``environment.<server>`` leaf.
    """
    return Controllable(
        name=f"env_inject:{server}",
        security_domain=security_domain,
        description=(
            f"Write attacker data into the live backend via the {server} injection "
            "server; the agent reads it later (DTAP environment vector). PostCall; "
            "the event's answer carries the genuine current content. JSON "
            '{"injection_mcp_tool": str, "kwargs": dict} or text forwarded as the '
            "inject payload."
        ),
        value_type="json",
    )


__all__ = [
    "USER_PROMPT_CTRL",
    "SYSTEM_PROMPT_CTRL",
    "SKILL_CTRL",
    "TOOL_DESCRIPTION_OVERRIDE_CTRL",
    "TOOL_DESCRIPTION_SUFFIX_CTRL",
    "FIXED_CONTROLLABLES",
    "env_tool_output_controllable",
    "env_inject_controllable",
]
