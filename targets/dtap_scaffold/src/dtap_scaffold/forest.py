"""SecurityDomain forest for the DTAP agent targets (shared by both agents).

Independent root trees in the AgentDojo / inspect_agent style: the parent/child
hierarchy encodes capability subsumption (a scope holding a parent tag includes
all descendants), and read-only access is granted by listing a tag in the
Controller's ``read_only`` set rather than via separate read tags.

IMPORTANT (identity semantics): ``SecurityDomainTag.includes`` walks the parent
chain comparing by ``is`` (identity), not equality. The fixed tags below are
module-level singletons. The per-server ``tools.<server>`` / ``environment.<server>``
leaves are dynamic, but the builders below are cached per name, so every
caller gets the ONE instance for a given server -- the same object reused in
:func:`build_domain`, in the controllables the target exposes, and in the events
it fires. (Do not reconstruct a leaf via ``SecurityDomainTag(...)`` directly;
that equal-but-not-identical instance would silently fail scope checks.) The
``tools.<server>`` leaf is itself the ROOT of a per-server authorization subtree
(``tools.<server>.<node>`` tags built from ``data/tool_trees.json`` -- see
:mod:`dtap_scaffold.tool_trees`); each subtree tag is likewise cached.

Roots:
- ``system`` -- agent-side surfaces:
    - ``system_prompt``        the agent system message (DTAP: Agent.system_prompt)
    - ``skill``                a SKILL.md the agent loads (DTAP skill vector)
    - ``tool_catalogue`` {``_add``, ``_edit``, ``_remove``}  the MCP tool registry
                               (DTAP tool vector = edit: suffix/override a description)
    - ``model_identity``       which model powers the agent
    - ``detailed_system_specification``  a leaked design brief of the target
    - ``max_turns``            the run's loop bound (observation only)
    - ``agent_trace`` {``agent_trace_messages`` (non-tool chat),
                       ``agent_trace_tool_calls`` (the agent's NATIVE bash/edit calls)}
- ``user``        -- the user-prompt channel (DTAP prompt vector / direct threat model)
- ``tools``       -- per-MCP-server returned-content surface; tampering a tool's
                     return is indirect injection (one ``tools.<server>`` subtree
                     ROOT per active env server, over a single-parent tree of
                     ``tools.<server>.<node>`` authorization tags)
- ``environment`` -- per-injection-server backend-write surface; the faithful DTAP
                     environment vector writes attacker data into the live backend
                     the agent later reads (one ``environment.<server>`` leaf per
                     active ``<domain>-injection`` server)
- ``host``        -- the target MACHINE the agent runs on (its container filesystem
                     + shell), a separately-compromisable locus distinct from the
                     MCP-backed ``environment`` services:
    - ``host_filesystem``      attacker write access to the machine's filesystem
                               BEFORE the run (place/add/edit/delete files the agent
                               later reads): a planted document, a poisoned config, a
                               malicious script on disk. A mount/supply-chain foothold
                               that shapes what is on disk without executing code.
    - ``host_code_execution``  attacker arbitrary code execution on the machine, run
                               REPEATEDLY as an interactive foothold. Subsumes
                               filesystem control in capability but is scoped
                               separately so an experiment can grant a disk-only
                               foothold without full RCE.

``tools`` and ``environment`` carry no children on their own; the active
environment set (chosen by the Task) supplies the per-server leaves.

Note the asymmetry with ``agent_trace_tool_calls``: that surface OBSERVES the
native tool calls the agent ITSELF makes; the ``host`` tree is the ATTACKER's own
write/execute access to the same machine (a Controllable, not an observation).
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

# ===========================================================================
# Tree 1: system (agent-side surfaces)
# ===========================================================================

SYSTEM_TAG: SecurityDomainTag = SecurityDomainTag("system")
SYSTEM_PROMPT_TAG: SecurityDomainTag = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
SKILL_TAG: SecurityDomainTag = SecurityDomainTag("skill", parent=SYSTEM_TAG)

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag("tool_catalogue", parent=SYSTEM_TAG)
TOOL_CATALOGUE_ADD_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_add", parent=TOOL_CATALOGUE_TAG
)
TOOL_CATALOGUE_EDIT_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_edit", parent=TOOL_CATALOGUE_TAG
)
TOOL_CATALOGUE_REMOVE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_remove", parent=TOOL_CATALOGUE_TAG
)

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
DETAILED_SYSTEM_SPECIFICATION_TAG: SecurityDomainTag = SecurityDomainTag(
    "detailed_system_specification", parent=SYSTEM_TAG
)
MAX_TURNS_TAG: SecurityDomainTag = SecurityDomainTag("max_turns", parent=SYSTEM_TAG)

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)
AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages", parent=AGENT_TRACE_TAG
)
AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls", parent=AGENT_TRACE_TAG
)
"""The agent's NATIVE (in-container bash/edit/exec/fs) tool calls. Observable
only by default; these are the agent's own execution substrate, not one of
DTAP's external injection vectors."""

# ===========================================================================
# Tree 2: user (the user-prompt channel)
# ===========================================================================

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")

# ===========================================================================
# Tree 3 / 4: tools (returned content) and environment (backend writes)
# ===========================================================================

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Root of the per-MCP-server returned-content surface. Each active env server
contributes a ``tools.<server>`` subtree ROOT, under which a single-parent
authorization tree of ``tools.<server>.<node>`` tags places each tool by the
trust boundary an attacker must compromise to control its return (see
:mod:`dtap_scaffold.tool_trees`). Supplied dynamically from the active env set."""

ENVIRONMENT_TAG: SecurityDomainTag = SecurityDomainTag("environment")
"""Root of the per-injection-server backend-write surface (the DTAP environment
vector). Children (one ``environment.<server>`` leaf per active injection
server) are supplied dynamically."""

# ===========================================================================
# Tree 5: host (the target machine the agent runs on)
# ===========================================================================

HOST_TAG: SecurityDomainTag = SecurityDomainTag("host")
"""Root of the target-machine trust boundary: the container the agent executes in
(its filesystem + shell), distinct from the MCP-backed ``environment`` services.
A separately-compromisable locus (a poisoned image, a shared mount, a
supply-chain artifact on disk, or full remote code execution). Holding ``host``
subsumes both capabilities below."""

HOST_FILESYSTEM_TAG: SecurityDomainTag = SecurityDomainTag("host_filesystem", parent=HOST_TAG)
"""Attacker write access to the target machine's filesystem BEFORE the run:
place / add / edit / delete files the agent later reads. Weaker than code
execution: a mounted-volume or supply-chain foothold that shapes what is on disk
without running code."""

HOST_CODE_EXECUTION_TAG: SecurityDomainTag = SecurityDomainTag(
    "host_code_execution", parent=HOST_TAG
)
"""Attacker arbitrary code execution on the target machine, run repeatedly as an
interactive foothold (each result feeds the next decision) until the attacker
stops. Subsumes filesystem control in capability but is scoped separately so an
experiment can grant a disk-only foothold without full RCE."""

# ===========================================================================
# Assembled forest
# ===========================================================================

FIXED_TAGS: tuple[SecurityDomainTag, ...] = (
    SYSTEM_TAG,
    SYSTEM_PROMPT_TAG,
    SKILL_TAG,
    TOOL_CATALOGUE_TAG,
    TOOL_CATALOGUE_ADD_TAG,
    TOOL_CATALOGUE_EDIT_TAG,
    TOOL_CATALOGUE_REMOVE_TAG,
    MODEL_IDENTITY_TAG,
    DETAILED_SYSTEM_SPECIFICATION_TAG,
    MAX_TURNS_TAG,
    AGENT_TRACE_TAG,
    AGENT_TRACE_MESSAGES_TAG,
    AGENT_TRACE_TOOL_CALLS_TAG,
    USER_TAG,
    TOOLS_TAG,
    ENVIRONMENT_TAG,
    HOST_TAG,
    HOST_FILESYSTEM_TAG,
    HOST_CODE_EXECUTION_TAG,
)
"""Every tag the targets always expose, independent of the active env set."""

_DYNAMIC_ROOTS: tuple[SecurityDomainTag, ...] = (TOOLS_TAG, ENVIRONMENT_TAG)


@cache
def tools_server_tag(server: str) -> SecurityDomainTag:
    """The ``tools.<server>`` subtree-root tag for *server* (cached per name).

    Cached so every caller shares the one instance (identity semantics). This is
    the ROOT of the server's authorization subtree; the per-node
    ``tools.<server>.<node>`` tags are built by :mod:`dtap_scaffold.tool_trees`.
    """
    return SecurityDomainTag(f"tools.{server}", parent=TOOLS_TAG)


@cache
def env_server_tag(server: str) -> SecurityDomainTag:
    """The ``environment.<server>`` leaf for an injection *server* (cached per name).

    Cached so every caller shares the one instance (identity semantics).
    """
    return SecurityDomainTag(f"environment.{server}", parent=ENVIRONMENT_TAG)


def _closure(tags: Iterable[SecurityDomainTag]) -> list[SecurityDomainTag]:
    """Collect *tags* plus every ancestor up to (excluding) a dynamic root.

    The dynamic roots (``tools``, ``environment``) are already in
    :data:`FIXED_TAGS`; this returns the leaves and any intermediate parents so a
    valid :class:`SecurityDomain` (which requires every parent to be present) can
    be assembled. Deduped by name.
    """
    seen: dict[str, SecurityDomainTag] = {}
    for tag in tags:
        cur: SecurityDomainTag | None = tag
        while cur is not None and cur not in _DYNAMIC_ROOTS:
            seen[cur.name] = cur
            cur = cur.parent
    return list(seen.values())


def build_domain(
    tool_server_tags: Iterable[SecurityDomainTag] = (),
    env_server_tags: Iterable[SecurityDomainTag] = (),
) -> SecurityDomain:
    """Assemble the forest: the fixed trees plus the active env set's leaves.

    *tool_server_tags* / *env_server_tags* are the (cached) per-server leaf tags
    the target parents under ``tools`` / ``environment`` for the active
    environments. Passing the leaves is enough; ancestors are added
    automatically.
    """
    return SecurityDomain([*FIXED_TAGS, *_closure([*tool_server_tags, *env_server_tags])])


DOMAIN: SecurityDomain = build_domain()
"""The default forest with no active env servers (fixed trees + bare ``tools`` /
``environment`` roots). A configured target exposes a richer domain via
:func:`build_domain` with its cached per-server leaves."""


__all__ = [
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
    "HOST_TAG",
    "HOST_FILESYSTEM_TAG",
    "HOST_CODE_EXECUTION_TAG",
    "FIXED_TAGS",
    "tools_server_tag",
    "env_server_tag",
    "build_domain",
    "DOMAIN",
]
