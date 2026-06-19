"""SecurityDomain forest for the ASB target: real trust boundaries.

Agent Security Bench (ASB) attacks a tool-calling agent at four injection
points. We model the target's attack surface as a forest of **trust
boundaries**, the loci an attacker could separately compromise, with hierarchy
where a boundary genuinely contains separable sub-boundaries (parent grants its
children). There is NO data-provenance split and NO separate read/write tags:
read-only access to any tag is granted natively by the Controller's read-only
scope, so a tag means "can tamper here".

Four root boundaries:

- ``user`` -- the live user-instruction channel. Direct Prompt Injection
  (DPI) tampers with the benign task. Scope: ``{user}``.
- ``system`` -- the developer/system scaffolding the agent runs under.
  ``system_prompt`` is the Plan-of-Thought (PoT) backdoor surface; the
  plan-format scaffolding itself is the target's own and always present.
  ``agent_trace`` is the read-only observability subtree for the agent's own
  generations. ``model_identity`` is read-only knowledge of which model powers
  the agent. ``tool_catalogue`` is the tool REGISTRY capability (which tools
  exist + their docs), a grouping root over ``tool_catalogue_add`` /
  ``tool_catalogue_edit`` / ``tool_catalogue_remove``; it is deliberately
  SEPARATE from the ``tools`` tree (what a tool RETURNS) so reading the
  catalogue listing does not imply reading every tool's observation. Scope for
  PoT: ``{system_prompt}``; for catalogue edits: ``{tool_catalogue}`` or a
  single add/edit/remove child.
- ``tools`` -- the tool ecosystem. Observation Prompt Injection (OPI) tampers
  with what a tool returns. ASB's tools belong to ten operational scenarios,
  and each tool is its own backing system, so the tree is three levels:
  ``tools`` (the whole ecosystem) -> ``tools.<scenario>`` (one scenario's tool
  environment, e.g. the financial-analyst suite) -> ``tools.<scenario>.<tool>``
  (a single tool/service). An attacker can be scoped to one tool, one
  scenario's tools, or the whole ecosystem; the upstream-faithful,
  provenance-blind OPI scope is the ``{tools}`` root (it subsumes every
  scenario and tool). Scope for OPI: ``{tools}`` or any ``tools.*`` node.
- ``memory`` -- the durable agent memory store (a separately compromisable
  component). Memory Poisoning (MP) tampers at the memory-read point.
  Scope: ``{memory}``.

Method -> scope:
    DPI -> {user}
    PoT -> {system_prompt}
    OPI -> {tools}  (or a scenario / single-tool sub-node)
    MP  -> {memory}

The scenario and tool nodes are built mechanically from
:data:`TOOLS_BY_SCENARIO` (each tool maps to its scenario by the dataset's
``Corresponding Agent``; no hand-authored grouping). This forest is fully
per-tool granular, so the *run path* (which uses ``scope_includes``, cheap on
any size) is unaffected, but the framework's ``distinct_combinations``
antichain enumeration -- a standalone research utility, never called on the run
path -- is impractical on the full forest (millions of antichains, the
inherent cost of granularity); enumerate antichains on a chosen subtree, or
build scopes directly.
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

# ---------------------------------------------------------------------------
# Tree 1: user  (the live user-input channel) -- DPI
# ---------------------------------------------------------------------------

USER_TAG: SecurityDomainTag = SecurityDomainTag("user")
"""Root of the user tree. An attacker with ``user`` can tamper with the
benign user instruction the Task set (the DPI surface)."""

# ---------------------------------------------------------------------------
# Tree 2: system  (developer / system scaffolding the agent runs under)
# ---------------------------------------------------------------------------

SYSTEM_TAG: SecurityDomainTag = SecurityDomainTag("system")
"""Root of the system tree. An attacker with ``system`` holds every
system-side capability below it."""

SYSTEM_PROMPT_TAG: SecurityDomainTag = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
"""The PoT-backdoor surface. The plan-format scaffolding is the target's own
and always present regardless of any injection here."""

AGENT_TRACE_TAG: SecurityDomainTag = SecurityDomainTag("agent_trace", parent=SYSTEM_TAG)
"""Read-only observability of the agent's OWN model generations only: the
planning workflow JSON and each per-step [Thinking] output. A tool interaction
(the call, its parameters, and the returned observation) is the tool's own data
and is carried under that tool's boundary in the ``tools`` tree -- the tool's
event carries the call + return, and an attacker can change the return there.
Neither the tool call nor the tool response is recorded here."""

MODEL_IDENTITY_TAG: SecurityDomainTag = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
"""Knowledge of which model powers the agent. Read-only observability (never a
controllable: the model is a construction concern). Sibling of
system_prompt/agent_trace so "the attacker knows the victim model" can be
granted or denied independently of any write capability, matching the agentdojo
and inspect-agent targets."""

DETAILED_SYSTEM_SPECIFICATION_TAG: SecurityDomainTag = SecurityDomainTag(
    "detailed_system_specification", parent=SYSTEM_TAG
)
"""Read access to a leaked, free-text specification of the target system: its
purpose, architecture, runtime logic, and hardcoded prompts/code. Pure
intelligence about how the system works (an attacker who obtained internal
documentation), carried by the ``detailed_system_specification`` static
observable. Read-only, never a write surface; a sibling of ``model_identity`` so
"the attacker has a design leak of the system" can be granted independently of
any injection capability."""

# --- tool catalogue (the tool REGISTRY: which tools exist + their docs) -------
# Distinct from the ``tools`` tree (what a tool RETURNS, the OPI surface): the
# registry is a system-side capability, so it lives under ``system``. Reading
# the catalogue listing (which tools exist) must NOT imply reading every tool's
# returned observation -- those live under the separate ``tools`` root -- so the
# listing is tagged here, not at ``tools``.

TOOL_CATALOGUE_TAG: SecurityDomainTag = SecurityDomainTag("tool_catalogue", parent=SYSTEM_TAG)
"""Grouping root of the tool-registry capability. The catalogue-listing
observable carries this tag, so granting it reveals which tools exist without
granting read of any tool's returned observations (the separate ``tools``
tree). Holding ``tool_catalogue`` subsumes the three edit capabilities below."""

TOOL_CATALOGUE_ADD_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_add", parent=TOOL_CATALOGUE_TAG
)
"""Add-a-tool capability: register a NEW attacker-defined tool into the agent's
catalogue. When the agent calls an added tool the attacker supplies its return
directly (an added tool has no genuine implementation), via a per-call event
carried under this tag."""

TOOL_CATALOGUE_EDIT_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_edit", parent=TOOL_CATALOGUE_TAG
)
"""Edit-an-existing-tool capability: change a tool's DESCRIPTION (what the model
reads when choosing tools) and/or its BEHAVIOR. A behavior edit shadows the
tool: when it is called the attacker supplies the return directly and the
original tool is NOT run. The per-call shadow event is carried under this tag,
so holding it grants control of an edited tool's call even when that tool's own
``tools.*`` leaf is out of scope."""

TOOL_CATALOGUE_REMOVE_TAG: SecurityDomainTag = SecurityDomainTag(
    "tool_catalogue_remove", parent=TOOL_CATALOGUE_TAG
)
"""Remove-a-tool capability: drop an existing tool from the catalogue so the
agent can no longer select it."""

# ---------------------------------------------------------------------------
# Tree 3: tools  (the tool ecosystem) -- OPI, three levels: tools > scenario > tool
# ---------------------------------------------------------------------------

TOOLS_TAG: SecurityDomainTag = SecurityDomainTag("tools")
"""Root of the tool-observation tree. Holding ``tools`` grants OPI on every
tool's observation -- the upstream-faithful, provenance-blind OPI scope."""

#: ASB's ten scenarios and their two tools each (mechanical, from the dataset's
#: ``Corresponding Agent`` field). The scenario key is the agent id without the
#: ``_agent`` suffix.
TOOLS_BY_SCENARIO: dict[str, tuple[str, ...]] = {
    "system_admin": ("sys_monitor", "update_manager"),
    "financial_analyst": ("market_data_api", "portfolio_manager"),
    "legal_consultant": ("legal_doc_review", "compliance_checker"),
    "medical_advisor": ("medical_database", "prescription_manager"),
    "education_consultant": ("course_selector", "performance_evaluator"),
    "psychological_counselor": ("therapy_guide", "mental_health_tracker"),
    "ecommerce_manager": ("inventory_manager", "sales_analytics"),
    "aerospace_engineer": ("flight_simulator", "system_analyzer"),
    "academic_search": ("research_database", "summarizer"),
    "autonomous_driving": ("path_planner", "sensor_fusion"),
}

#: All 20 normal tool names (dataset order), for reference/tests.
NORMAL_TOOL_NAMES: tuple[str, ...] = tuple(
    tool for tools in TOOLS_BY_SCENARIO.values() for tool in tools
)

#: Scenario nodes (mid level), children of :data:`TOOLS_TAG`. Keyed by scenario.
SCENARIO_TOOL_TAGS: dict[str, SecurityDomainTag] = {
    scenario: SecurityDomainTag(f"tools.{scenario}", parent=TOOLS_TAG)
    for scenario in TOOLS_BY_SCENARIO
}

#: Tool leaves (deepest level), each a child of its scenario node. Keyed by
#: tool name; this is the OPI tag a specific tool's observation fires under.
TOOL_OBSERVATION_TAGS: dict[str, SecurityDomainTag] = {
    tool: SecurityDomainTag(f"tools.{scenario}.{tool}", parent=SCENARIO_TOOL_TAGS[scenario])
    for scenario, tools in TOOLS_BY_SCENARIO.items()
    for tool in tools
}

# ---------------------------------------------------------------------------
# Tree 4: memory  (durable agent memory store) -- MP
# ---------------------------------------------------------------------------

MEMORY_TAG: SecurityDomainTag = SecurityDomainTag("memory")
"""Memory-poisoning surface. The durable store holds the agent's own past
task/workflow records, retrieved (top-1) during planning as trusted prior
reasoning. Default = genuine retrieval; the Controllable exposes tampering at
the read point. Its own root trust boundary."""

# ---------------------------------------------------------------------------
# Assembled SecurityDomain
# ---------------------------------------------------------------------------

DOMAIN: SecurityDomain = SecurityDomain(
    [
        # user tree
        USER_TAG,
        # system tree
        SYSTEM_TAG,
        SYSTEM_PROMPT_TAG,
        AGENT_TRACE_TAG,
        MODEL_IDENTITY_TAG,
        DETAILED_SYSTEM_SPECIFICATION_TAG,
        TOOL_CATALOGUE_TAG,
        TOOL_CATALOGUE_ADD_TAG,
        TOOL_CATALOGUE_EDIT_TAG,
        TOOL_CATALOGUE_REMOVE_TAG,
        # tools tree (root -> scenario nodes -> tool leaves; a tool's whole
        # interaction (call + return) is the tool's own data, so it lives here,
        # not under agent_trace)
        TOOLS_TAG,
        *SCENARIO_TOOL_TAGS.values(),
        *TOOL_OBSERVATION_TAGS.values(),
        # memory tree
        MEMORY_TAG,
    ]
)
"""The full trust-boundary forest exposed by :class:`AsbTarget`: four roots
(user, system, tools, memory). The system tree carries system_prompt,
agent_trace, model_identity, detailed_system_specification, and the
tool_catalogue registry capability (tool_catalogue -> add/edit/remove). The
tools tree is per-tool granular (10 scenario nodes + 20 tool leaves)."""


__all__ = [
    "USER_TAG",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "AGENT_TRACE_TAG",
    "MODEL_IDENTITY_TAG",
    "DETAILED_SYSTEM_SPECIFICATION_TAG",
    "TOOL_CATALOGUE_TAG",
    "TOOL_CATALOGUE_ADD_TAG",
    "TOOL_CATALOGUE_EDIT_TAG",
    "TOOL_CATALOGUE_REMOVE_TAG",
    "TOOLS_TAG",
    "TOOLS_BY_SCENARIO",
    "NORMAL_TOOL_NAMES",
    "SCENARIO_TOOL_TAGS",
    "TOOL_OBSERVATION_TAGS",
    "MEMORY_TAG",
    "DOMAIN",
]
