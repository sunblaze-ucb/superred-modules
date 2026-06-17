"""SecurityDomain forest for the ASB target: real trust boundaries.

Agent Security Bench (ASB) attacks a tool-calling agent at four injection
points. We model the target's attack surface as a forest of **trust
boundaries**, the loci an attacker could separately compromise, with
hierarchy where a boundary genuinely contains separable sub-boundaries
(parent grants its children). There is NO data-provenance split and NO
separate read/write tags: read-only access to any tag is granted natively by
the Controller's read-only scope, so a tag means "can tamper here".

Four root boundaries:

- ``user`` -- the live user-instruction channel. Direct Prompt Injection
  (DPI) tampers with the benign task. Scope: ``{user}``.
- ``system`` -- the developer/system scaffolding the agent runs under.
  ``system_prompt`` is the Plan-of-Thought (PoT) backdoor surface; the
  plan-format scaffolding itself is the target's own and always present.
  ``agent_trace`` is the read-only observability subtree for the agent's own
  generations. Scope for PoT: ``{system_prompt}``.
- ``tools`` -- the tool ecosystem. Observation Prompt Injection (OPI)
  tampers with what a tool returns. ASB's tools belong to ten operational
  scenarios; each scenario's tool environment is a separately compromisable
  system, so the root has one mechanical sub-boundary per scenario
  (``tools.<scenario>``), each grouping that scenario's tools. The
  upstream-faithful, provenance-blind OPI scope is the ``{tools}`` root (it
  subsumes every scenario), and a threat model can restrict an attacker to a
  single scenario's tools. Scope for OPI: ``{tools}`` or a ``tools.<scenario>``.
- ``memory`` -- the durable agent memory store (a separately compromisable
  component). Memory Poisoning (MP) tampers at the memory-read point.
  Scope: ``{memory}``.

Method -> scope:
    DPI -> {user}
    PoT -> {system_prompt}
    OPI -> {tools}  (or a single tools.<scenario>)
    MP  -> {memory}

The scenario sub-boundaries are built mechanically from
:data:`TOOLS_BY_SCENARIO` (each tool maps to its scenario by the dataset's
``Corresponding Agent``, no hand-authored grouping). Per-tool leaves were
deliberately not used: the framework's antichain enumeration
(``distinct_combinations``) is exponential in a node's child count, so 20
sibling tool leaves are infeasible while ten scenario leaves are cheap.
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
"""Read-only observability of the agent's runtime generations. Implies its
three finer-grained children."""

AGENT_TRACE_MESSAGES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_messages",
    parent=AGENT_TRACE_TAG,
)
"""The agent's own model generations: the planning workflow JSON and each
per-step model output ([Thinking] text)."""

AGENT_TRACE_TOOL_CALLS_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_calls",
    parent=AGENT_TRACE_TAG,
)
"""The tool-call decisions the agent executes."""

AGENT_TRACE_TOOL_RESPONSES_TAG: SecurityDomainTag = SecurityDomainTag(
    "agent_trace_tool_responses",
    parent=AGENT_TRACE_TAG,
)
"""The tool return the agent observed on the final step (non-final returns
are on the trajectory via their OPI controllable event)."""

# ---------------------------------------------------------------------------
# Tree 3: tools  (the tool ecosystem) -- OPI, one leaf per scenario
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

#: One OPI leaf per scenario, child of :data:`TOOLS_TAG`. Keyed by scenario.
SCENARIO_TOOL_TAGS: dict[str, SecurityDomainTag] = {
    scenario: SecurityDomainTag(f"tools.{scenario}", parent=TOOLS_TAG)
    for scenario in TOOLS_BY_SCENARIO
}

#: Reverse map: tool name -> its scenario's OPI tag.
TOOL_OBSERVATION_TAGS: dict[str, SecurityDomainTag] = {
    tool: SCENARIO_TOOL_TAGS[scenario]
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
        AGENT_TRACE_MESSAGES_TAG,
        AGENT_TRACE_TOOL_CALLS_TAG,
        AGENT_TRACE_TOOL_RESPONSES_TAG,
        # tools tree (root + one leaf per scenario)
        TOOLS_TAG,
        *SCENARIO_TOOL_TAGS.values(),
        # memory tree
        MEMORY_TAG,
    ]
)
"""The full trust-boundary forest exposed by :class:`AsbTarget`: four roots
(user, system, tools, memory), 17 tags total."""


__all__ = [
    "USER_TAG",
    "SYSTEM_TAG",
    "SYSTEM_PROMPT_TAG",
    "AGENT_TRACE_TAG",
    "AGENT_TRACE_MESSAGES_TAG",
    "AGENT_TRACE_TOOL_CALLS_TAG",
    "AGENT_TRACE_TOOL_RESPONSES_TAG",
    "TOOLS_TAG",
    "TOOLS_BY_SCENARIO",
    "NORMAL_TOOL_NAMES",
    "SCENARIO_TOOL_TAGS",
    "TOOL_OBSERVATION_TAGS",
    "MEMORY_TAG",
    "DOMAIN",
]
