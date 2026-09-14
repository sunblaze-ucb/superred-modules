"""The SafeClawArena trust-domain forest, mapped to a superred SecurityDomain.

SafeClawArena evaluates self-hosted agents against five classical cybersecurity
principles (I1 process isolation, I2 least privilege, I3 persistent-state
protection, I4 cross-boundary mediation, I5 data-instruction separation) grouped
into four dimensions (SSI / PSE / CDF / IPI). The forest follows the superred
security-domain guide: one leaf per separately compromisable store, with reading
a store and the agent's actions on it sharing that store's leaf (principle 6);
independent channels as independent roots (principle 5); and knowledge kept
apart from control (principle 4).

Forest (roots -> children):

- ``system``: the agent platform itself.
  - ``supply_chain``: code the agent loads (SSI; I1/I2).
    - ``skills``: the skills a task installs.
    - ``plugins``: the plugins a task installs.
  - ``persistent_state``: state that outlives a session (PSE; I3).
    - ``memory``: ``MEMORY.md`` and ``memory/*.md``.
    - ``config``: ``openclaw.json`` and the auth profiles.
    - ``cron``: scheduled jobs.
  - ``response``: the agent's replies.
  - ``model_identity``: which model and platform are under test.
  - ``task_metadata``: the benchmark task's id, dimension and title.
- ``user``: the user's instruction. SafeClawArena assumes an honest user, so
  none of the benchmark's own attacks arrive this way.
- ``tools``: every store the agent reads from and acts on (IPI inbound, CDF
  outbound; I4/I5). Nothing is emitted at this grouping root.
  - ``workspace_files``: the agent's workspace.
  - ``sim_google``: the simulated Google Workspace, with one leaf per service
    the tasks use (``sim_google.gmail``, ``sim_google.drive``, ...).
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG = SecurityDomainTag("system")
SUPPLY_CHAIN_TAG = SecurityDomainTag("supply_chain", parent=SYSTEM_TAG)
SKILLS_TAG = SecurityDomainTag("skills", parent=SUPPLY_CHAIN_TAG)
PLUGINS_TAG = SecurityDomainTag("plugins", parent=SUPPLY_CHAIN_TAG)
PERSISTENT_STATE_TAG = SecurityDomainTag("persistent_state", parent=SYSTEM_TAG)
MEMORY_TAG = SecurityDomainTag("memory", parent=PERSISTENT_STATE_TAG)
CONFIG_TAG = SecurityDomainTag("config", parent=PERSISTENT_STATE_TAG)
CRON_TAG = SecurityDomainTag("cron", parent=PERSISTENT_STATE_TAG)
RESPONSE_TAG = SecurityDomainTag("response", parent=SYSTEM_TAG)
MODEL_IDENTITY_TAG = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
TASK_METADATA_TAG = SecurityDomainTag("task_metadata", parent=SYSTEM_TAG)

USER_TAG = SecurityDomainTag("user")

TOOLS_TAG = SecurityDomainTag("tools")
WORKSPACE_FILES_TAG = SecurityDomainTag("workspace_files", parent=TOOLS_TAG)
SIM_GOOGLE_TAG = SecurityDomainTag("sim_google", parent=TOOLS_TAG)

#: One leaf per Sim-Google service that the tasks provision data for or tell the
#: agent to use. The vendored CLI also dispatches forms, people, classroom, admin
#: and appscript, but no task touches them: their calls are reported at the
#: ``sim_google`` node, which keeps ``DOMAIN.distinct_combinations()`` near 3
#: million scopes instead of about 95 million with all sixteen as leaves.
SIM_GOOGLE_SERVICES: tuple[str, ...] = (
    "gmail",
    "drive",
    "docs",
    "sheets",
    "slides",
    "calendar",
    "chat",
    "keep",
    "tasks",
    "contacts",
    "groups",
)

#: Service name -> its store leaf under ``sim_google``.
SIM_GOOGLE_SERVICE_TAGS: dict[str, SecurityDomainTag] = {
    service: SecurityDomainTag(f"sim_google.{service}", parent=SIM_GOOGLE_TAG)
    for service in SIM_GOOGLE_SERVICES
}

# CLI shortcuts that act on a service without naming it (sim-google's top-level
# dispatch: `open <id>` builds a Drive URL, `me` is the account's People profile).
_SIM_GOOGLE_ALIASES: dict[str, str] = {"open": "drive", "me": "people"}


def sim_google_call_tag(subcommand: str) -> SecurityDomainTag:
    """The store leaf one logged sim-google call acts on (``"gmail send"`` -> gmail).

    ``subcommand`` is the field the CLI writes to its call log. A call that
    touches no store with a leaf (``auth ...``, ``config``, ``version``, or one of
    the services no task uses) maps to the ``sim_google`` node itself.
    """
    head = subcommand.split(" ", 1)[0] if subcommand else ""
    head = _SIM_GOOGLE_ALIASES.get(head, head)
    return SIM_GOOGLE_SERVICE_TAGS.get(head, SIM_GOOGLE_TAG)


#: The full forest exposed by :class:`~safeclawarena_target.SafeClawArenaTarget`.
DOMAIN: SecurityDomain = SecurityDomain(
    [
        SYSTEM_TAG,
        SUPPLY_CHAIN_TAG,
        SKILLS_TAG,
        PLUGINS_TAG,
        PERSISTENT_STATE_TAG,
        MEMORY_TAG,
        CONFIG_TAG,
        CRON_TAG,
        RESPONSE_TAG,
        MODEL_IDENTITY_TAG,
        TASK_METADATA_TAG,
        USER_TAG,
        TOOLS_TAG,
        WORKSPACE_FILES_TAG,
        SIM_GOOGLE_TAG,
        *SIM_GOOGLE_SERVICE_TAGS.values(),
    ]
)

#: Dimension -> the node where the dimension's attack surface lives: the code the
#: agent loads for SSI, its persistent state for PSE, the tool stores for CDF and
#: IPI. The claim repeats ``attack_succeeded`` at this tag as
#: ``attack_succeeded_in_domain``, visible only to a scope that reaches that surface.
DIMENSION_TAG: dict[str, SecurityDomainTag] = {
    "SSI": SUPPLY_CHAIN_TAG,
    "PSE": PERSISTENT_STATE_TAG,
    "CDF": TOOLS_TAG,
    "IPI": TOOLS_TAG,
}

#: Taint-leak channel (upstream ``unauthorized_channels``) -> the node that
#: observes it. The claim tags each leak sub-score here, so it is visible exactly
#: to a scope that can see where the data went. A leak through
#: ``sim_google_calls`` is refined to the service whose call sent the marker with
#: :func:`sim_google_call_tag`. ``log_file`` sits at the ``system`` root because
#: the gateway log and the session transcript are the platform's own records,
#: which no single surface reaches. ``mcp_tool_input``, ``webhook_payload`` and
#: ``cron_output`` are absent because upstream never detects a leak through them.
LEAK_CHANNEL_TAG: dict[str, SecurityDomainTag] = {
    "agent_response": RESPONSE_TAG,
    "outbound_message": RESPONSE_TAG,
    "sim_google_calls": SIM_GOOGLE_TAG,
    "memory_write": MEMORY_TAG,
    "config_write": CONFIG_TAG,
    "file_write": WORKSPACE_FILES_TAG,
    "log_file": SYSTEM_TAG,
}

__all__ = [
    "SYSTEM_TAG",
    "SUPPLY_CHAIN_TAG",
    "SKILLS_TAG",
    "PLUGINS_TAG",
    "PERSISTENT_STATE_TAG",
    "MEMORY_TAG",
    "CONFIG_TAG",
    "CRON_TAG",
    "RESPONSE_TAG",
    "MODEL_IDENTITY_TAG",
    "TASK_METADATA_TAG",
    "USER_TAG",
    "TOOLS_TAG",
    "WORKSPACE_FILES_TAG",
    "SIM_GOOGLE_TAG",
    "SIM_GOOGLE_SERVICES",
    "SIM_GOOGLE_SERVICE_TAGS",
    "DOMAIN",
    "DIMENSION_TAG",
    "LEAK_CHANNEL_TAG",
    "sim_google_call_tag",
]
