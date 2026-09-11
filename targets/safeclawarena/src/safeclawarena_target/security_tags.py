"""The SafeClawArena trust-domain forest, mapped to a superred SecurityDomain.

SafeClawArena evaluates self-hosted agents against five classical cybersecurity
principles (I1 process isolation, I2 least privilege, I3 persistent-state
protection, I4 cross-boundary mediation, I5 data-instruction separation) grouped
into four dimensions (SSI / PSE / CDF / IPI). Those boundaries are expressed here
as a superred ``SecurityDomain`` forest so an attacker can be granted access
selectively (only the compromised domains).

Forest (roots -> children):

- ``system``
  - ``skills``  — the skill supply chain (SSI; I1/I2): managed + workspace skills.
  - ``memory``  — durable Markdown memory (PSE; I3).
  - ``config``  — agent config / auth profiles (PSE; I3).
- ``user`` — the user's instruction to the agent.
- ``external_data`` — untrusted inbound content the agent ingests (IPI; I5).
  - ``sim_google`` — content arriving through the simulated Google Workspace.
  - ``file_content`` — workspace files the agent reads.
- ``tools`` — the agent's outbound tool/service actions (CDF mediation; I4).
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG = SecurityDomainTag("system")
SKILLS_TAG = SecurityDomainTag("skills", parent=SYSTEM_TAG)
MEMORY_TAG = SecurityDomainTag("memory", parent=SYSTEM_TAG)
CONFIG_TAG = SecurityDomainTag("config", parent=SYSTEM_TAG)

USER_TAG = SecurityDomainTag("user")

EXTERNAL_DATA_TAG = SecurityDomainTag("external_data")
SIM_GOOGLE_TAG = SecurityDomainTag("sim_google", parent=EXTERNAL_DATA_TAG)
FILE_CONTENT_TAG = SecurityDomainTag("file_content", parent=EXTERNAL_DATA_TAG)

TOOLS_TAG = SecurityDomainTag("tools")

#: The full forest exposed by :class:`~safeclawarena_target.SafeClawArenaTarget`.
DOMAIN: SecurityDomain = SecurityDomain(
    [
        SYSTEM_TAG,
        SKILLS_TAG,
        MEMORY_TAG,
        CONFIG_TAG,
        USER_TAG,
        EXTERNAL_DATA_TAG,
        SIM_GOOGLE_TAG,
        FILE_CONTENT_TAG,
        TOOLS_TAG,
    ]
)

#: Dimension -> the trust-domain tag its attacks primarily cross. Used by the
#: claim to tag each task's sub-score to the boundary it exercises.
DIMENSION_TAG = {
    "SSI": SKILLS_TAG,
    "PSE": MEMORY_TAG,
    "CDF": TOOLS_TAG,
    "IPI": EXTERNAL_DATA_TAG,
}

__all__ = [
    "SYSTEM_TAG",
    "SKILLS_TAG",
    "MEMORY_TAG",
    "CONFIG_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "SIM_GOOGLE_TAG",
    "FILE_CONTENT_TAG",
    "TOOLS_TAG",
    "DOMAIN",
    "DIMENSION_TAG",
]
