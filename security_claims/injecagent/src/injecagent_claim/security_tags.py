"""The InjecAgent trust-boundary forest.

InjecAgent's threat model is indirect prompt injection against a tool-integrated
agent: the attacker controls content **returned by a tool** (``external_data``),
and the attack succeeds when the agent, swayed by that content, invokes an
attacker-chosen **tool**. The forest below is the shared vocabulary the paired
target declares as its ``security_domain`` and the claim uses to tag findings, so
they reference the *same* tag objects (the target depends on this claim).
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG = SecurityDomainTag("system")
USER_TAG = SecurityDomainTag("user", parent=SYSTEM_TAG)
EXTERNAL_DATA_TAG = SecurityDomainTag("external_data", parent=SYSTEM_TAG)
TOOLS_TAG = SecurityDomainTag("tools", parent=SYSTEM_TAG)

DOMAIN = SecurityDomain([SYSTEM_TAG, USER_TAG, EXTERNAL_DATA_TAG, TOOLS_TAG])

# Every InjecAgent case is IPI: the crossing exercised is the untrusted
# tool-returned content (external_data) breaching the instruction channel.
DIMENSION_TAG: dict[str, SecurityDomainTag] = {
    "dh": EXTERNAL_DATA_TAG,
    "ds": EXTERNAL_DATA_TAG,
}

__all__ = [
    "SYSTEM_TAG",
    "USER_TAG",
    "EXTERNAL_DATA_TAG",
    "TOOLS_TAG",
    "DOMAIN",
    "DIMENSION_TAG",
]
