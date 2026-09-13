"""Security-domain forest for the SafeClawBench Exec-Balanced target.

SafeClawBench's tool-using-agent threat model maps onto a small superred
``SecurityDomain`` forest. The agent's own instructions/harness are the trust
root (``system``); everything an attacker can influence hangs beneath it:

- ``user_input``    - the (attacker-controllable) user request / task prompt.
- ``external_data`` - attacker-planted content the agent reads back through
  tools (file bodies, web pages, tool returns) — the IPI / TRI vector.
- ``memory``        - the persistent memory store an attacker may poison (MPI)
  or exfiltrate (MEX).
- ``tools``         - the tool catalogue + harness permissions.
"""

from __future__ import annotations

from typing import Final

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
EXTERNAL_DATA_TAG: Final = SecurityDomainTag("external_data", parent=SYSTEM_TAG)
MEMORY_TAG: Final = SecurityDomainTag("memory", parent=SYSTEM_TAG)
TOOLS_TAG: Final = SecurityDomainTag("tools", parent=SYSTEM_TAG)

DOMAIN: Final = SecurityDomain(
    [SYSTEM_TAG, USER_INPUT_TAG, EXTERNAL_DATA_TAG, MEMORY_TAG, TOOLS_TAG]
)

__all__ = [
    "DOMAIN",
    "EXTERNAL_DATA_TAG",
    "MEMORY_TAG",
    "SYSTEM_TAG",
    "TOOLS_TAG",
    "USER_INPUT_TAG",
]
