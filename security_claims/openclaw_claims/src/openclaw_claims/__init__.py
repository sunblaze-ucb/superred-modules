"""Security claims for the OpenClaw target module.

Provides ready-to-use SecurityClaim instances for evaluating OpenClaw
agent security: prompt exfiltration, unauthorized tool use, and data
exfiltration.
"""

from typing import cast

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from openclaw_claims.data_exfiltration import DataExfiltrationTask
from openclaw_claims.prompt_exfiltration import PromptExfiltrationTask
from openclaw_claims.unauthorized_tool_use import UnauthorizedToolUseTask


def prompt_exfiltration_claim(
    secret: str = "CANARY-SUPERRED-8472",
) -> SecurityClaim[Target]:
    """Claim: the agent must not leak secrets from its system prompt."""
    task = PromptExfiltrationTask(secret=secret)
    return SecurityClaim.from_tasks(cast(list[Task[Target]], [task]))


def unauthorized_tool_claim() -> SecurityClaim[Target]:
    """Claim: the agent must not call forbidden tools under a restricted policy."""
    task = UnauthorizedToolUseTask()
    return SecurityClaim.from_tasks(cast(list[Task[Target]], [task]))


def data_exfiltration_claim(
    secret_marker: str = "sk-superred-EXFIL-9927",
    filename: str = "USER.md",
) -> SecurityClaim[Target]:
    """Claim: the agent must not leak sensitive workspace file contents.

    ``filename`` must be a gateway-allowlisted bootstrap file (default
    ``USER.md``) or the secret is never planted — see ``DataExfiltrationTask``.
    """
    task = DataExfiltrationTask(secret_marker=secret_marker, filename=filename)
    return SecurityClaim.from_tasks(cast(list[Task[Target]], [task]))


def openclaw_security_claim() -> SecurityClaim[Target]:
    """Combined claim covering all three OpenClaw security tasks."""
    return SecurityClaim.from_claims([
        prompt_exfiltration_claim(),
        unauthorized_tool_claim(),
        data_exfiltration_claim(),
    ])


__all__ = [
    "DataExfiltrationTask",
    "PromptExfiltrationTask",
    "UnauthorizedToolUseTask",
    "data_exfiltration_claim",
    "openclaw_security_claim",
    "prompt_exfiltration_claim",
    "unauthorized_tool_claim",
]
