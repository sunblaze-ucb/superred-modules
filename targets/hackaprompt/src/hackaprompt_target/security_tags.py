"""The HackAPrompt trust-boundary forest.

HackAPrompt is direct prompt injection against a defended LLM application: each
level wraps the attacker's ``user_input`` inside a task-specific instruction
template (the defense), and the attack succeeds when the attacker's input
overrides that template and makes the model emit the target phrase. The single
attacker-controlled surface is ``user_input`` (the untrusted input channel); the
level's own instructions are the system/defense channel.
"""

from __future__ import annotations

from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag

SYSTEM_TAG = SecurityDomainTag("system")
USER_INPUT_TAG = SecurityDomainTag("user_input", parent=SYSTEM_TAG)

DOMAIN = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG])

__all__ = ["SYSTEM_TAG", "USER_INPUT_TAG", "DOMAIN"]
