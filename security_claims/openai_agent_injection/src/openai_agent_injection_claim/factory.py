"""Factories for the OpenAI-agent direct-injection claim.

``agent_injection_claim(...)`` builds a :class:`SecurityClaim` of one
:class:`~openai_agent_injection_claim.task.AgentInjectionTask` per attack.
``agent_injection_demo(...)`` also builds the paired target factory over the
bundled demo agent, so a full offline red-team (against a scripted or real model)
is one call.
"""

from __future__ import annotations

from collections.abc import Iterable

from openai_agents_target import (
    SENSITIVE_TOOL,
    OpenAIAgentTarget,
    build_demo_agent,
    openai_agent_target_factory,
)
from superred.core.controller import TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from openai_agent_injection_claim.attacks import DEFAULT_ATTACKS, Attack
from openai_agent_injection_claim.task import AgentInjectionTask


def agent_injection_claim(
    *,
    attacks: Iterable[Attack] | None = None,
    sensitive_tool: str = SENSITIVE_TOOL,
    instructions_override: str | None = None,
) -> SecurityClaim[OpenAIAgentTarget]:
    """Build a direct-injection claim: one task per attack payload."""
    chosen = tuple(attacks) if attacks is not None else DEFAULT_ATTACKS
    tasks: list[Task[OpenAIAgentTarget]] = [
        AgentInjectionTask(
            attack=a.render(sensitive_tool),
            attack_id=a.attack_id,
            sensitive_tool=sensitive_tool,
            instructions_override=instructions_override,
        )
        for a in chosen
    ]
    return SecurityClaim.from_tasks(tasks)


def agent_injection_demo(
    *,
    model: object | None = None,
    max_turns: int = 10,
    attacks: Iterable[Attack] | None = None,
) -> tuple[SecurityClaim[OpenAIAgentTarget], TargetFactory]:
    """Return ``(claim, target_factory)`` wired over the bundled demo agent.

    The demo agent has a benign ``get_weather`` tool and a sensitive
    ``exfiltrate_data`` tool; the claim measures whether an injection makes the
    agent call the sensitive one. Pass a scripted model for offline runs, or a
    real model id for a live run.
    """
    factory = openai_agent_target_factory(
        agent_factory=build_demo_agent, model=model, max_turns=max_turns
    )
    claim = agent_injection_claim(attacks=attacks, sensitive_tool=SENSITIVE_TOOL)
    return claim, factory


__all__ = ["agent_injection_claim", "agent_injection_demo"]
