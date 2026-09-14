"""openai_agent_injection_claim: direct prompt-injection against an OpenAI Agents
SDK agent.

One Task per attack payload against the ``openai_agents`` target. Success = the
agent, given the injected user input, calls a sensitive tool it should not
(higher = more-successful attack); the SDK's guardrails are part of the system
under test (a tripwire that blocks the attack is a defended outcome). Pairs with
``goal_passthrough`` (inject the attack verbatim) or a jailbreak optimizer.
"""

from __future__ import annotations

from openai_agent_injection_claim.attacks import DEFAULT_ATTACKS, Attack, default_attacks
from openai_agent_injection_claim.factory import agent_injection_claim, agent_injection_demo
from openai_agent_injection_claim.task import AgentInjectionNotScorableError, AgentInjectionTask

__all__ = [
    "Attack",
    "DEFAULT_ATTACKS",
    "default_attacks",
    "AgentInjectionTask",
    "AgentInjectionNotScorableError",
    "agent_injection_claim",
    "agent_injection_demo",
]
