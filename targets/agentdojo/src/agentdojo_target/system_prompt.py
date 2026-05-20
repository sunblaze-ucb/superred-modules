"""Default agent system prompt.

Sourced from AgentDojo's :func:`agentdojo.agent_pipeline.agent_pipeline.load_system_message`,
which reads ``agentdojo/data/system_messages.yaml`` and returns the
``default`` key when no specific name is requested.  The paper's
Figure 14/15/16 (model-specific system prompts) are NOT in the upstream
repo and are out of v1 scope; documented in ``ASSUMPTIONS.md`` §E.2.
"""

from __future__ import annotations

from agentdojo.agent_pipeline.agent_pipeline import load_system_message


def default_system_prompt() -> str:
    """Return the AgentDojo default system prompt verbatim.

    Reads the upstream YAML fresh each call (cheap; lru_cached upstream).
    """
    return load_system_message(None)


__all__ = ["default_system_prompt"]
