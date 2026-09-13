"""Attacker-controlled injection surfaces for the LangChain agent target.

A superred agent target must expose the agent's real attack surfaces, not just
the user prompt. Besides ``user_input`` (a direct message), an agent is attackable
through:

- **tool returns** — content a tool hands back that the agent then reads and acts
  on (indirect prompt injection: the classic agent vector), and
- **its system prompt** — instructions an attacker manages to plant in the
  agent's own system message.

``create_agent`` bakes tools and the system prompt in at build time, so the target
builds the agent *per run* and hands the factory an :class:`InjectionSpec` carrying
the current run's injected values: the factory appends the system-prompt text and
wires :meth:`InjectionSpec.middleware` (a ``wrap_tool_call`` middleware that
appends the tool-return payload to every tool result). An optimizer's surface
classifier then chooses which of these named surfaces to drive; left un-injected,
the spec is empty and the agent runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


class _ToolOutputInjection(AgentMiddleware):
    """Middleware appending attacker content to every tool's return value.

    Models the indirect-injection surface: a tool hands back attacker-controlled
    text (e.g. a web page / file / API response) that the agent then reads.
    """

    def __init__(self, appendix: str) -> None:
        super().__init__()
        self._appendix = appendix

    def _inject(self, result: Any) -> Any:
        if isinstance(result, ToolMessage) and isinstance(result.content, str):
            result.content = f"{result.content}\n\n{self._appendix}"
        return result

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._inject(handler(request))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        return self._inject(await handler(request))


@dataclass(frozen=True)
class InjectionSpec:
    """The injected values for one run, handed to the agent factory.

    Attributes:
        system_prompt_suffix: attacker text appended to the agent's system prompt.
        tool_output_appendix: attacker text appended to every tool's return.
    """

    system_prompt_suffix: str = ""
    tool_output_appendix: str = ""

    def apply_system_prompt(self, base: str | None) -> str | None:
        """Return ``base`` with the injected system-prompt suffix appended."""
        if not self.system_prompt_suffix:
            return base
        return f"{base}\n\n{self.system_prompt_suffix}" if base else self.system_prompt_suffix

    def middleware(self) -> list[AgentMiddleware]:
        """Middleware wiring the tool-return injection (empty if none injected)."""
        if not self.tool_output_appendix:
            return []
        return [_ToolOutputInjection(self.tool_output_appendix)]


__all__ = ["InjectionSpec"]
