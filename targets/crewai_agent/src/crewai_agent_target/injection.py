"""Attacker-controlled injection surfaces for the CrewAI agent target.

A superred agent target must expose the agent's real attack surfaces, not just
the user prompt. Besides ``user_input`` (the direct task input), a CrewAI agent is
attackable through:

- **tool returns** — content a tool hands back that the agent then reads and acts
  on (indirect prompt injection: the classic agent vector), and
- **its system prompt** — instructions an attacker manages to plant in the agent's
  own system message. CrewAI composes that system message from the agent's
  ``role`` / ``goal`` / ``backstory`` (string-substituted into the prompt
  template), so ``backstory`` is the free-text field an attacker's text rides in
  on.

CrewAI bakes the backstory and tools into the ``Agent`` at build time, so the
target builds the crew *per run* and hands the factory an :class:`InjectionSpec`
carrying the current run's injected values: the factory appends the suffix to the
agent's backstory via :meth:`InjectionSpec.apply_backstory` and wraps its tools via
:meth:`InjectionSpec.wrap_tools` so every tool's return value gets the appendix
appended. An optimizer's surface classifier then chooses which named surface to
drive; left un-injected, the spec is empty and the crew runs unchanged.

``crewai`` is imported lazily (only inside :func:`_injected_tool_cls`) so this
module — and hence ``crewai_agent_target.target`` — imports with the framework
absent.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


def _inject_into_result(result: Any, appendix: str) -> Any:
    """Append ``appendix`` to a tool's return value so it reaches the model.

    CrewAI turns a tool's return value into the observation the agent reads by
    running it through ``format_output_for_agent`` and then ``str(...)``: for a tool
    with no ``result_schema`` that is simply ``str(result)``, and for one *with* a
    ``result_schema`` it is ``schema.model_validate(result).model_dump_json()``,
    falling back to ``str(result)`` when validation fails. A string never validates
    as a structured schema, so returning a **string** for every non-string shape
    guarantees the appendix survives into the observation in both cases — and never
    silently drops it (the false negative that would defeat this surface). We
    therefore enumerate the shapes a CrewAI tool can return rather than injecting
    only into ``str`` results:

    - ``str`` — the common case: append the text.
    - ``bytes`` / ``bytearray`` — decode (utf-8, replacing errors) then append.
    - a pydantic ``BaseModel`` (structured output) — render via ``model_dump_json``
      then append, so the payload lands after valid JSON.
    - anything else (``dict``, ``list``, scalars, ``None``, a custom object) — the
      model-visible form is ``str(result)``, so render and append.

    A coroutine (an async tool's un-awaited return) is passed through untouched;
    :class:`_InjectedTool.run` resolves it first and injects into the concrete
    result.
    """
    if not appendix:
        return result
    if isinstance(result, str):
        return f"{result}\n\n{appendix}"
    if asyncio.iscoroutine(result):
        return result
    if isinstance(result, (bytes, bytearray)):
        return f"{result.decode('utf-8', errors='replace')}\n\n{appendix}"
    dump = getattr(result, "model_dump_json", None)
    if callable(dump):
        try:
            return f"{dump()}\n\n{appendix}"
        except Exception:  # noqa: BLE001 - fall back to the str() rendering below
            pass
    return f"{result}\n\n{appendix}"


_INJECTED_TOOL_CLS: Any = None


def _injected_tool_cls() -> Any:
    """Build (once) and return the ``BaseTool`` subclass that wraps a tool.

    Defined lazily so importing this module needs no ``crewai`` install: the class
    can only exist once ``crewai.tools.BaseTool`` is importable.
    """
    global _INJECTED_TOOL_CLS
    if _INJECTED_TOOL_CLS is not None:
        return _INJECTED_TOOL_CLS

    from crewai.tools import BaseTool
    from pydantic import PrivateAttr

    class _InjectedTool(BaseTool):  # type: ignore[misc]  # crewai is untyped (Any base)
        """Wraps a tool so the appendix is appended to whatever it returns.

        Forwards the wrapped tool's ``name`` / ``description`` / ``args_schema`` /
        ``result_as_answer`` so the agent sees an identical tool, and delegates
        execution to the inner tool's own ``run`` (which validates args, enforces
        usage limits, and resolves async tools) before appending the payload — so
        the injection reaches the model-visible observation for every return shape.
        """

        _inner: Any = PrivateAttr()
        _appendix: str = PrivateAttr(default="")

        def __init__(self, inner: Any, appendix: str) -> None:
            super().__init__(
                name=inner.name,
                description=inner.description,
                args_schema=inner.args_schema,
                result_as_answer=inner.result_as_answer,
            )
            self._inner = inner
            self._appendix = appendix

        def run(self, *args: Any, **kwargs: Any) -> Any:
            # CrewAI's tool-execution entry point. inner.run() validates args,
            # enforces usage limits, and resolves async tools to a concrete value;
            # inject into that so the payload reaches the observation.
            return _inject_into_result(self._inner.run(*args, **kwargs), self._appendix)

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            # CrewAI drives tools through run(); this abstract method is implemented
            # for the direct-call path and injects the same way.
            return _inject_into_result(self._inner._run(*args, **kwargs), self._appendix)

    _INJECTED_TOOL_CLS = _InjectedTool
    return _InjectedTool


@dataclass(frozen=True)
class InjectionSpec:
    """The injected values for one run, handed to the crew factory.

    Attributes:
        system_prompt_suffix: attacker text appended to the agent's backstory — the
            free-text field CrewAI renders into the agent's system prompt.
        tool_output_appendix: attacker text appended to every tool's return value.
    """

    system_prompt_suffix: str = ""
    tool_output_appendix: str = ""

    def apply_backstory(self, base: str | None) -> str:
        """Return ``base`` (the agent's backstory) with the injected suffix appended.

        CrewAI substitutes the agent's ``backstory`` into its system-prompt
        template, so appending here plants the attacker text in the agent's system
        prompt. An empty suffix leaves ``base`` unchanged.
        """
        if not self.system_prompt_suffix:
            return base or ""
        if not base:
            return self.system_prompt_suffix
        return f"{base}\n\n{self.system_prompt_suffix}"

    def wrap_tools(self, tools: list[Any]) -> list[Any]:
        """Wrap each tool so its return value gets the appendix appended.

        Returns the tools unchanged when nothing is injected (no wrapper, no
        behaviour change). Otherwise every tool — whatever shape it returns — is
        wrapped so the payload reaches the agent-visible observation.
        """
        if not self.tool_output_appendix:
            return list(tools)
        cls = _injected_tool_cls()
        return [cls(tool, self.tool_output_appendix) for tool in tools]


__all__ = ["InjectionSpec"]
