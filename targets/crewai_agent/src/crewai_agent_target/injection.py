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
import re
from dataclasses import dataclass
from typing import Any

# CrewAI renders the agent's backstory through `interpolate_only`, which scans for
# this exact template-variable shape and raises KeyError for any name not in the
# kickoff `inputs` (only `user_input` is passed). An attacker suffix containing an
# unrelated `{identifier}` would therefore abort the run instead of being
# delivered — so we neutralize such shapes in the *attacker* suffix before it is
# appended (see `apply_backstory`). Kept in sync with CrewAI's own regex.
_TEMPLATE_VAR = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}")


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

        Forwards the inner tool's public config — ``name`` / ``description`` /
        ``args_schema`` / ``result_as_answer`` plus the usage-limit, failure-policy,
        result-schema and cache fields — so the agent sees an identical, identically
        *constrained* tool. That forwarding matters: CrewAI's
        ``to_structured_tool()`` builds the ``CrewStructuredTool`` that enforces
        ``max_usage_count`` / ``tool_failure_policy`` by reading those fields off
        *this* wrapper, so dropping them would silently reset e.g. the usage cap to
        unlimited on exactly the runs where ``tool_output`` is injected.

        The agent-invoked path is ``_run`` (CrewAI binds the tool callable to
        ``_run``, not ``run``); ``run`` is kept for direct callers. Both delegate to
        the inner tool's matching method and append the payload, awaiting a returned
        coroutine first, so the injection reaches the model-visible observation for
        every return shape (the async case is the #183 HIGH false-negative class).
        """

        _inner: Any = PrivateAttr()
        _appendix: str = PrivateAttr(default="")

        def __init__(self, inner: Any, appendix: str) -> None:
            # Forward the public config AND the behaviour-governing fields so a
            # wrapped tool is not silently less constrained than the inner one
            # (see class docstring). hasattr-guarded so this stays valid across
            # CrewAI versions that add/remove such fields.
            fields: dict[str, Any] = {
                "name": inner.name,
                "description": inner.description,
                "args_schema": inner.args_schema,
                "result_as_answer": inner.result_as_answer,
            }
            for extra in (
                "max_usage_count",
                "result_schema",
                "cache_function",
                "tool_failure_policy",
            ):
                if hasattr(inner, extra):
                    fields[extra] = getattr(inner, extra)
            super().__init__(**fields)
            self._inner = inner
            self._appendix = appendix

        def run(self, *args: Any, **kwargs: Any) -> Any:
            # inner.run() validates args, enforces usage limits, and (for most
            # tools) resolves the value; inject into that.
            result = self._inner.run(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return self._await_and_inject(result)
            return _inject_into_result(result, self._appendix)

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            # THE agent-driven path: CrewAI's to_structured_tool() binds the
            # tool's callable to _run (not run), so this is what actually executes
            # under the agent. For an ASYNC inner tool, inner._run(...) returns an
            # un-awaited coroutine — return an async wrapper that awaits it and
            # injects into the concrete result, so the appendix is not silently
            # dropped for async tools (the #183 HIGH-bug false-negative class).
            result = self._inner._run(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return self._await_and_inject(result)
            return _inject_into_result(result, self._appendix)

        async def _await_and_inject(self, coro: Any) -> Any:
            return _inject_into_result(await coro, self._appendix)

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

        CrewAI runs the backstory through ``interpolate_only`` at kickoff, which
        raises ``KeyError`` for any ``{identifier}`` not in the kickoff inputs (only
        ``user_input`` is passed). So a brace-bearing suffix like ``Ignore {system}
        rules`` would abort the run — the injection never reaches the model — rather
        than being delivered. Neutralize such shapes in the *attacker suffix*
        (``{x}`` -> ``{ x }``, which CrewAI's variable regex no longer matches; the
        engine has no brace-escape syntax, so doubling braces does not help). This
        is a faithful-enough delivery (the payload text is preserved bar the two
        spaces) and strictly better than aborting. ``base`` is left untouched: it is
        the caller's own backstory and may legitimately use ``{user_input}``.
        """
        if not self.system_prompt_suffix:
            return base or ""
        suffix = _TEMPLATE_VAR.sub(r"{ \1 }", self.system_prompt_suffix)
        if not base:
            return suffix
        return f"{base}\n\n{suffix}"

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
