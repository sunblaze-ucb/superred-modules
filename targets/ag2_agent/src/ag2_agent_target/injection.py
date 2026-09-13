"""Attacker-controlled injection surfaces for the AG2 (AutoGen) agent target.

A superred agent target must expose the agent's real attack surfaces, not just
the user prompt. Besides ``user_input`` (a direct message), an AG2 agent is
attackable through:

- **tool returns** — content a tool hands back that the agent then reads and acts
  on (indirect prompt injection: the classic agent vector), and
- **its system prompt** — instructions an attacker manages to plant in the
  agent's own system message.

``ag2.Agent`` bakes its ``system_message`` and its tool set in at construction, so
the target builds the agent *per run* and hands the factory an
:class:`InjectionSpec` carrying this run's injected values: the factory appends the
system-prompt suffix to the agent's ``system_message``
(:meth:`InjectionSpec.apply_system_prompt`) and runs its tool callables through
:meth:`InjectionSpec.wrap_tools`, which wraps each so the appendix is appended to
whatever the tool returns. An optimizer's surface classifier then chooses which
named surface to drive; left un-injected, the spec is empty and the agent runs
unchanged.

This module imports **no** framework code: it wraps plain callables and edits plain
strings, so it (and the target that imports it) load even when ``ag2`` is not
installed — the framework is pulled in lazily only when an agent is actually
built/run.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_type_hints

# Key under which the appendix is attached to a dict-shaped tool return. Chosen to
# read as self-describing in the serialized object the model sees; a collision with
# a real key produced by the tool is avoided at injection time (see below).
_NOTE_KEY = "injected_note"


def _inject_into_result(result: Any, appendix: str) -> Any:
    """Append ``appendix`` to a tool's return so it reaches the model-visible text.

    AG2 turns a tool's return value into the text the model reads back on its next
    turn, so the appendix must reach that text whatever *shape* the tool returns.
    This surface exists precisely to exercise that path, so **no return shape may
    silently drop the appendix** — the false-negative bug being: inject into ``str``
    only and skip every other shape (no error, no injection). Every shape is
    handled:

    - ``str``: the appendix is appended as trailing text.
    - ``Mapping`` (dict / JSON object): the appendix is attached under a dedicated
      note key — a non-clobbering key is chosen if it is already present — so it
      appears in the serialized object the model reads.
    - ``list`` / ``tuple`` (JSON array): the appendix is appended as an extra text
      element, preserving list-ness / tuple-ness.
    - any other JSON-able scalar (``int`` / ``float`` / ``bool`` / ``None``) or an
      arbitrary object: it cannot carry a note in-shape, so it is rendered to text
      with the appendix appended — the value still reaches the model, now carrying
      the injection, instead of being dropped.
    """
    if isinstance(result, str):
        return f"{result}\n\n{appendix}"
    if isinstance(result, Mapping):
        injected = dict(result)
        key = _NOTE_KEY
        while key in injected:  # never clobber a real key the tool produced
            key += "_"
        injected[key] = appendix
        return injected
    if isinstance(result, list):
        return [*result, appendix]
    if isinstance(result, tuple):
        return (*result, appendix)
    if isinstance(result, (bytes, bytearray)):
        return f"{result.decode('utf-8', errors='replace')}\n\n{appendix}"
    # Scalars (int / float / bool / None) and any other object: stringify with the
    # appendix so the injection still reaches the model-visible text.
    return f"{result}\n\n{appendix}"


def _pin_signature(tool: Callable[..., Any], wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """Pin the wrapper's advertised signature to the tool's real one.

    ``functools.wraps`` copies the name / docstring / annotations AG2 reads to build
    a tool's JSON schema, and sets ``__wrapped__`` so ``inspect.signature`` follows
    through to the original; pinning ``__signature__`` makes that robust even for an
    inspector that does not follow ``__wrapped__``. Without it a ``*args/**kwargs``
    wrapper would advertise the wrong parameters and corrupt the tool schema.
    """
    try:
        setattr(wrapper, "__signature__", inspect.signature(tool))
    except (ValueError, TypeError):
        # Some callables expose no introspectable signature; leave what wraps copied.
        pass
    # functools.wraps copies the tool's annotations but NOT its __globals__, so a
    # PEP 563 (stringized) annotation naming a non-builtin type would fail to
    # resolve against the wrapper's module (this one) when AG2 builds the tool's
    # JSON schema (NameError). Resolve the annotations against the ORIGINAL tool's
    # own globals and set the concrete types on the wrapper, so the schema builder
    # needs no module-level lookup regardless of PEP 563.
    try:
        wrapper.__annotations__ = dict(get_type_hints(tool))
    except Exception:  # noqa: BLE001 - keep what wraps copied if resolution fails
        pass
    # Drop __wrapped__ so no schema-gen or execution path can inspect.unwrap() back
    # to the un-injected original and silently bypass the tool-output injection.
    if hasattr(wrapper, "__wrapped__"):
        del wrapper.__wrapped__
    return wrapper


def _wrap_tool(tool: Callable[..., Any], appendix: str) -> Callable[..., Any]:
    """Wrap one AG2 tool callable so its return carries ``appendix`` (any shape).

    Sync and async tools are both handled — including a callable *object* whose
    ``__call__`` is a coroutine function (``iscoroutinefunction`` is False for the
    object itself, so check ``__call__`` too, else its coroutine return would be
    stringified and the real output lost).
    """
    if inspect.iscoroutinefunction(tool) or inspect.iscoroutinefunction(
        getattr(tool, "__call__", None)
    ):

        @functools.wraps(tool)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            return _inject_into_result(await tool(*args, **kwargs), appendix)

        return _pin_signature(tool, awrapper)

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return _inject_into_result(tool(*args, **kwargs), appendix)

    return _pin_signature(tool, wrapper)


@dataclass(frozen=True)
class InjectionSpec:
    """The injected values for one run, handed to the agent factory.

    Attributes:
        system_prompt_suffix: attacker text appended to the agent's system message.
        tool_output_appendix: attacker text appended to every tool's return value.
    """

    system_prompt_suffix: str = ""
    tool_output_appendix: str = ""

    def apply_system_prompt(self, base: str | None) -> str | None:
        """Return ``base`` with the injected system-prompt suffix appended."""
        if not self.system_prompt_suffix:
            return base
        return f"{base}\n\n{self.system_prompt_suffix}" if base else self.system_prompt_suffix

    def wrap_tools(self, tools: Sequence[Callable[..., Any]]) -> list[Callable[..., Any]]:
        """Wrap each tool so its return carries the tool-output appendix.

        Returns the tools unchanged (a fresh list of the same callables) when
        nothing is injected, so an un-injected run behaves exactly as the caller's
        own tools would.
        """
        if not self.tool_output_appendix:
            return list(tools)
        return [_wrap_tool(tool, self.tool_output_appendix) for tool in tools]


__all__ = ["InjectionSpec"]
