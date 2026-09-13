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
    """Give the wrapper the tool's real name, signature, and RESOLVED annotations.

    AG2 builds a tool's JSON schema from the callable's name, signature, and type
    hints. ``functools.wraps`` copies these for a plain function, but two gaps
    remain that this closes:

    - Annotations are copied as-is; under PEP 563 they are STRINGS resolvable only
      against the ORIGINAL tool's ``__globals__``, which this wrapper's module does
      not share, so a non-builtin annotation would ``NameError`` in AG2's schema
      builder. Resolve the hints against the tool's own globals and pin the concrete
      types on BOTH ``__annotations__`` and the pinned ``__signature__`` — a
      consumer may read either.
    - For a callable OBJECT (an instance whose ``__call__`` is the tool), ``wraps``
      can't copy ``__name__``/``__doc__`` (the instance has none); take the name
      from the class and read the signature/hints from ``__call__``.

    Also drops ``__wrapped__`` so nothing can ``inspect.unwrap()`` back to the
    un-injected original and bypass the tool-output injection.
    """
    is_function = inspect.isfunction(tool) or inspect.ismethod(tool)
    hint_source: Any = tool if is_function else getattr(tool, "__call__", tool)
    # Name/doc: correct even for a callable object (whose instance lacks __name__).
    name = getattr(tool, "__name__", None) or type(tool).__name__
    wrapper.__name__ = name
    wrapper.__qualname__ = getattr(tool, "__qualname__", None) or name
    doc = getattr(tool, "__doc__", None)
    if doc:
        wrapper.__doc__ = doc
    try:
        sig = inspect.signature(tool)
    except (ValueError, TypeError):
        # Some callables expose no introspectable signature; leave what wraps copied.
        return _drop_wrapped(wrapper)
    try:
        # include_extras=True keeps Annotated[T, ...] metadata (AG2's canonical
        # Annotated[str, "description"] param docs, Depends(...) markers) intact —
        # else the injected arm gets a stripped tool schema while the no-injection
        # control arm (wrap_tools is a no-op) keeps the full one, confounding the
        # before/after comparison.
        hints = dict(get_type_hints(hint_source, include_extras=True))
    except Exception:  # noqa: BLE001 - unresolved hints: keep the raw signature
        hints = {}
    if hints:
        params = [
            p.replace(annotation=hints.get(n, p.annotation))
            for n, p in sig.parameters.items()
        ]
        sig = sig.replace(
            parameters=params,
            return_annotation=hints.get("return", sig.return_annotation),
        )
        wrapper.__annotations__ = hints
    setattr(wrapper, "__signature__", sig)
    return _drop_wrapped(wrapper)


def _drop_wrapped(wrapper: Callable[..., Any]) -> Callable[..., Any]:
    """Remove ``__wrapped__`` so nothing can ``inspect.unwrap()`` past the wrapper."""
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
