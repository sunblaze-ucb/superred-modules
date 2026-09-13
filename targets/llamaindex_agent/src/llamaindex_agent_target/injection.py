"""Attacker-controlled injection surfaces for the LlamaIndex agent target.

A superred agent target must expose the agent's real attack surfaces, not just
the user prompt. Besides ``user_input`` (a direct message), an agent is attackable
through:

- **tool returns** — content a tool hands back that the agent then reads and acts
  on (indirect prompt injection: the classic agent vector), and
- **its system prompt** — instructions an attacker manages to plant in the
  agent's own system message.

A ``ReActAgent`` bakes its tools and system prompt in at construction time, so the
target builds the agent *per run* and hands the factory an :class:`InjectionSpec`
carrying the current run's injected values: the factory appends the system-prompt
text (via the ReActAgent's ``system_prompt`` parameter, which the ReAct formatter
folds into the system header) and wraps each tool via :meth:`InjectionSpec.wrap_tools`
so every tool's returned ``ToolOutput`` carries the attacker payload the agent
reads back. An optimizer's surface classifier then chooses which of these named
surfaces to drive; left un-injected, the spec is empty and the agent runs unchanged.

The framework is imported **lazily** (inside the wrapping helpers, never at module
top) so this module — and the target that imports it — stays importable without
``llama-index`` installed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final

# Sentinel distinguishing "attribute absent" from a real ``None`` attribute value.
_MISSING: Final = object()


def _as_text(value: Any) -> str:
    """Render ``value`` as the text the model reads (``str`` passes through)."""
    return value if isinstance(value, str) else str(value)


def _append_block(output: Any, appendix: str) -> None:
    """Append a text block to a tool output that carries content ``blocks``.

    Newer ``llama-index`` tool outputs can carry a list of multimodal content
    ``blocks`` alongside ``content``; when present, append a ``TextBlock`` so the
    appendix is visible there too. Guarded on every axis (attribute absent, not a
    list, ``TextBlock`` import unavailable) so an output shape without blocks is a
    safe no-op — ``content`` already carries the appendix in that case.
    """
    blocks = getattr(output, "blocks", None)
    if not isinstance(blocks, list):
        return
    try:
        from llama_index.core.base.llms.types import TextBlock
    except Exception:  # noqa: BLE001 - blocks are optional; content already injected
        return
    output.blocks = [*blocks, TextBlock(text=appendix)]


def _append_appendix(result: Any, appendix: str) -> Any:
    """Append ``appendix`` to the model-visible text of a tool result.

    A wrapped tool's ``call`` / ``acall`` returns a ``ToolOutput`` whose ``content``
    is the text the ReAct agent reads back as its observation. This must reach that
    text for **every** shape the result can take — silently dropping the payload on
    some shape is a false negative on the exact surface this exists to exercise
    (the #183 tool-return bug), with no error to signal it:

    - a ``ToolOutput`` (the normal case) — regardless of whether its ``raw_output``
      is a plain string or a structured object: append to ``content`` (the
      model-visible text), to any multimodal ``blocks``, and, when ``raw_output``
      is itself a string, to ``raw_output`` too so a consumer reading it stays
      consistent;
    - a bare ``str`` (defensive: some tools return the string directly);
    - any other shape (no ``content`` attribute, not a string) — coerced to text so
      the appendix still reaches the model rather than being dropped.
    """
    if not appendix:
        return result
    if isinstance(result, str):
        return f"{result}\n\n{appendix}"
    content = getattr(result, "content", _MISSING)
    if content is not _MISSING:
        result.content = f"{_as_text(content)}\n\n{appendix}"
        _append_block(result, appendix)
        raw = getattr(result, "raw_output", _MISSING)
        if isinstance(raw, str):
            result.raw_output = f"{raw}\n\n{appendix}"
        return result
    return f"{_as_text(result)}\n\n{appendix}"


def _wrap_tool(tool: Any, appendix: str) -> Any:
    """Return a copy of ``tool`` whose returns carry the tool-output injection.

    Wraps at the tool-call boundary (``call`` / ``acall``, the stable async-tool
    contract), so it is agnostic to how the tool builds its ``ToolOutput`` and to
    sync-vs-async dispatch (``__call__`` routes through ``call``). A shallow copy is
    mutated rather than the input tool, and the class is preserved so the runtime's
    ``isinstance`` checks and the tool's metadata/schema are untouched.
    """
    wrapped = copy.copy(tool)
    orig_call = tool.call
    orig_acall = getattr(tool, "acall", None)

    def call(*args: Any, **kwargs: Any) -> Any:
        return _append_appendix(orig_call(*args, **kwargs), appendix)

    async def acall(*args: Any, **kwargs: Any) -> Any:
        if orig_acall is not None:
            return _append_appendix(await orig_acall(*args, **kwargs), appendix)
        # No async variant: fall back to the sync call so the surface still fires.
        return _append_appendix(orig_call(*args, **kwargs), appendix)

    wrapped.call = call
    wrapped.acall = acall
    return wrapped


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

    def wrap_tools(self, tools: list[Any]) -> list[Any]:
        """Wrap each tool so its output carries the injection (unchanged if none)."""
        if not self.tool_output_appendix:
            return tools
        return [_wrap_tool(tool, self.tool_output_appendix) for tool in tools]


__all__ = ["InjectionSpec"]
