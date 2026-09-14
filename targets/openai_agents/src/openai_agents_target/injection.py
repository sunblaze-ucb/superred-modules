"""Attacker-controlled injection surfaces for the OpenAI Agents SDK target.

A superred agent target must expose the agent's real attack surfaces, not just
the user prompt. Besides ``user_input`` (a direct message), an agent is attackable
through:

- **tool returns** — content a tool hands back that the agent then reads and acts
  on (indirect prompt injection: the classic agent vector), and
- **its system prompt** — instructions an attacker manages to plant in the
  agent's own instructions.

The Agents SDK ``Agent`` is mutable, so the target applies an
:class:`InjectionSpec` to an already-built agent per run:
:meth:`InjectionSpec.apply_instructions` appends the system-prompt text to the
agent's instructions and :meth:`InjectionSpec.wrap_tools` swaps ``agent.tools``
for tools whose ``on_invoke_tool`` appends the tool-return payload to every
result. Left un-injected, the spec is empty and the agent runs unchanged.

Tool-return injection (the indirect surface) must reach the *model-visible text*
for every shape ``FunctionTool.on_invoke_tool`` may return per the SDK contract —
a plain ``str``, one of the structured outputs (:class:`ToolOutputText` /
:class:`ToolOutputImage` / :class:`ToolOutputFileContent`), the ``dict`` form of
those, or a ``list`` of them. Appending only to ``str`` results — the naive
version — silently drops the payload for structured and non-text returns, a false
negative on the very surface this exists to exercise, with no error to signal it.
For a non-text output (image / file) the appendix is added as an ADDITIONAL text
item rather than dropped, so it still reaches the model.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agents import (
    FunctionTool,
    ToolOutputFileContent,
    ToolOutputImage,
    ToolOutputText,
)

# The SDK's dynamic-instructions callable: ``(run_context, agent) -> str |
# Awaitable[str]`` (see agents.Agent.get_system_prompt). Typed with Any args to
# avoid coupling to the SDK's run-context / agent generics.
InstructionsFunc = Callable[[Any, Any], "str | Awaitable[str]"]

# The structured tool-output models the SDK converts to model-visible content
# blocks (see agents.items.ItemHelpers._convert_single_tool_output_pydantic_model):
# ToolOutputText -> input_text, ToolOutputImage -> input_image, ToolOutputFileContent
# -> input_file. Their dict forms carry a discriminating "type" field.
_STRUCTURED_MODELS = (ToolOutputText, ToolOutputImage, ToolOutputFileContent)
_STRUCTURED_DICT_TYPES = frozenset({"text", "image", "file"})


def _is_structured_item(item: Any) -> bool:
    """Whether ``item`` is an SDK structured tool-output (model or typed dict)."""
    if isinstance(item, _STRUCTURED_MODELS):
        return True
    return isinstance(item, dict) and item.get("type") in _STRUCTURED_DICT_TYPES


def _append_text(base: str, appendix: str) -> str:
    return f"{base}\n\n{appendix}" if base else appendix


def _inject_tool_output(result: Any, appendix: str) -> Any:
    """Append ``appendix`` to the model-visible text of any on_invoke_tool return.

    Handles every documented return shape so the injection is never silently
    dropped:

    - ``str`` (the common case) -> string-append.
    - :class:`ToolOutputText` (structured text) -> extend its ``text``.
    - :class:`ToolOutputImage` / :class:`ToolOutputFileContent` (non-text) ->
      return ``[output, ToolOutputText(appendix)]`` so the payload rides along as
      an ADDITIONAL text item instead of being dropped (an image/file block has no
      text to extend).
    - a structured-output ``dict`` (``{"type": "text"|"image"|"file", ...}``) ->
      the same handling on the dict form.
    - a ``list`` / ``tuple`` of structured outputs -> append a text item; a list
      the SDK would otherwise stringify is stringified here and appended.
    - anything else str()-able -> stringify and append (matches the SDK's own
      ``str(output)`` fallback, plus the payload).
    """
    text_item = ToolOutputText(text=appendix)
    if isinstance(result, str):
        return _append_text(result, appendix)
    if isinstance(result, ToolOutputText):
        return ToolOutputText(text=_append_text(result.text, appendix))
    if isinstance(result, (ToolOutputImage, ToolOutputFileContent)):
        return [result, text_item]
    if isinstance(result, dict) and _is_structured_item(result):
        if result.get("type") == "text" and "text" in result:
            # A dict that already validates as ToolOutputText: append to its text.
            merged = dict(result)
            merged["text"] = _append_text(str(result["text"]), appendix)
            return merged
        if result.get("type") != "text":
            # A structured non-text item (image/file): add a separate text item so
            # the original block is preserved.
            return [result, text_item]
        # type == "text" but no "text" field: it does NOT validate as ToolOutputText
        # (the SDK str()s the whole dict, showing every field), so forging a "text"
        # key would DROP the real fields. Fall through to the generic str()+append
        # below, which mirrors the SDK's own fallback and keeps the fields.
    if isinstance(result, (list, tuple)):
        items = list(result)
        if items and all(_is_structured_item(it) for it in items):
            return [*items, text_item]
        # Not an all-structured list -> the SDK would str() the whole thing; do the
        # same here so the appendix still reaches the model as text.
        return _append_text(str(result), appendix)
    # Any other str()-able value (dict without a type, number, None, object).
    return _append_text(str(result), appendix)


def _wrap_tool(tool: FunctionTool, appendix: str) -> FunctionTool:
    """Return a copy of ``tool`` whose output has ``appendix`` appended.

    The wrapper awaits the original ``on_invoke_tool`` (so the tool's own logic
    and its internal output-type validation still run) and injects into the
    result. The copy drops ``output_json_schema`` / ``_output_type_adapter`` so a
    strict-typed tool's now-augmented output is not re-validated against the
    original schema by the runner and rejected — without this, injecting into a
    tool declared with ``output_type=`` / ``output_json_schema=`` raises a
    ``UserError`` instead of delivering the payload.
    """
    original = tool.on_invoke_tool

    async def on_invoke_tool(ctx: Any, args_json: str) -> Any:
        return _inject_tool_output(await original(ctx, args_json), appendix)

    return dataclasses.replace(
        tool,
        on_invoke_tool=on_invoke_tool,
        output_json_schema=None,
        _output_type_adapter=None,
    )


@dataclass(frozen=True)
class InjectionSpec:
    """The injected values for one run.

    Attributes:
        system_prompt_suffix: attacker text appended to the agent's instructions.
        tool_output_appendix: attacker text appended to every tool's return.
    """

    system_prompt_suffix: str = ""
    tool_output_appendix: str = ""

    def apply_instructions(
        self, base: str | InstructionsFunc | None
    ) -> str | InstructionsFunc | None:
        """Return ``base`` with the injected system-prompt suffix appended.

        ``base`` is the agent's effective instructions: static (``str`` / ``None``)
        or the SDK's dynamic-instructions callable ``(run_context, agent) -> str |
        Awaitable[str]``. For a callable, the suffix is appended to the *resolved*
        text at call time via a wrapping callable, rather than being silently
        dropped for dynamic-instruction agents (the analogue of not dropping
        tool-output injection on non-str return shapes).
        """
        suffix = self.system_prompt_suffix
        if not suffix:
            return base
        if base is None or isinstance(base, str):
            return f"{base}\n\n{suffix}" if base else suffix

        original = base

        async def _with_suffix(run_context: Any, agent: Any) -> str:
            resolved = original(run_context, agent)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            return f"{resolved}\n\n{suffix}" if resolved else suffix

        return _with_suffix

    def wrap_tools(self, tools: list[Any]) -> list[Any]:
        """Return ``tools`` with tool-return injection wired.

        An empty ``tool_output_appendix`` returns the tools unchanged (no
        injection). Only :class:`FunctionTool`s are wrapped; other tool kinds
        (hosted web/file search, computer-use, hosted MCP) expose no
        ``on_invoke_tool`` this appendix could reach and are passed through
        untouched.
        """
        if not self.tool_output_appendix:
            return list(tools)
        return [
            _wrap_tool(t, self.tool_output_appendix) if isinstance(t, FunctionTool) else t
            for t in tools
        ]


__all__ = ["InjectionSpec"]
