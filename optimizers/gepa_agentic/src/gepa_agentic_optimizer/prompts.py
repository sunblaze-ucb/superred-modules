"""GEPA reflective mutation meta-prompt for agentic target rollouts.

Attribution: ``META_PROMPT_TEMPLATE`` below is copied verbatim from
``gepa-ai/gepa`` (``src/gepa/strategies/instruction_proposal.py``,
``InstructionProposalSignature.default_prompt_template``), MIT License,
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
(https://github.com/gepa-ai/gepa) — the same source ``superred-optimizer-gepa``
reproduces in its own ``prompts.py``. This package keeps its own copy rather
than importing that package's, so the two must be kept in sync by hand if the
upstream template ever changes; everything else in this module is original.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


META_PROMPT_TEMPLATE = """\
I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""


def render_meta_prompt(*, current_instruction: str, side_info: str) -> str:
    rendered = META_PROMPT_TEMPLATE.replace("<curr_param>", current_instruction)
    return rendered.replace("<side_info>", side_info)


def format_reflective_dataset(samples: Sequence[dict[str, Any]]) -> str:
    if not samples:
        return "(no reflective examples available)"

    blocks: list[str] = []
    for i, sample in enumerate(samples, start=1):
        lines = [f"# Example {i}"]
        for key, value in sample.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            lines.append(f"## {key}")
            lines.append(_render_value(value))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            parts.append(f"### {key}")
            parts.append(_render_value(item))
        return "\n".join(parts) if parts else ""
    if isinstance(value, (list, tuple)):
        parts = []
        for index, item in enumerate(value, start=1):
            parts.append(f"### Item {index}")
            parts.append(_render_value(item))
        return "\n".join(parts) if parts else ""
    return str(value).strip()


__all__ = [
    "META_PROMPT_TEMPLATE",
    "format_reflective_dataset",
    "render_meta_prompt",
]
