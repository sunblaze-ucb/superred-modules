"""GEPA reflective mutation meta-prompt (verbatim from the paper).

The meta-prompt and placeholder names are reproduced verbatim from the
official ``gepa-ai/gepa`` implementation:
``src/gepa/strategies/instruction_proposal.py``
(``InstructionProposalSignature.default_prompt_template``), which is also
the meta-prompt printed in Appendix B of the paper:

    Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
    Reinforcement Learning," arXiv:2507.19457, ICLR 2026.

Two placeholders are required by the upstream contract:

* ``<curr_param>`` — the current instruction text being mutated
  (in our setting, the user-message prompt sent to the target).
* ``<side_info>`` — the markdown-rendered reflective dataset of recent
  rollouts: inputs, assistant responses, and feedback.

Reflection output is expected as the new instruction text inside a
fenced code block (any or no language tag), matching the upstream
``output_extractor``.

Attribution: the meta-prompt template text below (``META_PROMPT_TEMPLATE``)
is copied verbatim from ``gepa-ai/gepa``, MIT License,
Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
(https://github.com/gepa-ai/gepa). Everything else in this module — the
extraction/formatting code around it — is original.
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
    """Substitute the two placeholders in the meta-prompt.

    Mirrors the upstream
    :py:meth:`InstructionProposalSignature.prompt_renderer` placeholder
    substitution. ``current_instruction`` slots into ``<curr_param>``;
    ``side_info`` slots into ``<side_info>`` (typically a markdown
    rendering of the reflective dataset).
    """
    rendered = META_PROMPT_TEMPLATE.replace("<curr_param>", current_instruction)
    rendered = rendered.replace("<side_info>", side_info)
    return rendered


def format_reflective_dataset(samples: Sequence[dict[str, Any]]) -> str:
    """Render a list of rollout records as markdown, one section per record.

    Matches the upstream ``format_samples`` shape: each sample is a
    block ``# Example N`` with ``## key`` for each field. Fields with
    empty / ``None`` values are dropped (so settings that omit
    ``response`` or ``score`` produce a smaller side-info block rather
    than blank sections).
    """
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
    """Render a single field value as markdown text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            if v is None:
                continue
            parts.append(f"### {k}")
            parts.append(_render_value(v))
        return "\n".join(parts) if parts else ""
    if isinstance(value, (list, tuple)):
        parts = []
        for j, item in enumerate(value, start=1):
            parts.append(f"### Item {j}")
            parts.append(_render_value(item))
        return "\n".join(parts) if parts else ""
    return str(value).strip()


__all__ = [
    "META_PROMPT_TEMPLATE",
    "format_reflective_dataset",
    "render_meta_prompt",
]
