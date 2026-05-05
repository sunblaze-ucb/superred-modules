"""Reflective LLM driver for GEPA.

Wraps :class:`superred.core.llm.LLMClient` for the single reflective
mutation step of GEPA's algorithm: build a side-info block from a few
recorded rollouts of the parent candidate, ask the reflection LM to
propose a new instruction, and parse the proposed instruction out of
the fenced code block. Mirrors
``InstructionProposalSignature.run_with_metadata`` semantics in the
upstream ``gepa-ai/gepa`` reference.

The reflective dataset gracefully tolerates missing fields, which is
what the optimizer relies on to operate across the four adversarial
information-access settings without a setting knob:

* ``response is None`` (out of scope) — no ``assistant_response`` field
  is emitted for that record.
* ``score is None`` and ``rationale == ""`` (no feedback) — no
  ``feedback`` field is emitted for that record.

Refer to ``ASSUMPTIONS.md`` for the full setting table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from superred.core.llm import LLMClient

from gepa_optimizer.prompts import (
    META_PROMPT_TEMPLATE,
    format_reflective_dataset,
    render_meta_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutRecord:
    """A single past rollout used to build the reflective dataset."""

    goal: str
    prompt: str
    response: str | None = None
    score: float | None = None
    rationale: str = ""

    def to_sample(self) -> dict[str, Any]:
        """Convert to the markdown-renderable sample dict.

        Empty / ``None`` fields are dropped by ``format_reflective_dataset``.
        """
        sample: dict[str, Any] = {
            "task_input": self.goal,
            "candidate_instruction": self.prompt,
        }
        if self.response is not None:
            sample["assistant_response"] = self.response
        feedback = _format_feedback(score=self.score, rationale=self.rationale)
        if feedback:
            sample["feedback"] = feedback
        return sample


@dataclass
class ReflectionResult:
    """The output of a reflective mutation step."""

    new_instruction: str
    raw_output: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Reflector:
    """Stateless reflective LLM driver.

    Each ``propose`` call builds a fresh meta-prompt from a parent
    instruction and a list of rollout records, and returns the parsed
    new instruction. There is no chat history — reflection is
    independent across iterations, matching the upstream
    ``ReflectiveMutationProposer`` shape.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        temperature: float = 1.0,
    ) -> None:
        self._llm = llm
        self._temperature = temperature

    async def propose(
        self,
        *,
        current_instruction: str,
        rollouts: list[RolloutRecord],
    ) -> ReflectionResult | None:
        """Propose a new instruction from the parent + recent rollouts.

        Returns ``None`` (and logs a warning) on parse failure, so the
        optimizer can fall back to keeping the parent unchanged for
        the next attempt without crashing the run.
        """
        side_info = format_reflective_dataset(
            [record.to_sample() for record in rollouts]
        )
        prompt = render_meta_prompt(
            current_instruction=current_instruction,
            side_info=side_info,
        )

        response = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        raw = response.choices[0].message.content or ""

        new_instruction = _extract_fenced_block(raw)
        if not new_instruction:
            logger.warning(
                "GEPA reflector: could not extract a fenced instruction "
                "from the reflection LM output; keeping parent instruction"
            )
            return None

        return ReflectionResult(
            new_instruction=new_instruction,
            raw_output=raw,
            prompt=prompt,
        )


def _format_feedback(*, score: float | None, rationale: str) -> str:
    """Render score + rationale into a single feedback string."""
    parts: list[str] = []
    if score is not None:
        parts.append(f"score: {score:.4f} on a scale of 0.0 to 1.0")
    if rationale.strip():
        parts.append(f"rationale: {rationale.strip()}")
    return "\n".join(parts)


_FENCE_RE = re.compile(r"```\S*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_fenced_block(text: str) -> str:
    """Extract the new instruction from the LM output.

    Mirrors the upstream ``output_extractor``: take the content between
    the first and last triple-backticks, stripping optional language
    specifier; tolerate missing closing fence by trimming the leading
    fence.
    """
    if not text or not text.strip():
        return ""

    matches = list(_FENCE_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.match(r"^```\S*\n?", stripped)
        if match:
            return stripped[match.end():].strip().rstrip("`").strip()
    if stripped.endswith("```"):
        return stripped[:-3].strip()

    return ""


__all__ = [
    "META_PROMPT_TEMPLATE",
    "ReflectionResult",
    "Reflector",
    "RolloutRecord",
]
