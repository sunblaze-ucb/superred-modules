"""Reflective LLM driver for GEPA-Agentic."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from superred.core.llm import LLMClient

from gepa_agentic_optimizer.prompts import (
    format_reflective_dataset,
    render_meta_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutRecord:
    """A single agentic rollout used to build GEPA's reflective dataset."""

    goal: str
    prompt: str
    response: str | None = None
    score: float | None = None
    rationale: str = ""
    target_observables: dict[str, str] | None = None
    selected_surface: dict[str, Any] | None = None
    observed_surfaces: list[dict[str, str]] | None = None
    tool_return: str | None = None
    agent_observations: list[str] | None = None
    selection_reason: str = ""

    def to_sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "task_input": self.goal,
            "candidate_instruction": self.prompt,
        }
        if self.target_observables:
            sample["target_observables"] = dict(self.target_observables)
        if self.selected_surface:
            sample["selected_surface"] = dict(self.selected_surface)
        if self.observed_surfaces:
            sample["observed_surfaces"] = list(self.observed_surfaces)
        if self.selection_reason.strip():
            sample["surface_selection_reason"] = self.selection_reason.strip()
        if self.tool_return is not None:
            sample["tool_return"] = self.tool_return
        if self.agent_observations:
            sample["agent_observations"] = list(self.agent_observations)
        if self.response is not None:
            sample["assistant_response"] = self.response
        feedback = _format_feedback(score=self.score, rationale=self.rationale)
        if feedback:
            sample["feedback"] = feedback
        return sample


@dataclass
class ReflectionResult:
    new_instruction: str
    raw_output: str
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Reflector:
    """Stateless GEPA reflective mutation driver."""

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
                "GEPA-Agentic reflector: could not extract fenced instruction; "
                "keeping parent instruction"
            )
            return None
        return ReflectionResult(
            new_instruction=new_instruction,
            raw_output=raw,
            prompt=prompt,
        )


def _format_feedback(*, score: float | None, rationale: str) -> str:
    parts: list[str] = []
    if score is not None:
        parts.append(f"score: {score:.4f} on a scale of 0.0 to 1.0")
    if rationale.strip():
        parts.append(f"rationale: {rationale.strip()}")
    return "\n".join(parts)


def _extract_fenced_block(text: str) -> str:
    if not text or not text.strip():
        return ""

    start = text.find("```") + 3
    end = text.rfind("```")
    if start >= end:
        stripped = text.strip()
        if stripped.startswith("```"):
            match = re.match(r"^```\S*\n?", text.lstrip())
            if match:
                return text.lstrip()[match.end():].strip()
        if stripped.endswith("```"):
            return stripped[:-3].strip()
        return ""

    content = text[start:end]
    match = re.match(r"^\S*\n", content)
    if match:
        content = content[match.end():]
    return content.strip()


__all__ = [
    "ReflectionResult",
    "Reflector",
    "RolloutRecord",
]
