"""Parse the simulator's JSON replies for each Bad Likert Judge step.

Upstream validates these with pydantic models in
``bad_likert_judge/schema.py`` (``LikertExamples``, ``RefinedExample``,
``NewAttackPrompt``) plus DeepTeam's shared ``NonRefusal``. This module reads
the same fields without taking a pydantic dependency, and returns ``None``
instead of raising -- upstream treats a failed ``generate`` as a backtrack, so
the caller needs a value it can branch on, not an exception.
"""

from __future__ import annotations

import json
import re


def _strip_fence(text: str) -> str:
    """Drop a ```json ... ``` fence if the model wrapped its reply in one."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return fence.group(1).strip() if fence else text.strip()


def _load(raw: str) -> dict | None:
    try:
        parsed = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_score_3_example(raw: str) -> str | None:
    """``LikertExamples.score_3_example`` -- the most harmful example."""
    data = _load(raw)
    if not data:
        return None
    value = data.get("score_3_example")
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_refined_example(raw: str) -> str | None:
    """``RefinedExample.refined_example``."""
    data = _load(raw)
    if not data:
        return None
    value = data.get("refined_example")
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_new_attack_prompt(raw: str) -> str | None:
    """``NewAttackPrompt.new_attack_prompt`` -- the turn actually sent."""
    data = _load(raw)
    if not data:
        return None
    value = data.get("new_attack_prompt")
    return value.strip() if isinstance(value, str) and value.strip() else None


def parse_refusal(raw: str) -> bool | None:
    """``NonRefusal.refusal``.

    ``None`` means the reply was unreadable. Upstream's ``generate`` would have
    raised there; the caller treats it as "not a refusal" so an unparseable
    check never silently costs a turn, matching upstream's behaviour of only
    backtracking on an explicit ``refusal: true``.
    """
    data = _load(raw)
    if not data:
        return None
    value = data.get("refusal")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
    return None


__all__ = [
    "parse_new_attack_prompt",
    "parse_refined_example",
    "parse_refusal",
    "parse_score_3_example",
]
