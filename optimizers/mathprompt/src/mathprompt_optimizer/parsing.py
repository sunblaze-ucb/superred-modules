"""Parse the attacker LLM's JSON replies for each multilingual step.

Upstream validates these with pydantic models (``math_problem/schema.py``
``EnhancedAttack``/``IsMathProblem`` and ``compliance.py`` ``ComplianceData``).
This module reads the same fields without a pydantic dependency and returns
``None``/defaults rather than raising, mirroring upstream's retry-on-failure.
"""

from __future__ import annotations

import json
import re


def _load(raw: str) -> dict | None:
    """The first JSON object in *raw*, tolerant of a fence or surrounding prose.

    Upstream constrains these replies with pydantic (``generate(prompt, Model,
    ...)``), so a chatty answer never reaches its parser. This module calls
    ``llm.complete`` with no structured-output enforcement, which makes
    "Here is the JSON: {...}" a realistic shape -- and a reply rejected here
    costs a retry and, once retries run out, silently degrades the attack to
    the un-encoded goal. The bare-object fallback matches the one the shared
    ``surface_llm.parse_json_object`` already uses.
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    text = fence.group(1).strip() if fence else raw.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_enhanced_attack(raw: str) -> tuple[str, str] | None:
    """``EnhancedAttack`` -> ``(math_strategy, input)``.

    ``None`` when unreadable or the encoded ``input`` is empty (upstream
    would retry).
    """
    data = _load(raw)
    if not data:
        return None
    encoded = data.get("input")
    if not isinstance(encoded, str) or not encoded.strip():
        return None
    strategy = data.get("math_strategy")
    return (strategy if isinstance(strategy, str) else ""), encoded.strip()


def parse_is_math_problem(raw: str) -> bool | None:
    """``IsMathProblem.is_math_problem``. ``None`` when unreadable."""
    data = _load(raw)
    if not data:
        return None
    value = data.get("is_math_problem")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def parse_non_compliant(raw: str) -> bool | None:
    """``ComplianceData.non_compliant`` -> True when the reply is a hard refusal.

    ``None`` when unreadable, which the caller treats conservatively (see the
    optimizer): an unreadable compliance check does not by itself accept a turn.
    """
    data = _load(raw)
    if not data:
        return None
    value = data.get("non_compliant")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def rebuild_dump(math_strategy: str, encoded_input: str) -> dict[str, str]:
    """Reconstruct ``EnhancedAttack.model_dump()`` for the downstream checks.

    Upstream feeds ``res.model_dump()`` (``{"math_strategy", "input"}``) into
    both the compliance and is-math-problem prompts, so the checks see the same
    JSON upstream does.
    """
    return {"math_strategy": math_strategy, "input": encoded_input}


__all__ = [
    "parse_enhanced_attack",
    "parse_is_math_problem",
    "parse_non_compliant",
    "rebuild_dump",
]
