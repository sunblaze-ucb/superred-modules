"""Response parsers for the X-Teaming attacker/judge (parse-only ports).

Mirror the extraction logic in MT-JailBench's
``engine/attacks/xteaming/xteaming_prompt_generator.py`` and
``xteaming_outcome_evaluator.py``. No prompt text is reproduced.
"""

from __future__ import annotations

import json

_CONV_OPEN = "<conversation>"
_CONV_CLOSE = "</conversation>"


def parse_strategies(response: str) -> list[dict] | None:
    """Parse the strategy list from the attacker response.

    Upstream expects a JSON object of exactly 10 strategies (a dict keyed by
    name). Returns the list of strategy dicts (values) or ``None`` on failure.
    A fenced code block (```...```) is unwrapped first, as upstream does.
    """
    if not response:
        return None
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    return list(parsed.values())


def strategy_is_valid(strategy: object) -> bool:
    """Whether a strategy dict carries the fields the turn prompts need."""
    return (
        isinstance(strategy, dict)
        and {"persona", "context", "approach", "conversation_plan"} <= set(strategy)
        and isinstance(strategy["conversation_plan"], dict)
        and len(strategy["conversation_plan"]) > 0
    )


def extract_conversation(response: str) -> str:
    """Extract text between ``<conversation>`` tags (upstream helper).

    Falls back to the stripped response if the tags are absent.
    """
    if not response:
        return ""
    start = response.find(_CONV_OPEN)
    end = response.find(_CONV_CLOSE)
    if start == -1 or end == -1:
        return response.strip()
    return response[start + len(_CONV_OPEN) : end].strip()


def parse_judge(response: str) -> tuple[int, str] | None:
    """Parse the X-Teaming judge JSON (``score`` 1-5 + ``reason``).

    Returns ``(score, reason)`` or ``None`` if the score is missing or out of
    the 1-5 range (upstream raises there; the optimizer degrades instead).
    """
    if not response:
        return None
    try:
        result = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(result, dict) or "score" not in result:
        return None
    try:
        score = int(result["score"])
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score, str(result.get("reason", ""))


def format_strategy(strategy: dict) -> str:
    """Render a strategy for the attacker system prompt (upstream layout)."""
    return "\n".join(
        [
            f"Persona: {strategy['persona']}",
            f"Context: {strategy['context']}",
            f"Approach: {strategy['approach']}",
            f"Conversation Plan: {strategy['conversation_plan']}",
        ]
    )
