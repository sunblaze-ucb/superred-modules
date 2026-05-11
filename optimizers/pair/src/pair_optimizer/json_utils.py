"""Official-style JSON extraction for PAIR attacker responses."""

from __future__ import annotations

import ast
from dataclasses import dataclass


class JsonExtractionError(ValueError):
    """Raised when an attacker response does not contain valid PAIR JSON."""


@dataclass(frozen=True)
class PairProposal:
    improvement: str
    prompt: str
    system_prompt: str | None = None


def extract_attack_json(content: str) -> PairProposal:
    """Extract PAIR's first JSON object using the official parsing shape."""
    start_pos = content.find("{")
    end_pos = content.find("}") + 1
    if start_pos < 0 or end_pos <= 0 or end_pos <= start_pos:
        raise JsonExtractionError("No JSON object found")
    json_str = content[start_pos:end_pos].replace("\n", "")
    try:
        parsed = ast.literal_eval(json_str)
    except (SyntaxError, ValueError) as exc:
        raise JsonExtractionError("Invalid JSON object") from exc
    if not isinstance(parsed, dict):
        raise JsonExtractionError("Extracted object is not a dictionary")
    if not all(key in parsed for key in ("improvement", "prompt")):
        raise JsonExtractionError("Missing improvement or prompt")
    improvement = parsed["improvement"]
    prompt = parsed["prompt"]
    system_prompt = parsed.get("system_prompt")
    if not isinstance(improvement, str) or not isinstance(prompt, str):
        raise JsonExtractionError("improvement and prompt must be strings")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise JsonExtractionError("system_prompt must be a string when present")
    return PairProposal(improvement=improvement, prompt=prompt, system_prompt=system_prompt)
