"""Template helpers and bundled official GPTFuzzer seeds."""

from __future__ import annotations

import csv
from importlib import resources

QUESTION_PLACEHOLDER = "[INSERT PROMPT HERE]"


def synthesis_message(question: str, prompt: str) -> str | None:
    """Insert the target question into a jailbreak template."""
    if QUESTION_PLACEHOLDER not in prompt:
        return None
    return prompt.replace(QUESTION_PLACEHOLDER, question)


def load_official_seed_templates() -> list[str]:
    """Load the official GPTFuzzer.csv seed templates bundled with this package."""
    ref = resources.files("gptfuzzer_optimizer") / "data" / "GPTFuzzer.csv"
    with resources.as_file(ref) as path:
        with open(path, newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            return [row["text"] for row in rows if row.get("text")]
