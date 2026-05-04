"""Small JSON parsing helpers shared by ManyShot components."""

from __future__ import annotations

import re


def strip_markdown_fences(text: str) -> str:
    """Remove optional markdown code fences wrapping JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```$", "", text)
    return text.strip()
