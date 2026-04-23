"""FlipAttack text transformation functions.

Four flip modes that disguise text by rearranging characters/words.
All functions are pure — no state, no LLM calls.
"""

from __future__ import annotations

import textwrap
from typing import Callable


def flip_word_order(text: str) -> str:
    """FWO: Reverse word sequence."""
    return " ".join(text.split()[::-1])


def flip_chars_in_word(text: str) -> str:
    """FCW: Reverse characters within each word."""
    return " ".join(word[::-1] for word in text.split())


def flip_chars_in_sentence(text: str) -> str:
    """FCS: Reverse entire string."""
    return text[::-1]


def flip_fool_model(text: str) -> str:
    """FMM: Same transformation as FCS, paired with different prompt."""
    return text[::-1]


def split_sentence_in_half(text: str) -> tuple[str, str]:
    """Split text roughly in half at a word boundary (for few-shot demos).

    Uses character midpoint (textwrap), matching the original FlipAttack.
    """
    mid = len(text) // 2
    lines = textwrap.wrap(text, width=mid, break_long_words=False, break_on_hyphens=False)
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    return text, ""


FLIP_MODES: dict[str, Callable[[str], str]] = {
    "FWO": flip_word_order,
    "FCW": flip_chars_in_word,
    "FCS": flip_chars_in_sentence,
    "FMM": flip_fool_model,
}
