"""FlipAttack text transformation functions.

Four flip modes that disguise text by rearranging characters/words.
All functions are pure — no state, no LLM calls.
"""

from __future__ import annotations

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
    """Split text roughly in half at a word boundary (for few-shot demos)."""
    words = text.split()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


FLIP_MODES: dict[str, Callable[[str], str]] = {
    "FWO": flip_word_order,
    "FCW": flip_chars_in_word,
    "FCS": flip_chars_in_sentence,
    "FMM": flip_fool_model,
}
