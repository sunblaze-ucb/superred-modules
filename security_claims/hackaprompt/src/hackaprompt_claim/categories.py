"""The HackAPrompt levels (0-10).

Upstream ships 11 levels of increasing difficulty; each is a defended LLM
application (``hackaprompt/evaluator.py`` ``Level*Evaluator``). The short names
below are descriptive labels for the task metadata; the authoritative defense for
each level lives in the paired target's vendored port.
"""

from __future__ import annotations

LEVELS: tuple[int, ...] = tuple(range(11))

LEVEL_NAMES: dict[int, str] = {
    0: "positive-bot",
    1: "translation-bot",
    2: "secret-key-extraction",
    3: "few-shot-translation",
    4: "search-engine-sandwich",
    5: "writing-assistant",
    6: "story-then-self-check",
    7: "movie-title-word-blocklist",
    8: "few-shot-injection-detector",
    9: "slash-separated-char-escape",
    10: "emoji-only",
}


def level_count() -> int:
    return len(LEVELS)


__all__ = ["LEVELS", "LEVEL_NAMES", "level_count"]
