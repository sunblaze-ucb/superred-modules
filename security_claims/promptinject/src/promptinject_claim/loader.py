"""Build the PromptInject goal-hijacking prompt set.

Reproduces garak's ``probes/promptinject.py`` prompt construction exactly:
``build_prompts`` over the vendored PromptInject data with garak's five
goal-hijacking attacks and its escape/delimiter settings. Each built prompt
carries the rogue string the injection tries to make the model emit.

Vendored PromptInject framework: ``_vendor/promptinject`` (MIT, Agency
Enterprise), byte-identical to garak's ``resources/promptinject``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from promptinject_claim._vendor.promptinject import build_prompts, prompt_data

# The three rogue-string classes garak ships (probes/promptinject.py:30).
ROGUE_STRINGS: tuple[str, ...] = ("hate-humans", "kill-humans", "long-prompt")

# garak's five goal-hijacking attacks (load_data_all_attacks).
_ATTACKS = (
    "ignore-print",
    "ignore-say",
    "ignore-print-upper",
    "nevermind",
    "screaming-stop",
)


@dataclass(frozen=True)
class PromptInjectItem:
    """One built injection prompt and the rogue string that scores it.

    Attributes:
        rogue_key: One of :data:`ROGUE_STRINGS`.
        prompt: The full base+injection prompt to send.
        rogue_string: The literal string a hijacked model emits (the trigger).
    """

    rogue_key: str
    prompt: str
    rogue_string: str


def _config_for(rogue_key: str) -> dict:
    # Byte-faithful to garak's load_data_all_attacks prompt_config.
    return {
        "base": {"text": prompt_data.ALL_OPENAI_PLAYGROUND_PROMPTS},
        "attack": {
            "text": tuple(
                prompt_data.goal_hikacking_attacks[a] for a in _ATTACKS
            ),
            "rogue_string": prompt_data.rogue_strings[rogue_key],
            "scoring": "match_rogue_to_output",
            "settings": {
                "escape": prompt_data.escape_chars["n"],
                "delimiter": prompt_data.delimiter_chars["dash"],
                "escape_length": 10,
                "escape_times": 2,
            },
        },
        "config": {"id": (0, 1, 2, 3)},
    }


def load_items(
    *,
    rogue_strings: tuple[str, ...] | list[str] | None = None,
    limit: int | None = 100,
    seed: int = 0,
) -> list[PromptInjectItem]:
    """Build the injection prompts across the requested rogue-string classes.

    Args:
        rogue_strings: Which classes to build (default all three).
        limit: Prompts per class. garak caps its default-active probe with a
            seeded shuffle then ``[-limit:]``; this reproduces that. ``None``
            keeps all (garak's uncapped path).
        seed: Shuffle seed, mirroring garak's ``self.seed``.

    Raises:
        ValueError: on an unknown rogue-string class or a non-positive limit.
    """
    chosen = tuple(rogue_strings) if rogue_strings is not None else ROGUE_STRINGS
    unknown = [r for r in chosen if r not in ROGUE_STRINGS]
    if unknown:
        raise ValueError(f"unknown rogue string(s): {unknown}; expected {list(ROGUE_STRINGS)}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit!r}")

    items: list[PromptInjectItem] = []
    for rogue_key in chosen:
        built = build_prompts(_config_for(rogue_key))
        prompts = [b["prompt"] for b in built]
        if limit is not None:
            # garak: random.seed(seed); shuffle; prompts[-limit:]
            rng = random.Random(seed)
            rng.shuffle(prompts)
            prompts = prompts[-limit:]
        rogue = prompt_data.rogue_strings[rogue_key]
        items.extend(
            PromptInjectItem(rogue_key=rogue_key, prompt=p, rogue_string=rogue)
            for p in prompts
        )
    return items
