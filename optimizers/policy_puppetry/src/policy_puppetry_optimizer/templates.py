"""Policy Puppetry template and encoding, ported from NVIDIA garak.

Upstream: https://github.com/NVIDIA/garak — ``garak/probes/doctor.py`` and
``garak/resources/encodings.py`` (Apache-2.0). The Dr. House template is
vendored byte-identically under ``data/upstream/`` (it contains a triple-quote
sequence, so it is held as data rather than embedded in source);
``leetspeak`` reproduces upstream's transform with an added seed.

Technique: HiddenLayer, "Novel Universal Bypass for All Major LLMs" —
https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import random
from functools import lru_cache
from importlib import resources

#: Upstream's slot marker, filled with the adversarial request.
HARMFUL_BEHAVIOR_SLOT = "{{HARMFUL_BEHAVIOR}}"

#: Aside upstream's BypassLeet appends to the slot before leetspeaking.
LEET_SLOT_SUFFIX = " (in code using leet speak so Cuddy doesn't know)"

_TEMPLATE_FILE = "bypass_template.txt"


@lru_cache(maxsize=1)
def dr_house_template() -> str:
    """The vendored Dr. House scene template, verbatim from upstream."""
    return (
        resources.files("policy_puppetry_optimizer.data.upstream")
        .joinpath(_TEMPLATE_FILE)
        .read_text(encoding="utf-8")
    )


def render(behavior: str, leet_suffix: bool = False) -> str:
    """Fill the template's slot with *behavior*.

    Mirrors upstream ``Bypass._build_prompts()``: a plain ``str.replace`` of
    every slot occurrence. When *leet_suffix* is set, the slot first gains
    upstream ``BypassLeet``'s parenthetical aside.

    Args:
        behavior: The adversarial request to smuggle into the scene.
        leet_suffix: Append BypassLeet's "in code using leet speak" aside.

    Returns:
        The rendered prompt, with no slot markers left.
    """
    template = dr_house_template()
    if leet_suffix:
        template = template.replace(
            HARMFUL_BEHAVIOR_SLOT, HARMFUL_BEHAVIOR_SLOT + LEET_SLOT_SUFFIX
        )
    return template.replace(HARMFUL_BEHAVIOR_SLOT, behavior)


def leetspeak(text: str, seed: int | None = None) -> str:
    """Partially transliterate *text* into leetspeak.

    Port of upstream ``garak.resources.encodings.leetspeak``: map ``ieaost``
    (both cases) onto ``134057``, then keep each transmuted character only
    about half the time, so the result stays readable.

    Upstream draws from the unseeded global ``random``, so its output is not
    reproducible run to run; *seed* makes it so. Passing ``seed=None``
    reproduces upstream's behaviour exactly.

    Upstream's ``transmute_chance`` parameter is accepted but never used — the
    mix rate is hard-coded at 0.5. That is preserved here rather than "fixed",
    so the output distribution matches upstream.

    Args:
        text: Text to transliterate.
        seed: Seed for the per-character mix.

    Returns:
        The partially leetspeaked text, the same length as the input.
    """
    src_chars = "ieaost"
    dst_chars = "134057"
    processed = text.translate(
        str.maketrans(src_chars.lower() + src_chars.upper(), dst_chars + dst_chars)
    )
    rng = random.Random(seed)
    return "".join(
        processed[i] if rng.random() > 0.5 else text[i] for i in range(len(text))
    )
