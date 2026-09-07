"""Policy Puppetry templates and encoding, ported from NVIDIA garak.

Upstream: https://github.com/NVIDIA/garak — ``garak/probes/doctor.py`` and
``garak/resources/encodings.py`` (Apache-2.0). Upstream ``Bypass`` carries
**two** Dr. House scene templates and builds a prompt from each; both are
vendored byte-identically under ``data/upstream/`` (they contain triple-quote
sequences, so they are held as data rather than embedded in source).
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

#: Upstream ``Bypass.templates`` holds two scenes; both are used.
TEMPLATE_COUNT = 2

_TEMPLATE_FILES = ("bypass_template_0.txt", "bypass_template_1.txt")


@lru_cache(maxsize=TEMPLATE_COUNT)
def bypass_template(index: int = 0) -> str:
    """Return one vendored Dr. House scene template, verbatim from upstream.

    Args:
        index: ``0`` or ``1`` — upstream ``Bypass.templates`` order.

    Raises:
        ValueError: if *index* is not a valid template index.
    """
    if not 0 <= index < TEMPLATE_COUNT:
        raise ValueError(
            f"template index must be 0..{TEMPLATE_COUNT - 1}, got {index!r}"
        )
    return (
        resources.files("policy_puppetry_optimizer.data.upstream")
        .joinpath(_TEMPLATE_FILES[index])
        .read_text(encoding="utf-8")
    )


def dr_house_template() -> str:
    """The first Dr. House template (backwards-compatible alias)."""
    return bypass_template(0)


def render(
    behavior: str, leet_suffix: bool = False, template_index: int = 0
) -> str:
    """Fill a template's slot with *behavior*.

    Mirrors upstream ``Bypass._build_prompts()``, which loops over **every**
    template and substitutes the behaviour into each: a plain ``str.replace``
    of the slot. When *leet_suffix* is set, the slot first gains upstream
    ``BypassLeet``'s parenthetical aside.

    Args:
        behavior: The adversarial request to smuggle into the scene.
        leet_suffix: Append BypassLeet's "in code using leet speak" aside.
        template_index: Which upstream template to render (``0`` or ``1``).

    Returns:
        The rendered prompt, with no slot markers left.
    """
    template = bypass_template(template_index)
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
