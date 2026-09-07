"""DAN persona corpus, vendored from NVIDIA garak.

Upstream: https://github.com/NVIDIA/garak — ``garak/data/dan/*.json`` loaded by
``garak/probes/dan.py`` (Apache-2.0). The JSON files under ``data/upstream/``
are byte-identical copies; ``scripts/sync_upstream.py`` refreshes them.

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

#: garak's placeholder for the victim model's name, substituted by its
#: ``DANProbeMeta.probe()`` before send. Prompts without it are sent as-is.
GENERATOR_NAME_PLACEHOLDER = "{generator.name}"

#: Persona order is the upstream file order (alphabetical), so a run index maps
#: to a stable persona across versions.
PERSONA_NAMES: tuple[str, ...] = (
    "AntiDAN",
    "ChatGPT_Developer_Mode_RANTI",
    "ChatGPT_Developer_Mode_v2",
    "ChatGPT_Image_Markdown",
    "DAN_Jailbreak",
    "DUDE",
    "Dan_10_0",
    "Dan_11_0",
    "Dan_6_0",
    "Dan_6_2",
    "Dan_7_0",
    "Dan_8_0",
    "Dan_9_0",
    "STAN",
)


@dataclass(frozen=True)
class Persona:
    """One DAN persona preamble.

    Attributes:
        name: Upstream probe/file name, e.g. ``Dan_11_0``.
        preamble: The jailbreak text, verbatim from upstream.
    """

    name: str
    preamble: str

    def render(self, generator_name: str | None = None) -> str:
        """Fill garak's ``{generator.name}`` placeholder.

        Mirrors upstream: substitution happens only when the placeholder is
        present; otherwise the prompt is returned unchanged.

        Args:
            generator_name: Victim model identity. When ``None`` and the
                preamble needs one, ``"ChatGPT"`` is used — the assistant the
                upstream prompts were written against.
        """
        if GENERATOR_NAME_PLACEHOLDER not in self.preamble:
            return self.preamble
        return self.preamble.replace(
            GENERATOR_NAME_PLACEHOLDER, generator_name or "ChatGPT"
        )


def load_personas(names: tuple[str, ...] | list[str] | None = None) -> list[Persona]:
    """Load persona preambles from the vendored upstream JSON.

    Args:
        names: Subset to load, in the order given. Defaults to
            :data:`PERSONA_NAMES`.

    Returns:
        One :class:`Persona` per prompt, in file order. A file holding several
        prompts contributes one persona per prompt, suffixed ``#n``.

    Raises:
        ValueError: If a requested name is not vendored.
    """
    wanted = tuple(names) if names is not None else PERSONA_NAMES
    unknown = [n for n in wanted if n not in PERSONA_NAMES]
    if unknown:
        raise ValueError(
            f"unknown persona(s): {unknown}; expected any of {list(PERSONA_NAMES)}"
        )

    root = resources.files("dan_personas_optimizer.data.upstream")
    personas: list[Persona] = []
    for name in wanted:
        raw = json.loads(root.joinpath(f"{name}.json").read_text(encoding="utf-8"))
        prompts = [str(p).strip() for p in raw if p]
        for index, prompt in enumerate(prompts):
            label = name if len(prompts) == 1 else f"{name}#{index}"
            personas.append(Persona(name=label, preamble=prompt))
    return personas


#: Upstream ``Ablation_Dan_11_0`` composes DAN 11.0 from seven optional
#: segments, emitting one prompt per non-empty bitmask (i = 1..127).
ABLATION_SEGMENTS_FILE = "ablation_dan_11_0_segments.json"
ABLATION_PROMPT_COUNT = 127

#: Upstream ``DanInTheWild`` -- ~700 real jailbreaks collected in the wild.
IN_THE_WILD_FILE = "inthewild_jailbreak_llms.json"


def load_ablation_personas() -> list[Persona]:
    """Reconstruct upstream ``Ablation_Dan_11_0``'s 127 prompts.

    Upstream builds these in code: seven optional segments keyed to bits 1..64
    plus four unconditional ones, concatenated in a fixed order for every
    non-empty bitmask. The segments are vendored verbatim (extracted from the
    probe's own literals) and reassembled here with upstream's exact ordering
    and bit tests, so the 127 prompts are identical.
    """
    raw = (
        resources.files("dan_personas_optimizer.data.upstream")
        .joinpath(ABLATION_SEGMENTS_FILE)
        .read_text(encoding="utf-8")
    )
    segments = json.loads(raw)
    out: list[Persona] = []
    for i in range(1, ABLATION_PROMPT_COUNT + 1):
        prompt = "".join(
            seg["text"] for seg in segments if seg["bit"] == 0 or (seg["bit"] & i)
        )
        out.append(Persona(name=f"Ablation_Dan_11_0#{i}", preamble=prompt))
    return out


def load_in_the_wild_personas(limit: int | None = None) -> list[Persona]:
    """Load upstream ``DanInTheWild``'s corpus of real-world jailbreaks.

    Args:
        limit: Keep at most this many, from the start of the file. Upstream's
            default-active ``DanInTheWild`` prunes to its prompt cap; the
            uncapped ``DanInTheWildFull`` is ``limit=None``.
    """
    raw = (
        resources.files("dan_personas_optimizer.data.upstream")
        .joinpath(IN_THE_WILD_FILE)
        .read_text(encoding="utf-8")
    )
    prompts = json.loads(raw)
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        prompts = prompts[:limit]
    return [
        Persona(name=f"DanInTheWild#{i}", preamble=str(text))
        for i, text in enumerate(prompts)
    ]
