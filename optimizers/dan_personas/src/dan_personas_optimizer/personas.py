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
