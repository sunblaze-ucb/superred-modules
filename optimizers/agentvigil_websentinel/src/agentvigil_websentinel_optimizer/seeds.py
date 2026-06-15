"""Initial AgentVigil/WebSentinel seed templates.

The seed objects are loaded from vendored official files instead of being
hand-copied into this adapter. ``OFFICIAL_HTML_SEEDS`` mirrors the official
``new_seeds.py`` web/content corpus used by the AgentVigil/WebSentinel run
script; ``OFFICIAL_TEXT_SEEDS`` exposes the older official text corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Seed:
    id: str
    text: str
    mutation_method: str | None = None
    mutation_seed: str | list[str] | None = None
    score: float = 0.0
    performance: float = 0.0
    results: dict[str, int] = field(default_factory=dict)


from agentvigil_websentinel_optimizer.official_data import (  # noqa: E402
    load_official_html_seeds,
    load_official_text_seeds,
)

OFFICIAL_HTML_SEEDS: tuple[Seed, ...] = load_official_html_seeds()
OFFICIAL_TEXT_SEEDS: tuple[Seed, ...] = load_official_text_seeds()
DEFAULT_SEEDS: tuple[Seed, ...] = OFFICIAL_HTML_SEEDS


__all__ = ["DEFAULT_SEEDS", "OFFICIAL_HTML_SEEDS", "OFFICIAL_TEXT_SEEDS", "Seed"]
