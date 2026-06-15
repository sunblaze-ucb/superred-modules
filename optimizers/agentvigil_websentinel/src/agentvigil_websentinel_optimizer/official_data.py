"""Load official AgentVigil/WebSentinel data and literals."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from agentvigil_websentinel_optimizer.mutator import MutationMethod
    from agentvigil_websentinel_optimizer.seeds import Seed

_OFFICIAL_PACKAGE = "agentvigil_websentinel_optimizer"
_OFFICIAL_DIR = ("data", "official")
OFFICIAL_FILE_NAMES = frozenset(
    {
        "README.md",
        "mutation_prompts.json",
        "new_seeds.json",
        "text_seeds.json",
    }
)


@lru_cache(maxsize=None)
def load_official_raw_file(name: str) -> str:
    """Return the UTF-8 text of a packaged official data/literal artifact."""

    if name not in OFFICIAL_FILE_NAMES:
        raise ValueError(f"{name!r} is not a packaged AgentVigil data artifact")
    path = resources.files(_OFFICIAL_PACKAGE).joinpath(*_OFFICIAL_DIR, name)
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _json_artifact(name: str) -> dict[str, Any]:
    data = json.loads(load_official_raw_file(name))
    if not isinstance(data, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return cast(dict[str, Any], data)


def _seed_from_row(row: dict[str, Any]) -> Seed:
    from agentvigil_websentinel_optimizer.seeds import Seed

    return Seed(
        id=str(row["id"]),
        text=str(row["text"]),
        mutation_method=row.get("mutation_method"),
        mutation_seed=row.get("mutation_seed"),
        score=float(row.get("score", 0.0)),
        performance=float(row.get("performance", 0.0)),
    )


def _seed_rows(name: str) -> tuple[dict[str, Any], ...]:
    rows = _json_artifact(name).get("seeds")
    if not isinstance(rows, list):
        raise TypeError(f"{name} must contain a seeds list")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{name} seed entries must be objects")
    return tuple(cast(list[dict[str, Any]], rows))


@lru_cache(maxsize=1)
def load_official_html_seed_rows() -> tuple[dict[str, Any], ...]:
    return _seed_rows("new_seeds.json")


@lru_cache(maxsize=1)
def load_official_text_seed_rows() -> tuple[dict[str, Any], ...]:
    return _seed_rows("text_seeds.json")


@lru_cache(maxsize=1)
def load_official_html_seeds() -> tuple[Seed, ...]:
    return tuple(_seed_from_row(row) for row in load_official_html_seed_rows())


@lru_cache(maxsize=1)
def load_official_text_seeds() -> tuple[Seed, ...]:
    return tuple(_seed_from_row(row) for row in load_official_text_seed_rows())


@lru_cache(maxsize=1)
def load_official_system_prompt() -> str:
    value = _json_artifact("mutation_prompts.json").get("system_prompt")
    if not isinstance(value, str):
        raise TypeError("mutation_prompts.json must contain a system_prompt string")
    return value


@lru_cache(maxsize=1)
def load_official_mutation_templates() -> dict[MutationMethod, str]:
    from agentvigil_websentinel_optimizer.mutator import MutationMethod

    value = _json_artifact("mutation_prompts.json").get("mutation_prompt_templates")
    if not isinstance(value, dict):
        raise TypeError("mutation_prompts.json must contain mutation_prompt_templates")
    name_map = {
        "Expand": MutationMethod.EXPAND,
        "Shorten": MutationMethod.SHORTEN,
        "Rephrase": MutationMethod.REPHRASE,
        "Crossover": MutationMethod.CROSSOVER,
        "GenerateSimilar": MutationMethod.GENERATE_SIMILAR,
    }
    out: dict[MutationMethod, str] = {}
    for name, template in cast(dict[str, Any], value).items():
        if not isinstance(template, str):
            raise TypeError("official mutation template values must be strings")
        out[name_map[name]] = template
    return out


def load_official_source_hash(artifact_name: str) -> str:
    value = _json_artifact(artifact_name).get("source_sha256")
    if not isinstance(value, str):
        raise TypeError(f"{artifact_name} must contain a source_sha256 string")
    return value


__all__ = [
    "OFFICIAL_FILE_NAMES",
    "load_official_html_seed_rows",
    "load_official_html_seeds",
    "load_official_mutation_templates",
    "load_official_raw_file",
    "load_official_source_hash",
    "load_official_system_prompt",
    "load_official_text_seed_rows",
    "load_official_text_seeds",
]
