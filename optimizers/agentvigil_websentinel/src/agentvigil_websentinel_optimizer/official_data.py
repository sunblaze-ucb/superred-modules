"""Load vendored official AgentVigil/WebSentinel literals and datasets."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentvigil_websentinel_optimizer.mutator import MutationMethod
    from agentvigil_websentinel_optimizer.seeds import Seed

_OFFICIAL_PACKAGE = "agentvigil_websentinel_optimizer"
_OFFICIAL_DIR = ("data", "official")
OFFICIAL_FILE_NAMES = frozenset(
    {
        "SAVE_RESUME_GUIDE.md",
        "adaptive_attack_data.json",
        "agent.py",
        "checkpoint_manager.py",
        "detection.py",
        "fuzzer.py",
        "mutate.py",
        "mutate_prompts.py",
        "new_seeds.py",
        "run.py",
        "run_with_resume.py",
        "seeds.py",
    }
)


@dataclass(frozen=True)
class OfficialFile:
    name: str
    content: str
    sha256: str


@lru_cache(maxsize=None)
def load_official_raw_file(name: str) -> OfficialFile:
    """Load a file copied verbatim from the official repository."""

    if name not in OFFICIAL_FILE_NAMES:
        raise ValueError(f"{name!r} is not a vendored official AgentVigil file")
    path = resources.files(_OFFICIAL_PACKAGE).joinpath(*_OFFICIAL_DIR, name)
    raw = path.read_bytes()
    return OfficialFile(
        name=name,
        content=raw.decode("utf-8"),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@lru_cache(maxsize=None)
def _parsed(name: str) -> ast.Module:
    return ast.parse(load_official_raw_file(name).content, filename=name)


def _assignment(module: ast.Module, target_name: str) -> ast.expr:
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == target_name:
                return stmt.value
    raise KeyError(f"{target_name!r} not found in official AgentVigil data")


def _seed_from_call(call: ast.Call) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        fields[keyword.arg] = ast.literal_eval(keyword.value)
    return fields


def _seed_rows(name: str, assignment: str) -> tuple[dict[str, Any], ...]:
    value = _assignment(_parsed(name), assignment)
    if not isinstance(value, ast.List):
        raise TypeError(f"official {assignment} must be a list")
    rows: list[dict[str, Any]] = []
    for item in value.elts:
        if not isinstance(item, ast.Call):
            raise TypeError(f"official {assignment} entries must be Seed(...) calls")
        rows.append(_seed_from_call(item))
    return tuple(rows)


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


@lru_cache(maxsize=1)
def load_official_html_seed_rows() -> tuple[dict[str, Any], ...]:
    return _seed_rows("new_seeds.py", "new_seeds")


@lru_cache(maxsize=1)
def load_official_text_seed_rows() -> tuple[dict[str, Any], ...]:
    return _seed_rows("seeds.py", "example_seeds")


@lru_cache(maxsize=1)
def load_official_html_seeds() -> tuple[Seed, ...]:
    return tuple(_seed_from_row(row) for row in load_official_html_seed_rows())


@lru_cache(maxsize=1)
def load_official_text_seeds() -> tuple[Seed, ...]:
    return tuple(_seed_from_row(row) for row in load_official_text_seed_rows())


@lru_cache(maxsize=1)
def load_official_system_prompt() -> str:
    value = ast.literal_eval(_assignment(_parsed("mutate_prompts.py"), "system_prompt"))
    if not isinstance(value, str):
        raise TypeError("official mutator system_prompt must be a string")
    return value


@lru_cache(maxsize=1)
def load_official_mutation_templates() -> dict[MutationMethod, str]:
    from agentvigil_websentinel_optimizer.mutator import MutationMethod

    value = _assignment(_parsed("mutate_prompts.py"), "mutation_prompt_templates")
    if not isinstance(value, ast.Dict):
        raise TypeError("official mutation_prompt_templates must be a dict")
    out: dict[MutationMethod, str] = {}
    name_map = {
        "Expand": MutationMethod.EXPAND,
        "Shorten": MutationMethod.SHORTEN,
        "Rephrase": MutationMethod.REPHRASE,
        "Crossover": MutationMethod.CROSSOVER,
        "GenerateSimilar": MutationMethod.GENERATE_SIMILAR,
    }
    for key, val in zip(value.keys, value.values, strict=True):
        if not isinstance(key, ast.Attribute):
            raise TypeError("official mutation template keys must be enum attributes")
        method = name_map[key.attr]
        template = ast.literal_eval(val)
        if not isinstance(template, str):
            raise TypeError("official mutation template values must be strings")
        out[method] = template
    return out


@lru_cache(maxsize=1)
def load_official_adaptive_attack_data() -> tuple[dict[str, Any], ...]:
    data = json.loads(load_official_raw_file("adaptive_attack_data.json").content)
    if not isinstance(data, list):
        raise TypeError("official adaptive_attack_data.json must be a list")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError("official adaptive_attack_data.json entries must be objects")
    return tuple(data)


__all__ = [
    "OfficialFile",
    "OFFICIAL_FILE_NAMES",
    "load_official_adaptive_attack_data",
    "load_official_html_seed_rows",
    "load_official_html_seeds",
    "load_official_mutation_templates",
    "load_official_raw_file",
    "load_official_system_prompt",
    "load_official_text_seed_rows",
    "load_official_text_seeds",
]
