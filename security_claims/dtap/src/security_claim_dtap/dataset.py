"""Dataset helpers for the DTAP claim: thin re-exports of the scaffold loader
plus the byte-faithfulness golden-hash manifest (build + load + per-task hash).

The DTAP per-task tree is NOT vendored in this package (it is large and licensed
separately; upstream auto-downloads it from HuggingFace, or point
``$DTAP_DATASET_ROOT`` at a local checkout). What this claim DOES ship is a
``data/golden_hashes.json`` manifest pinning the byte-identity of the two
faithfulness-critical pieces of each sampled task: the attacker objective
(``config.yaml`` ``Attack.malicious_goal``, which the Task turns into the Goal)
and the judge logic (``judge.py``, which decides success). A faithfulness test
re-hashes those task dirs and compares, so a future dataset change is caught.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import yaml
from dtap_scaffold.dataset import (
    TaskConfig,
    iter_task_config_paths,
    parse_task_config,
    resolve_dataset_root,
)

#: ``data/golden_hashes.json`` lives at the package root (sibling of ``src/``); it
#: is a dev/faithfulness manifest pinning external-dataset bytes, not runtime data.
GOLDEN_HASHES_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "golden_hashes.json"

#: Deterministic default sample for :func:`build_golden_hashes` (and the test): a
#: strided slice across one domain's sorted tasks so it spans benign + malicious.
GOLDEN_SAMPLE_DOMAINS: tuple[str, ...] = ("travel",)
GOLDEN_SAMPLE_SIZE: int = 16


def hash_task(task_dir: str | Path) -> str:
    """SHA-256 over a task's ``config.yaml`` ``malicious_goal`` + ``judge.py`` bytes.

    These are the two byte-identity-critical pieces of a DTAP task for this claim:
    the attacker objective the Task exposes as the Goal, and the judge that decides
    the outcome. A missing piece contributes empty bytes (benign tasks have no
    ``malicious_goal``; a task without a ``judge.py`` contributes none).
    """
    task_dir = Path(task_dir)
    cfg_path = task_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {} if cfg_path.is_file() else {}
    goal = (cfg.get("Attack") or {}).get("malicious_goal") or ""
    judge_path = task_dir / "judge.py"
    judge_bytes = judge_path.read_bytes() if judge_path.is_file() else b""

    digest = hashlib.sha256()
    digest.update(str(goal).encode("utf-8"))
    digest.update(judge_bytes)
    return digest.hexdigest()


def build_golden_hashes(
    root: str | Path,
    sample: int | None = GOLDEN_SAMPLE_SIZE,
    *,
    domains: Iterable[str] | None = None,
    write: bool = True,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Build a ``{relative_task_dir: hash_task(...)}`` manifest under *root*.

    *sample* caps the manifest size via a deterministic strided slice over the
    sorted task list (so it spans benign-then-malicious); ``None`` hashes every
    task. *domains* restricts which domains to enumerate (default
    :data:`GOLDEN_SAMPLE_DOMAINS`). When *write* is true the manifest is written
    to *path* (default :data:`GOLDEN_HASHES_PATH`).
    """
    root = Path(root)
    domain_list = list(domains) if domains is not None else list(GOLDEN_SAMPLE_DOMAINS)
    all_paths = sorted(iter_task_config_paths(root, domains=domain_list))
    if sample is not None and 0 < sample < len(all_paths):
        step = max(1, len(all_paths) // sample)
        all_paths = all_paths[::step][:sample]

    mapping: dict[str, str] = {}
    for config_path in all_paths:
        task_dir = config_path.parent
        rel = task_dir.relative_to(root).as_posix()
        mapping[rel] = hash_task(task_dir)

    if write:
        out = Path(path) if path is not None else GOLDEN_HASHES_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    return mapping


def load_golden_hashes(path: str | Path | None = None) -> dict[str, str]:
    """Load the committed golden-hash manifest (``{relative_task_dir: hash}``)."""
    p = Path(path) if path is not None else GOLDEN_HASHES_PATH
    return dict(json.loads(p.read_text()))


__all__ = [
    "TaskConfig",
    "resolve_dataset_root",
    "iter_task_config_paths",
    "parse_task_config",
    "GOLDEN_HASHES_PATH",
    "GOLDEN_SAMPLE_DOMAINS",
    "GOLDEN_SAMPLE_SIZE",
    "hash_task",
    "build_golden_hashes",
    "load_golden_hashes",
]
