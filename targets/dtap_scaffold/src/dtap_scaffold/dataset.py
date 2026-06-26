"""DTAP per-task dataset: root resolution, text-only enumeration, config parsing.

The per-task dataset (``config.yaml`` / ``setup.sh`` / ``judge.py`` / ``metadata/``)
is NOT shipped in the ``decodingtrust-agent-sdk`` wheel; upstream auto-downloads
it from HuggingFace (``AI-Secure/DecodingTrust-Agent-Platform``) on demand. This
module resolves the dataset root (a local override or an on-demand per-domain
download), enumerates the TEXT-ONLY tasks, and parses each ``config.yaml`` into a
:class:`TaskConfig` the claim and the targets consume.

Byte-faithfulness is enforced separately by the claim's golden-hash manifest over
the loaded ``config.yaml`` / ``judge.py`` bytes; pin ``DATASET_REVISION`` (or use
``DTAP_DATASET_ROOT`` pointing at a known checkout) for reproducible loads.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dtap_scaffold.text_domains import TEXT_ONLY_DOMAINS

DATASET_HF_REPO: str = "AI-Secure/DecodingTrust-Agent-Platform"
"""The HuggingFace dataset repo holding the per-task trees (repo_type='dataset')."""

DATASET_REVISION: str | None = None
"""Pinned HF revision for reproducible/byte-faithful loads. ``None`` = latest;
set to a commit SHA once chosen (the claim's golden hashes pin the bytes either
way). Prefer ``DTAP_DATASET_ROOT`` for fully offline, reproducible runs."""

DTAP_SDK_VERSION: str = "0.2.12"
"""The pinned decodingtrust-agent-sdk version this port targets."""


def resolve_dataset_root(
    domains: list[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    download: bool = False,
    quiet: bool = True,
) -> Path:
    """Return the dataset root, optionally downloading the requested domains.

    Resolution: explicit *root* arg > ``$DTAP_DATASET_ROOT`` > ``./dataset``.
    When *download* is true, the requested text-only *domains* (default: all
    text-only) are fetched from HuggingFace into the root if missing, mirroring
    the upstream loader (per-domain ``allow_patterns``, pinned to
    :data:`DATASET_REVISION`). Requires ``huggingface_hub`` (lazy import).
    """
    base = Path(
        root or os.environ.get("DTAP_DATASET_ROOT") or (Path.cwd() / "dataset")
    ).resolve()
    if not download:
        return base

    wanted = {d for d in (domains or TEXT_ONLY_DOMAINS) if d in TEXT_ONLY_DOMAINS}
    if base.is_dir():
        present = {p.name for p in base.iterdir() if p.is_dir()}
        wanted -= present
    if not wanted:
        return base

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only when downloading
        raise RuntimeError(
            "huggingface_hub is required to auto-download the DTAP dataset; install "
            "the [dataset] extra or set DTAP_DATASET_ROOT to a local checkout."
        ) from exc

    if not quiet:  # pragma: no cover - print path only
        print(
            f"[dtap] downloading {len(wanted)} dataset domain(s) "
            f"({', '.join(sorted(wanted))}) from huggingface.co/datasets/{DATASET_HF_REPO}"
        )
    base.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_HF_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=str(base),
        allow_patterns=[f"{d}/**" for d in sorted(wanted)],
    )
    return base


@dataclass(frozen=True)
class TaskConfig:
    """Parsed view of one DTAP per-task ``config.yaml`` (text-only domains).

    Identity / path facts come from the directory layout; the rest from the YAML.
    ``instructions`` is the benign ``task_instruction`` normalized to a list (the
    user-prompt baseline; empty for direct-threat tasks that have no benign
    cover). ``malicious_goal`` is the byte-identical attacker objective the claim
    turns into the ``Goal``. The byte-identical upstream ``attack_turns`` are NOT
    parsed here (the clean baseline carries no attack; replay lives in tests).
    """

    task_dir: Path
    domain: str
    type: str  # "benign" | "malicious"
    threat_model: str | None  # "direct" | "indirect" | None (benign)
    risk_category: str | None
    task_id: str
    system_prompt: str
    servers: tuple[str, ...]
    instructions: tuple[str, ...]
    malicious_goal: str | None
    available_injections: dict[str, Any] = field(default_factory=dict)
    env_injection_config: dict[str, Any] = field(default_factory=dict)


def _path_facts(
    config_path: Path, root: Path
) -> tuple[str, str, str | None, str | None, str]:
    """Derive (domain, type, threat_model, risk_category, task_id) from the layout.

    benign (flat):   <domain>/benign/<task_id>/config.yaml
    benign (nested): <domain>/benign/<risk_category>/<task_id>/config.yaml
    malicious:       <domain>/malicious/<threat_model>/<risk_category>/<task_id>/config.yaml
    """
    parts = config_path.parent.relative_to(root).parts
    domain = parts[0]
    task_id = parts[-1]
    if "malicious" in parts:
        i = parts.index("malicious")
        return domain, "malicious", parts[i + 1], parts[i + 2], task_id
    # benign
    risk = parts[2] if len(parts) >= 4 else None
    return domain, "benign", None, risk, task_id


def parse_task_config(
    config_path: str | os.PathLike[str], root: str | os.PathLike[str] | None = None
) -> TaskConfig:
    """Parse one ``config.yaml`` into a :class:`TaskConfig`.

    *root* anchors the path-fact derivation; if omitted it is inferred as the
    grandparent..-of the domain dir by locating the known ``benign``/``malicious``
    segment.
    """
    config_path = Path(config_path).resolve()
    if root is not None:
        root_p = Path(root).resolve()
    else:
        # Infer root as the parent of the <domain> dir (the segment before benign/malicious).
        parts = config_path.parts
        seg = "malicious" if "malicious" in parts else "benign"
        root_p = Path(*parts[: parts.index(seg) - 1])

    domain, ttype, threat_model, risk_category, task_id = _path_facts(
        config_path, root_p
    )

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    agent = cfg.get("Agent") or {}
    task = cfg.get("Task") or {}
    attack = cfg.get("Attack") or {}
    redteam = cfg.get("RedTeamingAgent") or {}

    servers = tuple(
        s["name"]
        for s in (agent.get("mcp_servers") or [])
        if isinstance(s, dict) and s.get("name") and s.get("enabled", True)
    )

    raw_instr = task.get("task_instruction")
    if isinstance(raw_instr, str):
        instructions: tuple[str, ...] = (raw_instr,)
    elif isinstance(raw_instr, list):
        instructions = tuple(str(x) for x in raw_instr)
    else:
        instructions = ()

    # Attack.risk_category is authoritative for malicious tasks when present.
    risk_category = attack.get("risk_category", risk_category)

    return TaskConfig(
        task_dir=config_path.parent,
        domain=domain,
        type=ttype,
        threat_model=attack.get("threat_model", threat_model),
        risk_category=risk_category,
        task_id=task_id,
        system_prompt=str(agent.get("system_prompt") or ""),
        servers=servers,
        instructions=instructions,
        malicious_goal=attack.get("malicious_goal"),
        available_injections=dict(redteam.get("available_injections") or {}),
        env_injection_config=dict(redteam.get("env_injection_config") or {}),
    )


def iter_task_config_paths(
    root: str | os.PathLike[str],
    *,
    domains: list[str] | None = None,
    types: list[str] | None = None,
    threat_models: list[str] | None = None,
) -> Iterator[Path]:
    """Yield ``config.yaml`` paths for TEXT-ONLY tasks under *root*, filtered.

    Non-text-only domains are always skipped. *types* filters benign/malicious;
    *threat_models* filters direct/indirect (malicious only).
    """
    root_p = Path(root).resolve()
    want_domains = {d for d in (domains or TEXT_ONLY_DOMAINS) if d in TEXT_ONLY_DOMAINS}
    for domain in sorted(want_domains):
        ddir = root_p / domain
        if not ddir.is_dir():
            continue
        for config_path in sorted(ddir.rglob("config.yaml")):
            _, ttype, tm, _, _ = _path_facts(config_path, root_p)
            if types is not None and ttype not in types:
                continue
            if threat_models is not None and (tm is None or tm not in threat_models):
                continue
            yield config_path


__all__ = [
    "DATASET_HF_REPO",
    "DATASET_REVISION",
    "DTAP_SDK_VERSION",
    "resolve_dataset_root",
    "TaskConfig",
    "parse_task_config",
    "iter_task_config_paths",
]
