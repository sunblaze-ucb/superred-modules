"""DTAP per-task dataset for the claim: root resolution, text-only enumeration,
config parsing, and the byte-faithfulness golden-hash manifest.

Defining the adversarial Goals is the SecurityClaim's job, so this module -- the
dataset/goal enumeration -- lives here, not in the target. The per-task dataset
(``config.yaml`` / ``setup.sh`` / ``judge.py`` / ``metadata/``) is NOT shipped in
the ``decodingtrust-agent-sdk`` wheel; upstream auto-downloads it from HuggingFace
(``AI-Secure/DecodingTrust-Agent-Platform``). This module resolves the dataset
root (a local override or an on-demand per-domain download), enumerates the
TEXT-ONLY tasks (restricted to the domains the target can run, per the target's
:mod:`dtap_scaffold.text_domains` capability list), and parses each ``config.yaml``
into a :class:`TaskConfig` the claim turns into Goals and per-task target config.

It also ships ``data/golden_hashes.json`` pinning the byte-identity of the two
faithfulness-critical pieces of each sampled task: the objective the Task turns
into the ``Goal`` (``config.yaml`` ``Attack.malicious_goal`` for a malicious task,
or the first ``Task.task_instruction`` for a benign task) and the judge logic
(``judge.py``, which decides success). A faithfulness test re-hashes those dirs
and compares, so a future dataset change is caught. Pin ``DATASET_REVISION`` (or
use ``DTAP_DATASET_ROOT`` pointing at a known checkout) for reproducible loads.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dtap_scaffold.text_domains import TEXT_ONLY_DOMAINS

# ---------------------------------------------------------------------------
# Dataset source + root resolution
# ---------------------------------------------------------------------------

DATASET_HF_REPO: str = "AI-Secure/DecodingTrust-Agent-Platform"
"""The HuggingFace dataset repo holding the per-task trees (repo_type='dataset')."""

DATASET_REVISION: str | None = None
"""Pinned HF revision for reproducible/byte-faithful loads. ``None`` = latest;
set to a commit SHA once chosen (the golden hashes pin the bytes either way).
Prefer ``DTAP_DATASET_ROOT`` for fully offline, reproducible runs."""

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
    base = Path(root or os.environ.get("DTAP_DATASET_ROOT") or (Path.cwd() / "dataset")).resolve()
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


# ---------------------------------------------------------------------------
# Per-task config parsing + enumeration
# ---------------------------------------------------------------------------


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
    #: Per-server ``env_vars`` from ``Agent.mcp_servers`` (``{server: {VAR: value}}``),
    #: upstream's HIGHEST-priority env tier (``mcp_helpers.task_env_overrides``). Sets
    #: the acting-user identity (``USER_ACCESS_TOKEN``) + per-task credentials the
    #: env-state judge verifies against; empty for servers that declare none.
    server_env_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Per-server ``tool_blacklist`` from ``Agent.mcp_servers`` (``{server: (tool,
    #: ...)}``): tools this task hides from the agent's catalogue. Empty when none.
    tool_blacklist: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: DTAP ``Attack.additional_information``: per-task attacker prerequisite facts
    #: (e.g. the victim inbox an injected email must target + the whitelisted attacker
    #: sender accounts). For an indirect email-injection task the victim inbox lives
    #: ONLY here, so the claim surfaces it to an in-scope optimizer. Empty when absent.
    additional_information: str = ""


def _path_facts(config_path: Path, root: Path) -> tuple[str, str, str | None, str | None, str]:
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

    domain, ttype, threat_model, risk_category, task_id = _path_facts(config_path, root_p)

    with config_path.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    agent = cfg.get("Agent") or {}
    task = cfg.get("Task") or {}
    attack = cfg.get("Attack") or {}
    redteam = cfg.get("RedTeamingAgent") or {}

    mcp_servers = [
        s
        for s in (agent.get("mcp_servers") or [])
        if isinstance(s, dict) and s.get("name") and s.get("enabled", True)
    ]
    servers = tuple(s["name"] for s in mcp_servers)
    # Per-server env_vars (upstream's top env tier: acting identity + per-task creds)
    # and tool_blacklist, keyed by server name; only non-empty entries are kept.
    server_env_overrides = {
        s["name"]: {str(k): str(v) for k, v in (s.get("env_vars") or {}).items()}
        for s in mcp_servers
        if s.get("env_vars")
    }
    tool_blacklist = {
        s["name"]: tuple(str(t) for t in (s.get("tool_blacklist") or []))
        for s in mcp_servers
        if s.get("tool_blacklist")
    }

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
        server_env_overrides=server_env_overrides,
        tool_blacklist=tool_blacklist,
        additional_information=str(attack.get("additional_information") or ""),
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


# ---------------------------------------------------------------------------
# Byte-faithfulness golden-hash manifest
# ---------------------------------------------------------------------------

#: ``data/golden_hashes.json`` lives at the package root (sibling of ``src/``); it
#: is a dev/faithfulness manifest pinning external-dataset bytes, not runtime data.
GOLDEN_HASHES_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "golden_hashes.json"

#: Deterministic default sample for :func:`build_golden_hashes` (and the test): a
#: strided slice across one domain's sorted tasks so it spans benign + malicious.
GOLDEN_SAMPLE_DOMAINS: tuple[str, ...] = ("travel",)
GOLDEN_SAMPLE_SIZE: int = 16


def hash_task(task_dir: str | Path) -> str:
    """SHA-256 over the bytes a DTAP task exposes as the Goal + its ``judge.py``.

    These are the two byte-identity-critical pieces of a DTAP task for this claim:
    the objective the Task turns into the ``Goal``, and the judge that decides the
    outcome. The Goal bytes are the attacker objective (``config.yaml``
    ``Attack.malicious_goal``) for a malicious task; when there is no
    ``malicious_goal`` (a benign task) they are the first ``Task.task_instruction``
    -- exactly the string the Task exposes as the benign Goal
    (``TaskConfig.instructions[0]``), so benign-goal drift is caught too. A missing
    piece contributes empty bytes (a task with neither goal source, or without a
    ``judge.py``, contributes none).
    """
    task_dir = Path(task_dir)
    cfg_path = task_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {} if cfg_path.is_file() else {}
    goal = (cfg.get("Attack") or {}).get("malicious_goal") or ""
    if not goal:
        # Benign task: pin the first task_instruction, the exact bytes the Task
        # uses as the Goal. Mirror parse_task_config's normalization of
        # ``Task.task_instruction`` into ``instructions[0]`` so the hashed bytes
        # equal the Goal bytes.
        raw_instr = (cfg.get("Task") or {}).get("task_instruction")
        if isinstance(raw_instr, str):
            goal = raw_instr
        elif isinstance(raw_instr, list) and raw_instr:
            goal = str(raw_instr[0])
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
    "DATASET_HF_REPO",
    "DATASET_REVISION",
    "DTAP_SDK_VERSION",
    "resolve_dataset_root",
    "TaskConfig",
    "parse_task_config",
    "iter_task_config_paths",
    "GOLDEN_HASHES_PATH",
    "GOLDEN_SAMPLE_DOMAINS",
    "GOLDEN_SAMPLE_SIZE",
    "hash_task",
    "build_golden_hashes",
    "load_golden_hashes",
]
