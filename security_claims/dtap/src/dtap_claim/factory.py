"""Factory functions building hierarchical DTAP security claims, plus the
convenience target factories that wire either DTAP agent target.

Top factory :func:`dtap_claim` enumerates one :class:`DtapTask` per DTAP per-task
``config.yaml`` under the dataset root, with filters by domain / type / threat
model / risk category. Convenience roll-ups cover the natural DTAP axes (per
domain, direct vs indirect threat model, benign, per risk category). The two
target factories (:func:`dtap_claudecode_target_factory`,
:func:`dtap_openclaw_target_factory`) lazily import the concrete target classes
so this claim package depends only on ``superred`` + ``dtap-scaffold`` (the target
packages carry the heavier Docker/agent-SDK deps).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import cast

from superred.core.controller import TargetFactory
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task

from dtap_claim.dataset import (
    iter_task_config_paths,
    parse_task_config,
    resolve_dataset_root,
)
from dtap_claim.task import DtapTask


def _as_list(values: Iterable[str] | None) -> list[str] | None:
    return list(values) if values is not None else None


def dtap_claim(
    *,
    domains: Iterable[str] | None = None,
    types: Iterable[str] | None = None,
    threat_models: Iterable[str] | None = None,
    risk_categories: Iterable[str] | None = None,
    dataset_root: str | None = None,
    download: bool = False,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
    judge_api_key: str | None = None,
    always_on_servers: Sequence[str] = (),
) -> SecurityClaim[Target]:
    """Build a DTAP claim: one :class:`DtapTask` per matching per-task config.

    Args:
        domains: restrict to these DTAP domains (default: all text-only domains).
        types: restrict to ``"benign"`` / ``"malicious"`` (default: both).
        threat_models: restrict malicious tasks to ``"direct"`` / ``"indirect"``.
        risk_categories: restrict to these ``Attack.risk_category`` values.
        dataset_root: local dataset root override (else ``$DTAP_DATASET_ROOT``
            then ``./dataset``); see ``dtap_claim.dataset.resolve_dataset_root``.
        download: when true, auto-download the requested domains from HuggingFace
            if missing (needs ``huggingface_hub``); default off (fully offline).
        judge_model / judge_api_base / judge_api_key: out-of-band DTAP judge LLM
            (passed through to ``run_dtap_judge``; the judge re-queries the live
            env and its cost is out of band, never the optimizer's budget).
        always_on_servers: extra MCP env servers unioned into EVERY task's active
            set, on top of the task's own (a "generalist assistant" bundle). Order
            reaches the agent's tool ordering; use canonical ``mcp.yaml`` casing
            (e.g. ``"OS-filesystem"``). Empty (default) is upstream-faithful.

    Raises:
        ValueError: if no task matches the filters.
    """
    domain_list = _as_list(domains)
    type_list = _as_list(types)
    threat_model_list = _as_list(threat_models)
    risk_set = set(risk_categories) if risk_categories is not None else None

    always_on = tuple(always_on_servers)  # re-iterable: consumed once per task

    root = resolve_dataset_root(domain_list, root=dataset_root, download=download)

    tasks: list[DtapTask] = []
    for config_path in iter_task_config_paths(
        root,
        domains=domain_list,
        types=type_list,
        threat_models=threat_model_list,
    ):
        tc = parse_task_config(config_path, root=root)
        if risk_set is not None and tc.risk_category not in risk_set:
            continue
        if always_on:
            tc = replace(tc, servers=tuple(dict.fromkeys([*tc.servers, *always_on])))
        tasks.append(
            DtapTask(
                task_config=tc,
                judge_model=judge_model,
                judge_api_base=judge_api_base,
                judge_api_key=judge_api_key,
            )
        )

    if not tasks:
        raise ValueError("dtap_claim produced no tasks for the given filters")
    return SecurityClaim.from_tasks(cast("list[Task[Target]]", tasks))


# -- Convenience roll-ups ----------------------------------------------------


def dtap_domain_claim(domain: str, **kwargs: object) -> SecurityClaim[Target]:
    """All DTAP tasks (benign + malicious) for a single domain."""
    return dtap_claim(domains=[domain], **kwargs)  # type: ignore[arg-type]


def dtap_direct_claim(**kwargs: object) -> SecurityClaim[Target]:
    """All malicious tasks under the ``direct`` threat model (the user is adversarial)."""
    return dtap_claim(types=["malicious"], threat_models=["direct"], **kwargs)  # type: ignore[arg-type]


def dtap_indirect_claim(**kwargs: object) -> SecurityClaim[Target]:
    """All malicious tasks under the ``indirect`` threat model (env/third-party injection)."""
    return dtap_claim(types=["malicious"], threat_models=["indirect"], **kwargs)  # type: ignore[arg-type]


def dtap_benign_claim(**kwargs: object) -> SecurityClaim[Target]:
    """All benign tasks (legitimate-utility baseline; primary == task_success)."""
    return dtap_claim(types=["benign"], **kwargs)  # type: ignore[arg-type]


def dtap_risk_claim(risk: str, **kwargs: object) -> SecurityClaim[Target]:
    """All tasks for a single ``Attack.risk_category`` (e.g. ``data-exfiltration``)."""
    return dtap_claim(risk_categories=[risk], **kwargs)  # type: ignore[arg-type]


def dtap_combined_claim(claims: list[SecurityClaim[Target]]) -> SecurityClaim[Target]:
    """Compose several DTAP sub-claims into one (lazy chaining)."""
    return SecurityClaim.from_claims(claims)


# -- Target wiring convenience ----------------------------------------------


def dtap_claudecode_target_factory(
    *,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    concurrency: int = 1,
    state_root: str | None = None,
    bedrock: bool = False,
) -> TargetFactory:
    """A ``TargetFactory`` for the Claude Code DTAP target.

    The target class is imported lazily inside ``create`` so this claim package
    needs only ``superred`` + ``dtap-scaffold``. ``model`` is the agent's own
    inference model (a construction concern run through the target's own client,
    NOT the optimizer's budget-locked ``LLMClient``). ``concurrency`` is how many
    isolated env+agent instances run in parallel (each task gets a fresh target);
    keep it at 1 unless the host can run several Docker env stacks at once.
    ``bedrock=True`` runs the agent on AWS Bedrock (``model`` is then an inference
    profile id, and ``api_base``/``api_key`` are unused).
    """

    def create() -> Target:
        from dtap_claudecode_target import DtapClaudeCodeTarget

        target: Target = DtapClaudeCodeTarget(
            model=model,
            api_base=api_base,
            api_key=api_key,
            state_root=state_root,
            bedrock=bedrock,
        )
        return target

    return TargetFactory(create=create, concurrency=concurrency)


def dtap_openclaw_target_factory(
    *,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    concurrency: int = 1,
    state_root: str | None = None,
) -> TargetFactory:
    """A ``TargetFactory`` for the OpenClaw DTAP target (see
    :func:`dtap_claudecode_target_factory` for the shared rationale)."""

    def create() -> Target:
        from dtap_openclaw_target import DtapOpenClawTarget

        target: Target = DtapOpenClawTarget(
            model=model, api_base=api_base, api_key=api_key, state_root=state_root
        )
        return target

    return TargetFactory(create=create, concurrency=concurrency)


__all__ = [
    "dtap_claim",
    "dtap_domain_claim",
    "dtap_direct_claim",
    "dtap_indirect_claim",
    "dtap_benign_claim",
    "dtap_risk_claim",
    "dtap_combined_claim",
    "dtap_claudecode_target_factory",
    "dtap_openclaw_target_factory",
]
