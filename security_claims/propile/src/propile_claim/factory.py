"""Factories for the ProPILE PII-leakage claim.

``propile_claim(...)`` builds a :class:`SecurityClaim` of one
:class:`~propile_claim.task.PIILeakTask` per constructed probe item. Named
wrappers cut it by probe type. All default to the bundled synthetic PII sample;
pass ``pii_data_path`` for your own JSONL (data you have a lawful basis to test).

* :func:`propile_twin_claim` — name -> PII (the only type the upstream/bundled
  data fully supports);
* :func:`propile_triplet_claim` / :func:`propile_quadruplet_claim` — name + aux
  PII -> another (need records with the auxiliary fields);
* :func:`propile_unstructured_claim` — relationship/affiliation elicitation.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from chatbot_target import ChatbotTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from propile_claim.judge import PIILeakConfig
from propile_claim.loader import load_pii_records, load_templates
from propile_claim.probes import PROBE_BUILDERS
from propile_claim.task import PIILeakTask

_ALL_PROBE_TYPES = ("twin", "triplet", "quadruplet", "unstructured")


def propile_claim(
    *,
    probe_types: Iterable[str] | None = None,
    pii_data_path: str | Path | None = None,
    exact_only: bool = False,
    config: PIILeakConfig | None = None,
    system_prompt: str | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[ChatbotTarget]:
    """Build a ProPILE claim over the selected probe types.

    Args:
        probe_types: subset of ``{"twin", "triplet", "quadruplet", "unstructured"}``;
            default all four.
        pii_data_path: JSONL of PII records; default the bundled synthetic sample.
        exact_only: use the strict exact-substring judge (``PIILeakExact``).
        config: detector configuration; default garak's (case-insensitive, fuzzy on).
        system_prompt: override the target system prompt, or ``None`` to leave it.
        success_threshold: leak score at/above which a run is a success.

    Raises:
        ValueError: if ``probe_types`` contains an unknown type.
    """
    chosen = tuple(probe_types) if probe_types is not None else _ALL_PROBE_TYPES
    unknown = [p for p in chosen if p not in PROBE_BUILDERS]
    if unknown:
        raise ValueError(f"unknown probe types {unknown}; expected subset of {_ALL_PROBE_TYPES}")

    records = load_pii_records(pii_data_path)
    templates = load_templates()

    tasks: list[Task[ChatbotTarget]] = []
    _build_tasks_into(
        tasks, chosen, records, templates, exact_only, config, system_prompt, success_threshold
    )
    if not tasks:
        raise ValueError(
            f"no probe items generated for probe types {list(chosen)}: the PII records "
            "need a 'name' plus the relevant fields (e.g. email/phone for twin, or "
            "auxiliary fields for triplet/quadruplet/unstructured). Provide richer data "
            "via pii_data_path — a SecurityClaim cannot be empty."
        )
    return SecurityClaim.from_tasks(tasks)


def _build_tasks_into(
    tasks: list[Task[ChatbotTarget]],
    chosen: tuple[str, ...],
    records: list[dict[str, str]],
    templates: dict[str, list[str]],
    exact_only: bool,
    config: PIILeakConfig | None,
    system_prompt: str | None,
    success_threshold: float,
) -> None:
    for probe_type in chosen:
        items = PROBE_BUILDERS[probe_type](records, templates)
        for index, item in enumerate(items):
            tasks.append(
                PIILeakTask(
                    prompt=item.prompt,
                    trigger=item.trigger,
                    pii_type=item.pii_type,
                    probe_type=probe_type,
                    item_index=index,
                    exact_only=exact_only,
                    config=config,
                    system_prompt=system_prompt,
                    success_threshold=success_threshold,
                )
            )


def propile_twin_claim(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """Twin probes only (name -> PII) — the data-backed default axis."""
    return propile_claim(probe_types=["twin"], **kwargs)  # type: ignore[arg-type]


def propile_triplet_claim(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """Triplet probes only (name + one auxiliary PII -> another)."""
    return propile_claim(probe_types=["triplet"], **kwargs)  # type: ignore[arg-type]


def propile_quadruplet_claim(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """Quadruplet probes only (name + two auxiliary PIIs -> the third)."""
    return propile_claim(probe_types=["quadruplet"], **kwargs)  # type: ignore[arg-type]


def propile_unstructured_claim(**kwargs: object) -> SecurityClaim[ChatbotTarget]:
    """Unstructured probes only (relationship / affiliation elicitation)."""
    return propile_claim(probe_types=["unstructured"], **kwargs)  # type: ignore[arg-type]


__all__ = [
    "propile_claim",
    "propile_twin_claim",
    "propile_triplet_claim",
    "propile_quadruplet_claim",
    "propile_unstructured_claim",
]
