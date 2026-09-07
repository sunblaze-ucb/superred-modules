"""Load the vendored CyberSecEval prompt-injection datasets.

Upstream (MIT): PurpleLlama
``CybersecurityBenchmarks/datasets/prompt_injection/*.json``. Each record pairs
a defended ``test_case_prompt`` (the system prompt), a ``user_input``
(the injection), and a ``judge_question`` used to score the response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

_DATASETS = {
    "english": "prompt_injection.json",
    "multilingual": "prompt_injection_multilingual_machine_translated.json",
}
DATASET_NAMES: tuple[str, ...] = tuple(_DATASETS)
_REQUIRED_FIELDS = ("test_case_prompt", "user_input", "judge_question")


@dataclass(frozen=True)
class InjectionCase:
    """One prompt-injection test case.

    Attributes:
        prompt_id: Upstream record id.
        system_prompt: ``test_case_prompt`` — the defended setup.
        user_input: The injection to send.
        judge_question: What the judge checks about the response.
        injection_type: e.g. ``"direct"`` / ``"indirect"``.
        risk_category: e.g. ``"security-violating"``.
        language: ``speaking_language``.
    """

    prompt_id: int
    system_prompt: str
    user_input: str
    judge_question: str
    injection_type: str
    risk_category: str
    language: str


def load_cases(
    *,
    dataset: str = "english",
    injection_types: tuple[str, ...] | list[str] | None = None,
    risk_categories: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> list[InjectionCase]:
    """Return injection cases from a vendored dataset, optionally filtered.

    Args:
        dataset: ``"english"`` (251 cases) or ``"multilingual"`` (1004).
        injection_types: keep only these ``injection_type`` values.
        risk_categories: keep only these ``risk_category`` values.
        limit: keep at most this many, in file order, after filtering.
    """
    if dataset not in _DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected {list(DATASET_NAMES)}")
    text = (
        resources.files("cyberseceval_pi_claim.data")
        .joinpath(_DATASETS[dataset])
        .read_text("utf-8")
    )
    raw = json.loads(text)
    if not raw:
        raise ValueError(f"{_DATASETS[dataset]} is empty")
    missing = [f for f in _REQUIRED_FIELDS if f not in raw[0]]
    if missing:
        raise ValueError(f"dataset missing fields {missing}")

    rows = raw
    if injection_types is not None:
        wanted = {t.lower() for t in injection_types}
        rows = [r for r in rows if str(r.get("injection_type", "")).lower() in wanted]
    if risk_categories is not None:
        wanted = {c.lower() for c in risk_categories}
        rows = [r for r in rows if str(r.get("risk_category", "")).lower() in wanted]
    if not rows:
        raise ValueError("no cases matched the given filters")
    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit!r}")
        rows = rows[:limit]

    return [
        InjectionCase(
            prompt_id=int(r.get("prompt_id", i)),
            system_prompt=str(r["test_case_prompt"]),
            user_input=str(r["user_input"]),
            judge_question=str(r["judge_question"]),
            injection_type=str(r.get("injection_type", "")),
            risk_category=str(r.get("risk_category", "")),
            language=str(r.get("speaking_language", "")),
        )
        for i, r in enumerate(rows)
    ]
