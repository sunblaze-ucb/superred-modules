"""Dataset loading for the AgentHarm claim.

Thin wrapper over upstream ``load_dataset`` / ``filter_dataset`` so the pinned
HuggingFace revision and cache path are guaranteed identical to a real
``inspect eval`` run.  Scope: harmful ``test_public`` (176) is the wired and
verified split; ``benign`` / other splits are reachable via parameters but not
wired as named claims.
"""

from __future__ import annotations

from inspect_ai.dataset import Dataset, Sample
from inspect_evals.agentharm.utils import (
    AGENTHARM_DATASET_REVISION,
    filter_dataset,
    load_dataset,
)

DATASET_REVISION: str = AGENTHARM_DATASET_REVISION
"""Pinned upstream dataset revision (single-sourced from inspect_evals)."""


def load_agentharm_dataset(
    *,
    dataset: str = "harmful",
    split: str = "test_public",
    behavior_ids: list[str] | None = None,
    detailed_behaviors: bool | None = None,
    hint_included: bool | None = None,
    force_redownload: bool = False,
) -> Dataset:
    """Load and filter an AgentHarm dataset split (default: harmful test_public = 176)."""
    ds = load_dataset(dataset, split, force_redownload)  # type: ignore[arg-type]
    return filter_dataset(ds, behavior_ids, detailed_behaviors, hint_included)


def dataset_by_id(ds: Dataset) -> dict[str, Sample]:
    """Index a dataset by sample id."""
    return {str(s.id): s for s in ds}


__all__ = ["DATASET_REVISION", "load_agentharm_dataset", "dataset_by_id"]
