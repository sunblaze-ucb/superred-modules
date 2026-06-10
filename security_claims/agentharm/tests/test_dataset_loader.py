"""Dataset loader: 176 harmful test_public, by-id index, pinned revision."""

from __future__ import annotations

from security_claim_agentharm.dataset_loader import (
    DATASET_REVISION,
    dataset_by_id,
    load_agentharm_dataset,
)


def test_loads_176_harmful_test_public() -> None:
    ds = load_agentharm_dataset()
    assert len(ds) == 176


def test_revision_pinned() -> None:
    assert DATASET_REVISION == "e23b3fe60a0da9037314b88e5ee3a0c054970dad"


def test_dataset_by_id() -> None:
    ds = load_agentharm_dataset()
    index = dataset_by_id(ds)
    assert len(index) == 176
    assert "1-1" in index


def test_behavior_id_filter() -> None:
    ds = load_agentharm_dataset(behavior_ids=["1-1"])
    assert len(ds) == 1
    assert str(ds[0].id) == "1-1"
