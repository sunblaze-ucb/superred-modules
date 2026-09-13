"""Factory tests: probe-type selection, validation, custom data, sizes."""

from __future__ import annotations

import pytest

from propile_claim import (
    PIILeakTask,
    propile_claim,
    propile_triplet_claim,
    propile_twin_claim,
    propile_unstructured_claim,
)


def test_twin_claim_is_data_backed() -> None:
    tasks = list(propile_twin_claim())
    assert tasks, "twin claim should have tasks from the synthetic sample"
    assert all(isinstance(t, PIILeakTask) for t in tasks)
    assert all(t.dimension.startswith("twin:") for t in tasks if isinstance(t, PIILeakTask))


def test_full_claim_spans_probe_types() -> None:
    tasks = [t for t in propile_claim() if isinstance(t, PIILeakTask)]
    probe_types = {t.dimension.split(":")[0] for t in tasks}
    # the synthetic sample exercises at least twin + unstructured
    assert "twin" in probe_types
    assert "unstructured" in probe_types


def test_probe_type_selection() -> None:
    assert all(
        t.dimension.startswith("triplet:")
        for t in propile_triplet_claim()
        if isinstance(t, PIILeakTask)
    )
    assert all(
        t.dimension.startswith("unstructured:")
        for t in propile_unstructured_claim()
        if isinstance(t, PIILeakTask)
    )


def test_unknown_probe_type_raises() -> None:
    with pytest.raises(ValueError):
        propile_claim(probe_types=["bogus"])


def test_custom_pii_data_path(tmp_path) -> None:
    p = tmp_path / "mine.jsonl"
    p.write_text('{"name": "Test User", "email": "test.user@example.com"}\n', encoding="utf-8")
    tasks = [t for t in propile_twin_claim(pii_data_path=p) if isinstance(t, PIILeakTask)]
    # one record with an email -> one task per twin_email template
    assert tasks and all(t.dimension == "twin:email" for t in tasks)


def test_empty_data_raises_clear_error(tmp_path) -> None:
    # a SecurityClaim cannot be empty; no probe items -> a clear ValueError
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no probe items"):
        propile_twin_claim(pii_data_path=p)
