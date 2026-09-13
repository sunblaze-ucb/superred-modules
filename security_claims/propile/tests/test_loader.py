"""Loader + vendored-data integrity tests."""

from __future__ import annotations

from propile_claim import actual_sha256, expected_sha256, load_pii_records, load_templates
from propile_claim.loader import _SAMPLE_PII, _TEMPLATES_TSV

_TEMPLATES_SHA = "a57e6dcf7f2facdf54422ee632ef89311963803a6b210b3f0717b3135c847341"


def test_templates_load_all_categories() -> None:
    t = load_templates()
    # twin/triplet/quadruplet + unstructured categories are all present
    assert {"twin_email", "twin_phone", "twin_address"} <= set(t)
    assert {"triplet_email", "triplet_phone", "triplet_address"} <= set(t)
    assert {"quadruplet_email", "quadruplet_phone", "quadruplet_address"} <= set(t)
    assert any(k.startswith("unstructured_") for k in t)
    # each template mentions the {name} placeholder
    for templates in t.values():
        for tmpl in templates:
            assert "{name}" in tmpl


def test_sample_records() -> None:
    records = load_pii_records()
    assert len(records) == 8
    assert all("name" in r for r in records)
    # synthetic: every email is an example.* domain (no real PII)
    for r in records:
        if "email" in r:
            assert r["email"].split("@", 1)[1].startswith("example.")


def test_custom_pii_data_path(tmp_path) -> None:
    p = tmp_path / "mine.jsonl"
    p.write_text('{"name": "Test User", "email": "t@example.com"}\n', encoding="utf-8")
    records = load_pii_records(p)
    assert records == [{"name": "Test User", "email": "t@example.com"}]


def test_vendored_templates_sha_pinned() -> None:
    assert actual_sha256(_TEMPLATES_TSV) == expected_sha256(_TEMPLATES_TSV) == _TEMPLATES_SHA
    # the synthetic sample is also pinned (disk == manifest)
    assert actual_sha256(_SAMPLE_PII) == expected_sha256(_SAMPLE_PII)
