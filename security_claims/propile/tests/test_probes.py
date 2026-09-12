"""Probe prompt-construction tests over the synthetic sample."""

from __future__ import annotations

from propile_claim import (
    build_quadruplet_items,
    build_triplet_items,
    build_twin_items,
    build_unstructured_items,
    load_pii_records,
    load_templates,
)

_RECORDS = load_pii_records()
_TEMPLATES = load_templates()


def test_twin_items_cover_present_fields() -> None:
    items = build_twin_items(_RECORDS, _TEMPLATES)
    assert items, "twin should produce items from the sample"
    # each item's prompt embeds the name and the trigger is the target PII
    for it in items:
        assert it.pii_type in ("email", "phone", "address")
        assert "{name}" not in it.prompt  # formatted
    # Ada has email+phone+address -> her name appears with all three types
    ada = [it for it in items if "Ada Lovelace" in it.prompt]
    assert {it.pii_type for it in ada} == {"email", "phone", "address"}
    assert any(it.trigger == "ada.lovelace@example.com" for it in ada)


def test_triplet_items_need_two_fields() -> None:
    items = build_triplet_items(_RECORDS, _TEMPLATES)
    # records with both email+phone (e.g. Ada) yield triplet items; trigger != aux
    assert any(it.pii_type in ("email", "phone", "address") for it in items)
    for it in items:
        assert "{name}" not in it.prompt


def test_quadruplet_items_need_three_fields() -> None:
    items = build_quadruplet_items(_RECORDS, _TEMPLATES)
    # Ada/Alan have email+phone+address -> quadruplet_email/phone/address fire
    kinds = {it.pii_type for it in items}
    assert kinds  # non-empty for the sample
    for it in items:
        assert "{name}" not in it.prompt


def test_unstructured_items_use_affiliations() -> None:
    items = build_unstructured_items(_RECORDS, _TEMPLATES)
    # Grace/Katherine/Radia have employer; Edsger university; Barbara organization
    ptypes = {it.pii_type for it in items}
    assert any(p.startswith("affiliation_") for p in ptypes)
    for it in items:
        assert "{name}" not in it.prompt


def test_empty_data_yields_no_items() -> None:
    # data-driven: no records -> no prompts (garak's own behaviour)
    assert build_twin_items([], _TEMPLATES) == []
    assert build_unstructured_items([{"name": "No Fields"}], _TEMPLATES) == []
