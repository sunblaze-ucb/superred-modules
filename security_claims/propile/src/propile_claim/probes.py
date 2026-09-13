"""Prompt construction for the four ProPILE probe types.

Ports the ``__init__`` prompt-building logic of ``garak.probes.propile``'s
``PIILeakTwin`` / ``PIILeakTriplet`` / ``PIILeakQuadruplet`` / ``PIILeakUnstructured``
into pure functions over (records, templates), each yielding ``ProbeItem``s:
the prompt, the target PII (``trigger``), and the ``pii_type``.

Twin uses the name alone to elicit a PII field; triplet/quadruplet add one/two
auxiliary PII fields as context; unstructured elicits relationship/affiliation
info. The bundled synthetic data exercises all four; upstream's own data only
supports twin (see the module README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Upstream field/template wiring (garak probes/propile.py).
_TWIN_TARGETS: Final = (
    ("email", "twin_email"),
    ("phone", "twin_phone"),
    ("address", "twin_address"),
)
_TRIPLET_CONFIGS: Final = (
    ("phone", "email", "triplet_email"),
    ("email", "phone", "triplet_phone"),
    ("email", "address", "triplet_address"),
)
_QUADRUPLET_CONFIGS: Final = (
    (("address", "phone"), "email", "quadruplet_email"),
    (("address", "email"), "phone", "quadruplet_phone"),
    (("email", "phone"), "address", "quadruplet_address"),
)
_RELATIONSHIPS: Final = ("father", "mother", "wife", "husband")
_AFFILIATIONS: Final = ("university", "employer", "organization")


@dataclass(frozen=True)
class ProbeItem:
    """One constructed probe: prompt, target PII, and its type."""

    prompt: str
    trigger: str
    pii_type: str


Records = list[dict[str, str]]
Templates = dict[str, list[str]]


def build_twin_items(records: Records, templates: Templates) -> list[ProbeItem]:
    """Port of ``PIILeakTwin.__init__``: name -> email/phone/address."""
    items: list[ProbeItem] = []
    for record in records:
        name = record.get("name", "")
        if not name:
            continue
        for target_field, category in _TWIN_TARGETS:
            target_value = record.get(target_field, "")
            if not target_value:
                continue
            for template in templates.get(category, []):
                items.append(ProbeItem(template.format(name=name), target_value, target_field))
    return items


def build_triplet_items(records: Records, templates: Templates) -> list[ProbeItem]:
    """Port of ``PIILeakTriplet.__init__``: name + one auxiliary PII -> another."""
    items: list[ProbeItem] = []
    for record in records:
        name = record.get("name", "")
        if not name:
            continue
        for aux_field, target_field, category in _TRIPLET_CONFIGS:
            if not record.get(aux_field, "") or not record.get(target_field, ""):
                continue
            for template in templates.get(category, []):
                prompt = template.format(
                    name=name,
                    email=record.get("email", ""),
                    phone=record.get("phone", ""),
                )
                items.append(ProbeItem(prompt, record[target_field], target_field))
    return items


def build_quadruplet_items(records: Records, templates: Templates) -> list[ProbeItem]:
    """Port of ``PIILeakQuadruplet.__init__``: name + two auxiliary PIIs -> the third."""
    items: list[ProbeItem] = []
    for record in records:
        name = record.get("name", "")
        if not name:
            continue
        for aux_fields, target_field, category in _QUADRUPLET_CONFIGS:
            if not all(record.get(f, "") for f in aux_fields) or not record.get(target_field, ""):
                continue
            for template in templates.get(category, []):
                prompt = template.format(
                    name=name,
                    email=record.get("email", ""),
                    phone=record.get("phone", ""),
                    address=record.get("address", ""),
                )
                items.append(ProbeItem(prompt, record[target_field], target_field))
    return items


def build_unstructured_items(records: Records, templates: Templates) -> list[ProbeItem]:
    """Port of ``PIILeakUnstructured.__init__``: elicit relationship/affiliation info."""
    items: list[ProbeItem] = []
    for record in records:
        name = record.get("name", "")
        if not name:
            continue
        for rel in _RELATIONSHIPS:
            rel_name = record.get(rel, "")
            if rel_name:
                for template in templates.get("unstructured_relation", []):
                    items.append(
                        ProbeItem(
                            template.format(name=name, relationship=rel),
                            rel_name,
                            f"relation_{rel}",
                        )
                    )
        for affiliation in _AFFILIATIONS:
            value = record.get(affiliation, "")
            if value:
                for template in templates.get(f"unstructured_{affiliation}", []):
                    # pii_type is the bare affiliation name (matches garak exactly);
                    # it routes to the generic Jaro-Winkler matcher like relations do.
                    items.append(ProbeItem(template.format(name=name), value, affiliation))
    return items


PROBE_BUILDERS: Final = {
    "twin": build_twin_items,
    "triplet": build_triplet_items,
    "quadruplet": build_quadruplet_items,
    "unstructured": build_unstructured_items,
}


__all__ = [
    "ProbeItem",
    "build_twin_items",
    "build_triplet_items",
    "build_quadruplet_items",
    "build_unstructured_items",
    "PROBE_BUILDERS",
]
