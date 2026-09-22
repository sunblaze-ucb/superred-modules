"""NVIDIA Aegis 1.0 content-safety taxonomy.

The Aegis / *Nemotron Content Safety Dataset V1* taxonomy has **13 harm
categories** plus the gray-zone disposition ``"Needs Caution"`` and the
negative label ``"Safe"`` (dataset card,
https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0).

The names here are the dataset's own ``labels_0`` spellings, which differ in
punctuation and wording from the taxonomy table printed on the dataset card
(e.g. card ``"Guns /Illegal Weapons"`` vs. data ``"Guns and Illegal Weapons"``;
card ``"PII"`` vs. data ``"PII/Privacy"``; card ``"Sexual Minor"`` vs. data
``"Sexual (minor)"``). We follow the **data**, not the card, because the loader
matches against the labels that actually appear in the vendored file.
``test_categories.py`` pins this list against the vendored test split so the two
can never drift apart silently.

A single row's ``labels_0`` may be a **comma-joined multi-label** string (e.g.
``"Criminal Planning/Confessions, Violence"``) — the aggregation of several
annotators' votes. Callers that need the individual harm categories can split on
", " (see :func:`atomic_categories`).
"""

from __future__ import annotations

import re

#: The 13 Aegis harm categories, in the dataset's own ``labels_0`` spelling.
HARM_CATEGORIES: list[str] = [
    "Controlled/Regulated Substances",
    "Criminal Planning/Confessions",
    "Guns and Illegal Weapons",
    "Harassment",
    "Hate/Identity Hate",
    "Other",
    "PII/Privacy",
    "Profanity",
    "Sexual",
    "Sexual (minor)",
    "Suicide and Self Harm",
    "Threat",
    "Violence",
]

#: Gray-zone disposition: the annotators flagged the content as needing caution
#: without assigning a specific harm category. Not one of the 13 harm
#: categories; the dataset card lists it separately.
NEEDS_CAUTION: str = "Needs Caution"

#: The negative (safe) label.
SAFE: str = "Safe"

#: Every atomic label that can appear as a comma-separated component of
#: ``labels_0``: the 13 harm categories, plus "Needs Caution" and "Safe".
ALL_LABELS: list[str] = [*HARM_CATEGORIES, NEEDS_CAUTION, SAFE]

_HARM_SET = frozenset(HARM_CATEGORIES)
_ALL_SET = frozenset(ALL_LABELS)


def atomic_categories(label: str) -> list[str]:
    """Split a (possibly multi-label) ``labels_0`` value into atomic categories.

    ``"Criminal Planning/Confessions, Violence"`` ->
    ``["Criminal Planning/Confessions", "Violence"]``. Whitespace around each
    component is stripped; empty components are dropped.
    """
    return [part.strip() for part in label.split(",") if part.strip()]


def is_unsafe(label: str, *, include_needs_caution: bool = True) -> bool:
    """Whether a ``labels_0`` value denotes an unsafe prompt.

    A label is unsafe when it names at least one harm category. When
    ``include_needs_caution`` is True (the default, matching deepteam's
    ``labels_0 != "Safe"`` filter), a bare ``"Needs Caution"`` also counts as
    unsafe. With ``include_needs_caution=False`` only the 13 harm categories
    count, so a prompt labelled solely ``"Needs Caution"`` is excluded.
    """
    atoms = atomic_categories(label)
    if any(a in _HARM_SET for a in atoms):
        return True
    if include_needs_caution and NEEDS_CAUTION in atoms:
        return True
    return False


def slugify(label: str) -> str:
    """A stable, key-safe slug for a category label.

    ``"Hate/Identity Hate"`` -> ``"hate_identity_hate"``. Used to build
    ``sub_scores`` keys that are safe to sort and index on.
    """
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


assert len(HARM_CATEGORIES) == 13
assert len(set(HARM_CATEGORIES)) == 13


__all__ = [
    "HARM_CATEGORIES",
    "NEEDS_CAUTION",
    "SAFE",
    "ALL_LABELS",
    "atomic_categories",
    "is_unsafe",
    "slugify",
]
