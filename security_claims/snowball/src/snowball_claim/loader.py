"""Load the vendored Snowball hallucination question sets.

Upstream data: garak ``garak/data/{graph_connectivity,primality_testing,
senator_search}.json`` (Apache-2.0), derived from Zhang et al., "How Language
Model Hallucinations Can Snowball" (2023).

Each subset pairs its questions with a fixed detector polarity:
- graph_connectivity / senator_search: a hallucination *affirms* an
  impossibility -> polarity ``"yes"``.
- primality_testing: a hallucination *denies* a true prime -> polarity
  ``"no"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

# subset -> (data file, json shape, detector polarity)
_SUBSETS: dict[str, tuple[str, str, str]] = {
    "graph_connectivity": ("graph_connectivity.json", "list", "yes"),
    "primes": ("primality_testing.json", "question_dicts", "no"),
    "senators": ("senator_search.json", "list", "yes"),
}
SUBSET_NAMES: tuple[str, ...] = tuple(_SUBSETS)


@dataclass(frozen=True)
class SnowballItem:
    """One snowball question and how to judge its answer.

    Attributes:
        subset: One of :data:`SUBSET_NAMES`.
        question: The prompt to send.
        polarity: ``"yes"`` or ``"no"`` — which confident answer is the
            hallucination for this question.
    """

    subset: str
    question: str
    polarity: str


def _read(file_name: str) -> object:
    text = (
        resources.files("snowball_claim.data").joinpath(file_name).read_text("utf-8")
    )
    return json.loads(text)


def load_items(
    *,
    subsets: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> list[SnowballItem]:
    """Return snowball questions across the requested subsets.

    Args:
        subsets: Which subsets to include, in order. Defaults to all three.
        limit: Keep at most this many questions *per subset*, taken from the
            end of the file — matching garak's ``self.prompts[-limit:]`` in
            its capped (default-active) probe variants.

    Raises:
        ValueError: on an unknown subset or a non-positive limit.
    """
    chosen = tuple(subsets) if subsets is not None else SUBSET_NAMES
    unknown = [s for s in chosen if s not in _SUBSETS]
    if unknown:
        raise ValueError(f"unknown subset(s): {unknown}; expected {list(SUBSET_NAMES)}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit!r}")

    items: list[SnowballItem] = []
    for subset in chosen:
        file_name, shape, polarity = _SUBSETS[subset]
        data = _read(file_name)
        if shape == "question_dicts":
            questions = [str(entry["question"]) for entry in data]
        else:
            questions = [str(entry) for entry in data]
        if limit is not None:
            questions = questions[-limit:]
        items.extend(
            SnowballItem(subset=subset, question=q, polarity=polarity)
            for q in questions
        )
    return items
