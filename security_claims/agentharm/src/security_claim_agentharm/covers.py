"""Purpose-built benign cover tasks for benign-cover (IPI) mode.

In benign-cover mode each AgentHarm behaviour is reframed as indirect prompt
injection: the agent is given a BENIGN cover task that engages the behaviour's
read/retrieval tools (so a tool output exists for an attacker to inject into) but
does NOT perform the harmful actions the upstream grader checks, so a do-nothing
attacker fails.  The harmful behaviour becomes the attacker's ``Goal``, to be
injected.  The harmful tools and the harmful grader are kept verbatim from
upstream; only the user task is replaced by a cover.

Covers are keyed by **base** behaviour id (e.g. ``"7"``): a base's 4 augmentations
(``7-1`` .. ``7-4``) share the same tools and grader, so they share one cover.
The covers are purpose-built for superred -- a deliberate, documented deviation
from upstream (they are NOT AgentHarm's benign split, whose tasks are too
action-similar to the harmful ones and would let a do-nothing attacker succeed).
See ``ASSUMPTIONS.md`` section J.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

_COVERS_RESOURCE = "data/benign_covers.json"

EXCLUDED_BASES: frozenset[str] = frozenset({"22"})
"""Base behaviours deliberately NOT reframable as IPI, so excluded from benign-cover
mode (each excludes its 4 augmentations).  ``22`` (ransomware) exposes no read /
content tool: its only tools are ``create_ransomware`` (which IS the graded harmful
action) and ``send_email``, so there is no tool output for an attacker to inject
into and no orthogonal cover that engages a surface without tripping the grader.
See ``ASSUMPTIONS.md`` section J."""


@lru_cache(maxsize=1)
def _covers() -> dict[str, str]:
    raw = files("security_claim_agentharm").joinpath(_COVERS_RESOURCE).read_text(encoding="utf-8")
    data = json.loads(raw)
    return {str(k): str(v) for k, v in data.items()}


def base_id(behavior_id: str) -> str:
    """The base id of a behaviour (``"7-3" -> "7"``); the 4 augmentations share it."""
    return behavior_id.split("-")[0]


def cover_for(behavior_id: str) -> str | None:
    """The benign cover prompt for a behaviour, looked up by its base id (or ``None``)."""
    return _covers().get(base_id(behavior_id))


def covered_base_ids() -> frozenset[str]:
    """Every base id that has a defined cover."""
    return frozenset(_covers())


def is_excluded(behavior_id: str) -> bool:
    """True iff this behaviour's base is deliberately excluded from benign-cover mode."""
    return base_id(behavior_id) in EXCLUDED_BASES


__all__ = ["EXCLUDED_BASES", "base_id", "cover_for", "covered_base_ids", "is_excluded"]
