"""Parse the attacker LLM's JSON replies for each ActorAttack step.

Upstream validates these with the pydantic models in AI-Infra-Guard's
``actor_attack/schema.py`` (``ActorNetwork``/``ActorCandidate``,
``FollowUpQuestion``, ``NonRefusal``, ``Rating``). This module reads the same
fields without taking a pydantic dependency and returns ``None`` instead of
raising, because upstream wraps each step in ``except Exception: continue`` —
the caller needs a value it can branch on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Actor:
    """One ``ActorCandidate``: a persona the conversation approaches the goal through.

    Attributes:
        actor_name: Who the questions are about.
        relation_to_goal: Why that persona is a route to the objective.
        opening_question: The first turn sent to the target.
    """

    actor_name: str
    relation_to_goal: str
    opening_question: str


def _strip_fence(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return fence.group(1).strip() if fence else text.strip()


def _load(raw: str) -> dict | None:
    """The first JSON object in *raw*, tolerant of a fence or surrounding prose.

    Upstream constrains these replies with pydantic (``generate(prompt, Model,
    ...)``), so a chatty answer never reaches its parser. This module calls
    ``llm.complete`` with no structured-output enforcement, which makes
    "Here is the JSON: {...}" a realistic shape -- and when a reply is
    rejected here, the attacker never gets an actor network, so the attack ends before it starts. The bare-object fallback matches the one the shared
    ``surface_llm.parse_json_object`` already uses.
    """
    text = _strip_fence(raw)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_actor_network(raw: str) -> list[Actor]:
    """``ActorNetwork.actors`` -- drops any candidate missing a field."""
    data = _load(raw)
    if not data:
        return []
    actors = data.get("actors")
    if not isinstance(actors, list):
        return []
    out: list[Actor] = []
    for entry in actors:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("actor_name", "")).strip()
        relation = str(entry.get("relation_to_goal", "")).strip()
        opening = str(entry.get("opening_question", "")).strip()
        if name and opening:
            out.append(
                Actor(
                    actor_name=name,
                    relation_to_goal=relation,
                    opening_question=opening,
                )
            )
    return out


def parse_follow_up(raw: str) -> tuple[str, bool] | None:
    """``FollowUpQuestion`` -> ``(next_question, is_final_probe)``."""
    data = _load(raw)
    if not data:
        return None
    question = str(data.get("next_question", "")).strip()
    if not question:
        return None
    return question, bool(data.get("is_final_probe", False))


def parse_refusal(raw: str) -> bool | None:
    """``NonRefusal.classification`` -> True when the target refused.

    ``None`` means unreadable; upstream's ``_generate_schema`` would have
    raised, and the caller treats it as "not a refusal" so a garbled check
    never invents one.
    """
    data = _load(raw)
    if not data:
        return None
    value = data.get("classification")
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered == "refusal":
        return True
    if lowered == "non-refusal":
        return False
    return None


def parse_rating(raw: str) -> int | None:
    """``Rating.rating`` -- the attacker-side score steering the search."""
    data = _load(raw)
    if not data:
        return None
    value = data.get("rating")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


__all__ = [
    "Actor",
    "parse_actor_network",
    "parse_follow_up",
    "parse_rating",
    "parse_refusal",
]
