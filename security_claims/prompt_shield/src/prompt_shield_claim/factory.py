"""Factories for the Prompt Shields evasion / over-block claim.

``prompt_shield_claim(...)`` builds a :class:`SecurityClaim` of one
:class:`~prompt_shield_claim.task.PromptShieldTask` per seed. Two named wrappers
cut it by axis:

* :func:`prompt_shield_evasion_claim` — attack seeds only (the detector should
  flag these; an evasion is a miss);
* :func:`prompt_shield_overblock_claim` — benign seeds only (the detector should
  not flag these; a flag is over-blocking).

All default to the bundled illustrative seed set; pass ``seeds=`` (a sequence of
:class:`~prompt_shield_claim.seeds.Seed`) for your own corpus.
"""

from __future__ import annotations

from collections.abc import Iterable

from prompt_shield_target import PromptShieldTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from prompt_shield_claim.seeds import DEFAULT_SEEDS, Seed
from prompt_shield_claim.task import PromptShieldTask


def _tasks(seeds: Iterable[Seed], success_threshold: float) -> list[Task[PromptShieldTarget]]:
    return [
        PromptShieldTask(
            seed_id=s.seed_id,
            text=s.text,
            kind=s.kind,
            channel=s.channel,
            success_threshold=success_threshold,
        )
        for s in seeds
    ]


def prompt_shield_claim(
    *,
    seeds: Iterable[Seed] | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[PromptShieldTarget]:
    """Build a Prompt Shields claim over all seeds (both axes)."""
    chosen = tuple(seeds) if seeds is not None else DEFAULT_SEEDS
    return SecurityClaim.from_tasks(_tasks(chosen, success_threshold))


def prompt_shield_evasion_claim(
    *,
    seeds: Iterable[Seed] | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[PromptShieldTarget]:
    """Attack seeds only — evasion (attackDetected == false on a real attack)."""
    chosen = tuple(seeds) if seeds is not None else DEFAULT_SEEDS
    attacks = [s for s in chosen if s.kind == "attack"]
    return SecurityClaim.from_tasks(_tasks(attacks, success_threshold))


def prompt_shield_overblock_claim(
    *,
    seeds: Iterable[Seed] | None = None,
    success_threshold: float = 1.0,
) -> SecurityClaim[PromptShieldTarget]:
    """Benign seeds only — over-blocking (attackDetected == true on a benign input)."""
    chosen = tuple(seeds) if seeds is not None else DEFAULT_SEEDS
    benign = [s for s in chosen if s.kind == "benign"]
    return SecurityClaim.from_tasks(_tasks(benign, success_threshold))


__all__ = [
    "prompt_shield_claim",
    "prompt_shield_evasion_claim",
    "prompt_shield_overblock_claim",
]
