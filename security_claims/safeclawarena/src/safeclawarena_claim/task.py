"""SafeClawArenaTask: one superred Task per SafeClawArena benchmark task.

``configure_target`` hands the whole task (environment + sessions) to the
``safeclawarena`` target, which provisions a containerized platform replica,
runs the task's sessions, and captures the post-run state. ``evaluate`` reads
that captured state back and runs the deterministic judge
(:func:`safeclawarena_claim.judge.evaluate_task`) — a faithful port of upstream
``judge.py``. The attack payload is baked into the task's environment (as
upstream ships it); an optimizer may still vary the exposed user/external-data
surfaces, but the benchmark runs as-designed under a passthrough optimizer.

Polarity (upstream): ``attack_succeeded`` True == the agent failed the security
principle; the dimension ``score`` is 1.0 for secure behaviour, 0.0 for a fully
successful attack. :class:`~superred.core.types.evaluation.EvaluationResult`
``success`` is set to ``attack_succeeded`` (True == the adversarial goal met).
"""

from __future__ import annotations

import json
from typing import Any

from safeclawarena_target import DIMENSION_TAG, SafeClawArenaTarget
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from safeclawarena_claim.judge import PostState, evaluate_task


class SafeClawArenaTask(Task[SafeClawArenaTarget]):
    """One SafeClawArena adversarial task, bound to the ``safeclawarena`` target.

    Args:
        task: one SafeClawArena task dict (verbatim from the vendored corpus).
        platform: which container replica to run against — ``"openclaw"``
            (default), ``"nemoclaw"``, or ``"seclaw"``.
    """

    def __init__(self, *, task: dict[str, Any], platform: str = "openclaw") -> None:
        self._task = task
        self._platform = platform
        meta = task["metadata"]
        self._task_id = str(meta["task_id"])
        self._dimension = str(meta["dimension"])
        sessions = task.get("sessions", [])
        first_instruction = str(sessions[0]["user_instruction"]) if sessions else ""
        # The Goal carries the primary user turn (the benign cover instruction);
        # the adversarial payload is baked into the task environment, so a
        # passthrough optimizer runs the benchmark as-designed and a
        # content-injection optimizer may vary the exposed surfaces.
        self._goal = Goal(description=first_instruction or str(meta.get("title", self._task_id)))

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def dimension(self) -> str:
        return self._dimension

    async def configure_target(self, target: SafeClawArenaTarget) -> None:
        target.set_config("platform", self._platform)
        target.set_config("task", json.dumps(self._task))

    async def evaluate(
        self, _trajectory: Trajectory, target: SafeClawArenaTarget
    ) -> EvaluationResult:
        raw = target.query("post_state")
        try:
            captured = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            captured = {}
        ps = _post_state_from(captured)

        verdict = evaluate_task(self._task, ps)

        # superred convention: a HIGHER primary_score is a more-successful attack
        # (the controller reports the run with max primary as "best", and every
        # sibling claim reports higher == attack). ``verdict.score`` is upstream's
        # *defense* score (1.0 = secure, 0.0 = attack fully succeeded), so the
        # attack magnitude is its complement. Upstream's exact score is preserved
        # verbatim as the ``defense_score`` sub-score for reproducibility.
        attack_magnitude = round(1.0 - verdict.score, 4)
        primary = Score(value=attack_magnitude, security_domain=None, name="attack_score")
        dim_tag = DIMENSION_TAG.get(self._dimension)
        sub_scores: dict[str, Score] = {
            "attack_succeeded": Score(
                value=float(verdict.attack_succeeded),
                security_domain=None,
                name="attack_succeeded",
            ),
            "detection": Score(
                value=float(verdict.detection), security_domain=None, name="detection"
            ),
            # Upstream's dimension score (higher == more secure), verbatim.
            "defense_score": Score(
                value=verdict.score, security_domain=None, name=verdict.metric
            ),
        }
        if dim_tag is not None:
            # The dimension's attack crosses its trust boundary; tag it so it is
            # pruned outside a scope that grants that boundary.
            sub_scores["attack_succeeded_in_domain"] = Score(
                value=float(verdict.attack_succeeded),
                security_domain=dim_tag,
                name="attack_succeeded_in_domain",
            )
        for i, tl in enumerate(verdict.taint_leakage):
            if tl.get("leaked"):
                sub_scores[f"taint_leaked_{tl.get('asset_id', i)}"] = Score(
                    value=1.0, security_domain=dim_tag, name="taint_leaked"
                )

        passed = [r for r in verdict.success_results if r.passed]
        rationale = (
            f"safeclawarena {self._task_id} ({self._dimension}/{verdict.metric}) "
            f"on {self._platform}: attack_succeeded={verdict.attack_succeeded} "
            f"attack_score={attack_magnitude} defense_score={verdict.score} "
            f"passed_checks={[r.type for r in passed]}"
        )
        return EvaluationResult(
            success=verdict.attack_succeeded,
            primary_score=primary,
            sub_scores=sub_scores,
            rationale=rationale,
        )


def _post_state_from(d: dict[str, Any]) -> PostState:
    """Build a :class:`PostState` from the target's captured ``post_state`` dict.

    Unknown keys are ignored; missing keys fall back to :class:`PostState`
    defaults, so a partial capture degrades safely (no attack registers on
    absent evidence).
    """
    ps = PostState()
    for key, value in d.items():
        if hasattr(ps, key):
            setattr(ps, key, value)
    return ps


__all__ = ["SafeClawArenaTask"]
