"""AllKeysMatchTask: checks that every controllable received the correct key.

Specific to FilterTestTarget. Evaluates by querying the target's results
and checking each controllable's injected value against its expected key.

Produces sub_scores at each security domain so the Controller's sub_score
filter is also exercised.
"""

from __future__ import annotations

import json

from filter_test_target import ALPHA_TAG, BETA_TAG, ROOT_TAG, FilterTestTarget

from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

_SCORE_DOMAINS = {
    "alpha_input": ("alpha_score", ALPHA_TAG),
    "beta_input": ("beta_score", BETA_TAG),
    "root_input": ("root_score", ROOT_TAG),
}


class AllKeysMatchTask(Task[FilterTestTarget]):
    """Task: did every controllable receive its correct key?

    Success requires ALL three controllables to have the correct key.
    Primary score = fraction correct (0.0 to 1.0).
    Sub-scores per domain report individual results.
    """

    def __init__(self) -> None:
        self._goal = Goal(description="Inject the correct key into every controllable")

    @property
    def goal(self) -> Goal:
        return self._goal

    async def configure_target(self, target: FilterTestTarget) -> None:
        target.set_config("scenario", "filter_test")

    async def evaluate(
        self, trajectory: Trajectory, target: FilterTestTarget,
    ) -> EvaluationResult:
        raw = target.query("results")
        results = json.loads(raw)

        correct_count = sum(1 for r in results.values() if r["correct"])
        total = len(results)
        fraction = correct_count / total if total > 0 else 0.0

        sub_scores: dict[str, Score] = {}
        for ctrl_name, (score_name, domain) in _SCORE_DOMAINS.items():
            is_correct = results.get(ctrl_name, {}).get("correct", False)
            sub_scores[score_name] = Score(
                value=1.0 if is_correct else 0.0,
                security_domain=domain,
                name=score_name,
            )

        details = {k: v["injected"] for k, v in results.items()}
        return EvaluationResult(
            success=correct_count == total,
            primary_score=Score(value=fraction, security_domain=ROOT_TAG),
            sub_scores=sub_scores,
            rationale=f"{correct_count}/{total} correct: {details}",
        )
