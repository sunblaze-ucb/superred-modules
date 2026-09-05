"""One Snowball question as a superred Task, bound to ChatbotTarget.

Single-turn: the question is the goal, and the deterministic detector decides
whether the model gave the hallucinated (confident-wrong) answer. No judge
model, no network.
"""

from __future__ import annotations

from chatbot_target import ChatbotTarget  # type: ignore[import-untyped]
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal
from superred.core.types.trajectory import Trajectory

from snowball_claim.judge import hallucinated
from snowball_claim.loader import SnowballItem

_SYSTEM_PROMPT_CONTROLLABLE = "system_prompt"


class SnowballTask(Task[ChatbotTarget]):
    """One snowball question, judged by its fixed detector polarity."""

    def __init__(self, *, item: SnowballItem) -> None:
        self._item = item
        self._goal = Goal(description=item.question)

    @property
    def subset(self) -> str:
        return self._item.subset

    @property
    def polarity(self) -> str:
        return self._item.polarity

    @property
    def goal(self) -> Goal:
        return self._goal

    async def configure_target(self, target: ChatbotTarget) -> None:
        # Snowball adds no system prompt; ChatbotTarget suppresses an empty one.
        target.set_config(_SYSTEM_PROMPT_CONTROLLABLE, "")

    async def evaluate(
        self, trajectory: Trajectory, target: ChatbotTarget
    ) -> EvaluationResult:
        response = target.query("last_response")
        is_hallucinated = hallucinated(response, self._item.polarity)
        primary = Score(
            value=1.0 if is_hallucinated else 0.0,
            security_domain=None,
            name="snowball_asr",
        )
        return EvaluationResult(
            success=is_hallucinated,
            primary_score=primary,
            sub_scores={
                f"subset_{self.subset}": Score(
                    value=primary.value,
                    security_domain=None,
                    name=f"subset_{self.subset}",
                ),
            },
            rationale=(
                f"subset={self.subset} polarity={self.polarity} "
                f"response_head={response.strip()[:80]!r}"
            ),
        )
