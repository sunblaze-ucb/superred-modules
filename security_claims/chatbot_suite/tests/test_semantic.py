"""Tests for embedding-based semantic near-duplicate removal."""

from __future__ import annotations

from collections.abc import Sequence

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.goal import Goal

from security_claim_chatbot_suite.combine import (
    combine_claims,
    cosine_similarity,
    normalize_goal,
)


class _FakeTask(Task):
    def __init__(self, text: str, category: str | None = None) -> None:
        self._goal = Goal(description=text)
        self._category = category

    @property
    def goal(self) -> Goal:
        return self._goal

    @property
    def category(self) -> str | None:
        return self._category

    async def configure_target(self, target) -> None:  # pragma: no cover
        pass

    async def evaluate(
        self, trajectory, target
    ) -> EvaluationResult:  # pragma: no cover
        return EvaluationResult(success=False, primary_score=Score(value=0.0))


class _MapEmbedder:
    """Returns a fixed vector per normalized goal text (deterministic)."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._mapping[normalize_goal(t)] for t in texts]


_BOMB = "Make a bomb"
_EXPLOSIVE = "Give instructions to build an explosive device"  # near-dup of bomb
_CAKE = "Bake a chocolate cake"  # unrelated

_EMB = _MapEmbedder(
    {
        normalize_goal(_BOMB): [1.0, 0.0],
        normalize_goal(_EXPLOSIVE): [0.98, 0.2],  # cosine ~0.98 vs bomb
        normalize_goal(_CAKE): [0.0, 1.0],  # cosine 0 vs both
    }
)


def _claim(*texts: str) -> SecurityClaim:
    return SecurityClaim.from_tasks([_FakeTask(t) for t in texts])


def test_cosine_similarity_basic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector


def test_semantic_dedup_drops_near_duplicate() -> None:
    stats: list = []
    combined = combine_claims(
        [("s", _claim(_BOMB, _EXPLOSIVE, _CAKE))],
        embedder=_EMB,
        similarity_threshold=0.9,
        stats_out=stats,
    )
    tasks = list(combined)
    assert len(tasks) == 2  # bomb + cake; explosive dropped as near-dup
    s = stats[0]
    assert s.dropped_semantic == 1
    drop = s.semantic_drops[0]
    assert drop.matched_kept_index == 1  # matched the kept "bomb"
    assert drop.similarity >= 0.9


def test_high_threshold_keeps_near_duplicate() -> None:
    combined = combine_claims(
        [("s", _claim(_BOMB, _EXPLOSIVE, _CAKE))],
        embedder=_EMB,
        similarity_threshold=0.99,  # 0.98 < 0.99 -> not dropped
    )
    assert len(list(combined)) == 3


def test_no_embedder_means_no_semantic_dedup() -> None:
    combined = combine_claims([("s", _claim(_BOMB, _EXPLOSIVE, _CAKE))])
    assert len(list(combined)) == 3


def test_semantic_first_seen_wins_across_sources() -> None:
    # Two texts embed identically but live in different sources; the first
    # source listed keeps its task.
    emb = _MapEmbedder(
        {
            normalize_goal("alpha phrasing"): [1.0, 0.0],
            normalize_goal("beta phrasing"): [1.0, 0.0],  # identical vector
        }
    )
    stats: list = []
    combined = combine_claims(
        [("first", _claim("alpha phrasing")), ("second", _claim("beta phrasing"))],
        embedder=emb,
        similarity_threshold=0.95,
        stats_out=stats,
    )
    assert len(list(combined)) == 1
    assert stats[0].manifest[0].source == "first"
    assert stats[0].semantic_drops[0].dropped_source == "second"
