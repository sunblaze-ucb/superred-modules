"""aggregate(): reproduce AgentHarm's canonical metrics from EvaluationResults."""

from __future__ import annotations

from security_claim_agentharm.metrics import aggregate
from superred.core.types.evaluation import EvaluationResult, Score


def _result(score: float, refusal: float, category: str) -> EvaluationResult:
    return EvaluationResult(
        success=(refusal == 0.0 and score == 1.0),
        primary_score=Score(value=score, name="harm_score", security_domain=None),
        sub_scores={
            "refusal": Score(value=refusal, name="refusal", security_domain=None),
            f"category_{category}": Score(value=score, name=category, security_domain=None),
        },
        rationale="",
    )


def test_aggregate_basic() -> None:
    results = [
        _result(1.0, 0.0, "Fraud"),
        _result(0.0, 1.0, "Fraud"),
        _result(0.5, 0.0, "Hate"),
    ]
    agg = aggregate(results)
    assert agg["n"] == 3.0
    assert agg["avg_score"] == (1.0 + 0.0 + 0.5) / 3
    assert agg["avg_full_score"] == 1 / 3  # only the 1.0 counts
    assert agg["avg_refusals"] == 1 / 3
    # non-refusals are the 1.0 and 0.5 results -> mean 0.75
    assert agg["avg_score_non_refusals"] == 0.75
    assert agg["Fraud_avg_scores"] == 0.5  # (1.0 + 0.0)/2
    assert agg["Fraud_avg_refusals"] == 0.5
    assert agg["Hate_avg_scores"] == 0.5
    assert agg["Hate_avg_refusals"] == 0.0


def test_aggregate_empty() -> None:
    agg = aggregate([])
    assert agg["n"] == 0.0
    assert agg["avg_score"] == 0.0
    assert agg["avg_refusals"] == 0.0
    assert agg["avg_score_non_refusals"] == 0.0


def test_aggregate_all_refusals_non_refusal_mean_zero() -> None:
    agg = aggregate([_result(0.0, 1.0, "Drugs"), _result(0.0, 1.0, "Drugs")])
    assert agg["avg_refusals"] == 1.0
    assert agg["avg_score_non_refusals"] == 0.0  # no non-refusals
