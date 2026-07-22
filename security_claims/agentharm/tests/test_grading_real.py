"""Run the REAL upstream grading functions through our bridge, offline.

The other grading tests mock the grader; this one invokes the actual
``inspect_evals.agentharm`` grading function for a sample of behaviors (one per
category) via ``grading_bridge.grade``, on an empty message trace, with a stub
semantic judge. It verifies the wiring (metadata injection, getattr dispatch,
mean over the score_dict, error handling) against real graders without LLM cost
or hardcoding any gated expected values. An empty trace exercises no harmful
action, so the score is expected to be low (typically 0).
"""

from __future__ import annotations

from typing import Any

import pytest

from agentharm_claim import grading_bridge as gb
from agentharm_claim.dataset_loader import load_agentharm_dataset


class _StubJudge:
    """Stand-in for an inspect Model: upstream semantic_judge reads .completion."""

    async def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        class _Out:
            completion = "fail"

        return _Out()


def _one_sample_per_category() -> list:
    seen: dict[str, Any] = {}
    for sample in load_agentharm_dataset():
        cat = str(sample.metadata["category"])
        seen.setdefault(cat, sample)
    return [seen[c] for c in sorted(seen)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample", _one_sample_per_category(), ids=lambda s: str(s.metadata["category"])
)
async def test_real_grader_runs_through_bridge(sample) -> None:  # noqa: ANN001
    md = dict(sample.metadata)
    score, score_dict, err = await gb.grade(md, [], semantic_judge=_StubJudge())
    assert err is None, f"{md['grading_function']} raised: {err}"
    assert isinstance(score_dict, dict) and score_dict, "grader returned empty score_dict"
    assert all(isinstance(v, bool) for v in score_dict.values())
    assert 0.0 <= score <= 1.0
