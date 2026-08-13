"""Shared fixtures for the sorry_bench_claim test suite.

No synthetic test fixture is shipped; tests use the real gated
``question.jsonl`` from HuggingFace. The ``question_jsonl_path`` fixture
resolves the dataset via:

1. ``$SORRY_BENCH_QUESTION_JSONL`` env var (an absolute path the user
   has downloaded), then
2. HuggingFace lazy-load via ``hf_hub_download`` (relies on the user
   having accepted the gate and authenticated via ``hf auth login`` or
   ``HF_TOKEN``), then
3. ``pytest.fail`` with an actionable error -- except under ``CI``, where it
   ``pytest.skip``s instead, because the gate cannot be accepted there.

The dataset never gets bundled. Without the env var or ``HF_TOKEN``, the 43
tests taking ``question_jsonl_path`` skip and the module's other 104 still run.
Note ``run_modules.py`` sets ``CI=true`` for every module, so a local run
through that script skips them too; bare ``pytest`` still fails loudly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sorry_bench_claim.factory import (
    DATASET_FILENAME,
    DATASET_REPO_ID,
    DATASET_REVISION,
)


@pytest.fixture(scope="session")
def question_jsonl_path() -> str:
    """Resolve a path to the real SORRY-Bench ``question.jsonl``.

    Fails loudly if neither the env var nor HF auth is available — we
    deliberately do not fall back to a synthetic fixture.
    """
    env_path = os.environ.get("SORRY_BENCH_QUESTION_JSONL")
    if env_path and Path(env_path).is_file():
        return env_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover — huggingface_hub is a hard dep
        pytest.fail(
            "huggingface_hub is required to resolve the SORRY-Bench dataset; "
            "`pip install huggingface_hub>=0.20`."
        )

    try:
        return hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=DATASET_FILENAME,
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    except Exception as exc:
        # Locally we fail loudly so nobody mistakes a skipped run for a real
        # one. CI cannot accept the HuggingFace gate, so there it skips
        # instead of failing on every push. Set SORRY_BENCH_QUESTION_JSONL or
        # HF_TOKEN in CI to run these for real.
        reporter = pytest.skip if os.environ.get("CI") else pytest.fail
        reporter(
            "Cannot resolve the SORRY-Bench question.jsonl. Either:\n"
            "  1. Download question.jsonl manually and "
            "`export SORRY_BENCH_QUESTION_JSONL=/abs/path/question.jsonl`, OR\n"
            f"  2. Accept the gate at https://huggingface.co/datasets/{DATASET_REPO_ID} "
            "and authenticate via `hf auth login` (or set HF_TOKEN).\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record judge back-off delays instead of waiting for them.

    Returns the list the recorder appends to, so a test can assert both
    that a transient failure backed off and that a terminal one did not.
    """
    delays: list[float] = []

    async def _record(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("sorry_bench_claim.judge_failure._default_sleep", _record)
    return delays
