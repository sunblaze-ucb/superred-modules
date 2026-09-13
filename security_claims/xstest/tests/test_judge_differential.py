"""Golden differential: the ported ``strmatch_label`` must reproduce upstream's
own published labels **exactly**.

Upstream ships model completions (``model_completions/*.csv``) and the labels its
own classifier assigned them (``evaluation/automated_evaluation_labels/*_streval.csv``).
Running our port over the completions must reproduce those labels byte-for-byte.

The completion CSVs carry Meta/Mistral/OpenAI model-output licenses, so they are
**not vendored**. This test is therefore opt-in: point ``XSTEST_UPSTREAM_DIR`` at
a clone of ``github.com/paul-rottger/xstest`` to run it (it verified 2250/2250 at
port time across all five model sets). Without the env var it skips.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from xstest_claim import strmatch_label

_UPSTREAM = os.getenv("XSTEST_UPSTREAM_DIR")

pytestmark = pytest.mark.skipif(
    not _UPSTREAM,
    reason="XSTEST_UPSTREAM_DIR not set (points at a paul-rottger/xstest clone)",
)

_MODELS = ["gpt4", "llama2new", "llama2orig", "mistralinstruct", "mistralguard"]


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("model", _MODELS)
def test_strmatch_reproduces_upstream_labels(model: str) -> None:
    root = Path(_UPSTREAM)  # type: ignore[arg-type]
    completions = {
        r["id"]: r["completion"]
        for r in _read(root / "model_completions" / f"xstest_v2_completions_{model}.csv")
    }
    labels = {
        r["id"]: r["strmatch_label"]
        for r in _read(
            root
            / "evaluation"
            / "automated_evaluation_labels"
            / f"xstest_v2_completions_{model}_streval.csv"
        )
    }
    assert labels, f"no upstream labels found for {model}"
    mismatches = [_id for _id, lab in labels.items() if strmatch_label(completions[_id]) != lab]
    assert not mismatches, f"{model}: {len(mismatches)} label mismatch(es): {mismatches[:5]}"
