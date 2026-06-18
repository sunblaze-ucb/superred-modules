"""Recompute the committed semantic-duplicate selection in ``_static_dedup.py``.

Run when the benchmarks, the canonical sampling, the embedding model, or the
threshold change. It embeds every prompt in the canonical claim and prints a
ready-to-paste ``SEMANTIC_DUPLICATE_PROVENANCE`` block; copy it into
``_static_dedup.py`` (the static set is derived from it).

Usage:
    LITELLM_API_BASE=... LITELLM_API_KEY=... \\
        python -m security_claim_chatbot_suite.regenerate_static_dedup

Requires the proxy (an embedding model + the judge creds for SORRY-Bench's
config) and the SORRY-Bench dataset (HF cache or SORRY_BENCH_DATASET_PATH).
Embedding is out of band and cheap (~80 short prompts, once).
"""

from __future__ import annotations

import os
import sys

from security_claim_chatbot_suite.combine import (
    CombineStats,
    LiteLLMEmbedder,
    TaskRecord,
)
from security_claim_chatbot_suite.suite import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    chatbot_suite_claim,
)


def main() -> int:
    base = os.environ.get("LITELLM_API_BASE")
    key = os.environ.get("LITELLM_API_KEY")
    if not base or not key:
        print("set LITELLM_API_BASE and LITELLM_API_KEY", file=sys.stderr)
        return 2

    manifest: list[TaskRecord] = []
    stats: list[CombineStats] = []
    chatbot_suite_claim(
        # The victim only selects SORRY-Bench's system prompt; the prompt SET
        # (hence the dedup) is target-independent.
        target_model_id="openai/gpt-4o-mini",
        judge_api_base=base,
        judge_api_key=key,
        sorry_dataset_path=os.environ.get("SORRY_BENCH_DATASET_PATH"),
        apply_static_dedup=False,  # recompute, do not apply the committed set
        embedder=LiteLLMEmbedder(
            model=DEFAULT_EMBEDDING_MODEL, api_base=base, api_key=key
        ),
        similarity_threshold=DEFAULT_SEMANTIC_THRESHOLD,
        manifest_out=manifest,
        stats_out=stats,
    )
    s = stats[0]

    print(
        f"# threshold {DEFAULT_SEMANTIC_THRESHOLD}, embedding {DEFAULT_EMBEDDING_MODEL}"
    )
    print(f"# kept {s.kept}, dropped_semantic {s.dropped_semantic}")
    print("SEMANTIC_DUPLICATE_PROVENANCE = (")
    for d in s.semantic_drops:
        kept = manifest[d.matched_kept_index - 1]
        print("    (")
        print(f"        {d.dropped_goal!r},")
        print(f"        {d.dropped_source!r},")
        print(f"        {kept.goal_preview!r},")
        print(f"        {kept.source!r},")
        print(f"        {d.similarity},")
        print("    ),")
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
