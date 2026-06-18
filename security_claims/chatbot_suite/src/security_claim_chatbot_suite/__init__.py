"""Combined chatbot safety SecurityClaim for superred.

Unions the chatbot harmful-instruction benchmarks (HarmBench, SORRY-Bench,
StrongREJECT) into one strong overall claim, removing exact AND semantic
duplicate harmful behaviours (the same intent worded differently across
benchmarks). The semantic selection is pre-computed and committed, so the claim
builds deterministically with no embedding at runtime.

- :func:`chatbot_suite_claim` -- the headline factory; call it with a target id
  and judge credentials to get the canonical 79-task claim.
- :func:`combine_claims` -- the generic, benchmark-agnostic combinator.
"""

from security_claim_chatbot_suite._static_dedup import STATIC_SEMANTIC_DUPLICATES
from security_claim_chatbot_suite.combine import (
    CombineStats,
    Embedder,
    LiteLLMEmbedder,
    SemanticDrop,
    TaskRecord,
    category_of,
    combine_claims,
    cosine_similarity,
    normalize_goal,
)
from security_claim_chatbot_suite.suite import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HARMBENCH_JUDGE_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SORRY_JUDGE_MODEL,
    chatbot_suite_claim,
    manifest_to_dicts,
    semantic_drops_to_dicts,
)

__all__ = [
    "chatbot_suite_claim",
    "combine_claims",
    "normalize_goal",
    "category_of",
    "cosine_similarity",
    "Embedder",
    "LiteLLMEmbedder",
    "CombineStats",
    "TaskRecord",
    "SemanticDrop",
    "STATIC_SEMANTIC_DUPLICATES",
    "manifest_to_dicts",
    "semantic_drops_to_dicts",
    "DEFAULT_SORRY_JUDGE_MODEL",
    "DEFAULT_HARMBENCH_JUDGE_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SEMANTIC_THRESHOLD",
]
