"""Combined chatbot safety SecurityClaim for superred.

Unions the chatbot harmful-instruction benchmarks (HarmBench, SORRY-Bench,
StrongREJECT) into one strong overall claim, de-duplicating functionally
equivalent prompts so no attacker/judge tokens are spent re-testing the same
behaviour.

- :func:`chatbot_suite_claim` -- the headline factory (the three benchmarks).
- :func:`combine_claims` -- the generic, benchmark-agnostic combinator.
"""

from security_claim_chatbot_suite.combine import (
    CombineStats,
    TaskRecord,
    category_of,
    combine_claims,
    normalize_goal,
)
from security_claim_chatbot_suite.suite import (
    DEFAULT_HARMBENCH_JUDGE_MODEL,
    DEFAULT_SORRY_JUDGE_MODEL,
    chatbot_suite_claim,
    manifest_to_dicts,
)

__all__ = [
    "chatbot_suite_claim",
    "combine_claims",
    "normalize_goal",
    "category_of",
    "CombineStats",
    "TaskRecord",
    "manifest_to_dicts",
    "DEFAULT_SORRY_JUDGE_MODEL",
    "DEFAULT_HARMBENCH_JUDGE_MODEL",
]
