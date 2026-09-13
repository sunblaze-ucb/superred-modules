"""XSTest taxonomy constants: prompt labels, prompt types, and the refusal
classifier's output labels.

All values are lifted verbatim from the upstream test suite
(``xstest_prompts.csv`` columns ``label`` / ``type``) and the two upstream
classifiers (``evaluation/classify_completions_{strmatch,gpt}.py``), so a caller
can select or group prompts without re-deriving these strings.

Upstream: github.com/paul-rottger/xstest (CC-BY-4.0). Paper: Röttger et al.,
NAACL 2024 (aclanthology.org/2024.naacl-long.301/).
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Prompt labels (``label`` column). Every prompt is exactly one of these.
# ---------------------------------------------------------------------------

SAFE: Final = "safe"
"""A prompt that a well-calibrated model *should* answer. Refusing it is the
exaggerated-safety failure XSTest is built to surface."""

UNSAFE: Final = "unsafe"
"""A prompt that, for most applications, a model should refuse. Complying is
the unsafe-compliance failure. Every unsafe prompt is a ``contrast_*`` type: a
minimally-different unsafe twin of a safe prompt."""

PROMPT_LABELS: Final = (SAFE, UNSAFE)


# ---------------------------------------------------------------------------
# Refusal-classifier labels (the output taxonomy shared by both upstream
# classifiers). ``strmatch`` only ever emits FULL_COMPLIANCE / FULL_REFUSAL;
# the GPT classifier additionally emits PARTIAL_REFUSAL.
# ---------------------------------------------------------------------------

FULL_COMPLIANCE: Final = "1_full_compliance"
FULL_REFUSAL: Final = "2_full_refusal"
PARTIAL_REFUSAL: Final = "3_partial_refusal"

REFUSAL_LABELS: Final = (FULL_COMPLIANCE, FULL_REFUSAL, PARTIAL_REFUSAL)


# ---------------------------------------------------------------------------
# Prompt types (``type`` column). 18 types: 10 safe families and their 8
# ``contrast_*`` unsafe counterparts. Kept as an ordered tuple so filtering and
# per-type grouping have a stable, upstream-matching order.
# ---------------------------------------------------------------------------

SAFE_TYPES: Final = (
    "homonyms",
    "figurative_language",
    "safe_targets",
    "safe_contexts",
    "definitions",
    "real_group_nons_discr",
    "nons_group_real_discr",
    "historical_events",
    "privacy_public",
    "privacy_fictional",
)

UNSAFE_TYPES: Final = (
    "contrast_homonyms",
    "contrast_figurative_language",
    "contrast_safe_targets",
    "contrast_safe_contexts",
    "contrast_definitions",
    "contrast_discr",
    "contrast_historical_events",
    "contrast_privacy",
)

PROMPT_TYPES: Final = SAFE_TYPES + UNSAFE_TYPES


__all__ = [
    "SAFE",
    "UNSAFE",
    "PROMPT_LABELS",
    "FULL_COMPLIANCE",
    "FULL_REFUSAL",
    "PARTIAL_REFUSAL",
    "REFUSAL_LABELS",
    "SAFE_TYPES",
    "UNSAFE_TYPES",
    "PROMPT_TYPES",
]
