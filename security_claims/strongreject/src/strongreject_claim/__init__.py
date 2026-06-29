"""StrongREJECT (Souly et al., NeurIPS 2024) as a superred SecurityClaim.

Public surface: eight factory functions returning
``SecurityClaim[ChatbotTarget]``. Six wrap individual StrongREJECT
categories; ``strongreject_full_claim`` composes them via
``SecurityClaim.from_claims``; ``strongreject_small_claim`` loads the
60-prompt curated subset from a separate vendored CSV.

Citation::

    @inproceedings{souly2024strongreject,
        title={A Strong{REJECT} for Empty Jailbreaks},
        author={Souly, Alexandra and Lu, Qingyuan and Bowen, Dillon and
                Trinh, Tu and Hsieh, Elvis and Pandey, Sana and
                Abbeel, Pieter and Svegliato, Justin and Emmons, Scott
                and Watkins, Olivia and Toyer, Sam},
        booktitle={The Thirty-eighth Annual Conference on
                   Neural Information Processing Systems},
        year={2024}
    }

Upstream code (MIT): https://github.com/dsbowen/strong_reject (canonical)
and https://github.com/alexandrasouly/strongreject (deprecated for code,
still the dataset host). See ``LICENSES/`` and ``LICENSES/NOTICE.md``
for the full attribution.
"""

from __future__ import annotations

from typing import Final

from chatbot_target import ChatbotTarget
from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

from strongreject_claim.judge import MODEL, JudgeConfig
from strongreject_claim.loader import (
    StrongREJECTRow,
    load_small_rows,
    rows_for_category,
)
from strongreject_claim.task import StrongREJECTTask

# ---------------------------------------------------------------------------
# Canonical category labels (lowercased; match the loader's normalisation)
# ---------------------------------------------------------------------------

_DISINFORMATION: Final = "disinformation and deception"
_HATE: Final = "hate, harassment and discrimination"
_ILLEGAL_GOODS: Final = "illegal goods and services"
_NONVIOLENT_CRIMES: Final = "non-violent crimes"
_SEXUAL_CONTENT: Final = "sexual content"
_VIOLENCE: Final = "violence"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tasks(
    rows: list[StrongREJECTRow],
    judge_config: JudgeConfig,
    success_threshold: float,
) -> list[Task[ChatbotTarget]]:
    """Convert dataset rows into a list of stateless Tasks."""
    return [
        StrongREJECTTask(
            forbidden_prompt=row["forbidden_prompt"],
            category=row["category"],
            judge_config=judge_config,
            success_threshold=success_threshold,
        )
        for row in rows
    ]


def _category_claim(
    category: str,
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float,
    judge_max_retries: int,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """Build the SecurityClaim for one StrongREJECT category."""
    judge_config = JudgeConfig(
        api_base=judge_api_base,
        api_key=judge_api_key,
        max_retries=judge_max_retries,
        model=judge_model,
    )
    rows = rows_for_category(category)
    tasks = _build_tasks(rows, judge_config, success_threshold)
    return SecurityClaim.from_tasks(tasks)


# ---------------------------------------------------------------------------
# Per-category factories
# ---------------------------------------------------------------------------


def strongreject_disinformation_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Disinformation and deception* category (50 prompts)."""
    return _category_claim(
        _DISINFORMATION,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


def strongreject_hate_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Hate, harassment and discrimination* category (50 prompts)."""
    return _category_claim(
        _HATE,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


def strongreject_illegal_goods_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Illegal goods and services* category (50 prompts)."""
    return _category_claim(
        _ILLEGAL_GOODS,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


def strongreject_nonviolent_crimes_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Non-violent crimes* category (59 prompts)."""
    return _category_claim(
        _NONVIOLENT_CRIMES,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


def strongreject_sexual_content_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Sexual content* category (50 prompts)."""
    return _category_claim(
        _SEXUAL_CONTENT,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


def strongreject_violence_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """StrongREJECT *Violence* category (54 prompts)."""
    return _category_claim(
        _VIOLENCE,
        judge_api_base=judge_api_base,
        judge_api_key=judge_api_key,
        success_threshold=success_threshold,
        judge_max_retries=judge_max_retries,
        judge_model=judge_model,
    )


# ---------------------------------------------------------------------------
# Top-level factories
# ---------------------------------------------------------------------------


def strongreject_full_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """The full 313-prompt StrongREJECT benchmark.

    Implemented as ``SecurityClaim.from_claims`` over the six per-category
    sub-claims. Iteration order: disinformation, hate, illegal_goods,
    nonviolent_crimes, sexual_content, violence. Within a category, rows
    follow CSV order. Determinism guaranteed because each factory call
    builds fresh sub-claims from the same vendored CSV.
    """
    sub_claims = [
        strongreject_disinformation_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
        strongreject_hate_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
        strongreject_illegal_goods_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
        strongreject_nonviolent_crimes_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
        strongreject_sexual_content_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
        strongreject_violence_claim(
            judge_api_base=judge_api_base,
            judge_api_key=judge_api_key,
            success_threshold=success_threshold,
            judge_max_retries=judge_max_retries,
            judge_model=judge_model,
        ),
    ]
    return SecurityClaim.from_claims(sub_claims)


def strongreject_small_claim(
    *,
    judge_api_base: str,
    judge_api_key: str,
    success_threshold: float = 1.0,
    judge_max_retries: int = 5,
    judge_model: str = MODEL,
) -> SecurityClaim[ChatbotTarget]:
    """The 60-prompt curated StrongREJECT-small subset.

    This is a 60-prompt strict subset of the full 313 (10 per category,
    all ``source=custom``). We ship the small CSV verbatim from upstream
    rather than deriving it from the full set: byte-equality with the
    upstream file preserves reproducibility against published numbers,
    and the upstream casing inconsistency (small uses
    ``Disinformation and Deception``, full uses
    ``Disinformation and deception``) means programmatic derivation
    would still need normalisation to recover the same subset.
    """
    judge_config = JudgeConfig(
        api_base=judge_api_base,
        api_key=judge_api_key,
        max_retries=judge_max_retries,
        model=judge_model,
    )
    rows = load_small_rows()
    tasks = _build_tasks(rows, judge_config, success_threshold)
    return SecurityClaim.from_tasks(tasks)


__all__ = [
    "strongreject_disinformation_claim",
    "strongreject_hate_claim",
    "strongreject_illegal_goods_claim",
    "strongreject_nonviolent_crimes_claim",
    "strongreject_sexual_content_claim",
    "strongreject_violence_claim",
    "strongreject_full_claim",
    "strongreject_small_claim",
]
