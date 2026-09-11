"""xstest_claim: the XSTest exaggerated-safety test suite as a superred SecurityClaim.

One Task per XSTest prompt (450 total: 250 ``safe`` + 200 ``unsafe`` contrasts),
bound to the shared ``chatbot`` target. The win condition is upstream's
**deterministic** string-match refusal classifier
(:func:`~xstest_claim.judge.strmatch_label`) — a pure function of the model
response, no LLM in the default path. Polarity follows superred's convention
(higher = worse behaviour): refusing a *safe* prompt is over-refusal; complying
with an *unsafe* prompt is unsafe compliance.

Run it with the ``goal_passthrough`` optimizer so each prompt is delivered to
the target verbatim (over-refusal measurement requires the safe prompt to reach
the model unmodified).

Provenance: ported from ``paul-rottger/xstest`` (CC-BY-4.0). Cite Röttger et al.,
NAACL 2024 (aclanthology.org/2024.naacl-long.301/):

    @inproceedings{rottger-etal-2024-xstest,
        title = "{XST}est: A Test Suite for Identifying Exaggerated Safety
                 Behaviours in Large Language Models",
        author = {R{\"o}ttger, Paul and Kirk, Hannah and Vidgen, Bertie and
                  Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
        booktitle = "Proceedings of NAACL-HLT 2024",
        year = "2024",
        pages = "5377--5400",
    }
"""

from __future__ import annotations

from xstest_claim.categories import (
    FULL_COMPLIANCE,
    FULL_REFUSAL,
    PARTIAL_REFUSAL,
    PROMPT_LABELS,
    PROMPT_TYPES,
    REFUSAL_LABELS,
    SAFE,
    SAFE_TYPES,
    UNSAFE,
    UNSAFE_TYPES,
)
from xstest_claim.factory import (
    xstest_claim,
    xstest_full_claim,
    xstest_safe_claim,
    xstest_unsafe_claim,
)
from xstest_claim.judge import (
    GPTJudgeConfig,
    GPTJudgeError,
    Verdict,
    attack_magnitude,
    gpt_label,
    judge_gpt,
    judge_strmatch,
    normalize_gpt_label,
    refusal_fraction,
    render_gpt_prompt,
    strmatch_label,
)
from xstest_claim.loader import (
    XSTestRow,
    actual_sha256,
    expected_sha256,
    load_gpt_prompt_template,
    load_rows,
)
from xstest_claim.task import XSTestNotScorableError, XSTestTask

__all__ = [
    # taxonomy
    "SAFE",
    "UNSAFE",
    "PROMPT_LABELS",
    "PROMPT_TYPES",
    "SAFE_TYPES",
    "UNSAFE_TYPES",
    "FULL_COMPLIANCE",
    "FULL_REFUSAL",
    "PARTIAL_REFUSAL",
    "REFUSAL_LABELS",
    # judge
    "strmatch_label",
    "refusal_fraction",
    "attack_magnitude",
    "Verdict",
    "judge_strmatch",
    "GPTJudgeConfig",
    "GPTJudgeError",
    "render_gpt_prompt",
    "normalize_gpt_label",
    "gpt_label",
    "judge_gpt",
    # loader
    "XSTestRow",
    "load_rows",
    "load_gpt_prompt_template",
    "expected_sha256",
    "actual_sha256",
    # task + factories
    "XSTestTask",
    "XSTestNotScorableError",
    "xstest_claim",
    "xstest_safe_claim",
    "xstest_unsafe_claim",
    "xstest_full_claim",
]
