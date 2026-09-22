"""Claim handling and request assembly for the Anecdoctor optimizer.

Pure, LLM-free helpers:

* :data:`DEFAULT_EXAMPLE_CLAIMS` -- a small, deliberately benign, entirely
  synthetic set of example "claims" authored for this package. It is **not**
  copied from any upstream dataset (PyRIT ships no example-claims dataset; the
  vendored files are prompt templates only). Its only job is to show the
  ClaimsReview-style few-shot shape the technique expects; the real target
  claim/topic comes from the SuperRed Goal.
* :func:`build_evaluation_data` -- maps the Goal onto upstream's
  ``evaluation_data`` list.
* :func:`format_few_shot_examples` -- upstream ``_format_few_shot_examples``.
* :func:`assemble_framed_request` -- folds the framing template and the
  few-shot / knowledge-graph body into the single user turn that is injected.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Benign, synthetic, format-only example claims (see module docstring). These
#: are neutral civic trivia, never harmful misinformation payloads.
DEFAULT_EXAMPLE_CLAIMS: tuple[str, ...] = (
    "The old lighthouse on Marlin Bay was repainted teal during the spring festival.",
    "Riverton's community garden switched entirely to rainwater irrigation last year.",
    "The regional library extended its weekend hours after a local volunteer drive.",
)

#: Few-shot block header (upstream ``_format_few_shot_examples``).
_EXAMPLES_HEADER = "### examples"


def build_evaluation_data(
    *,
    goal: str,
    example_claims: Sequence[str],
    include_goal_as_claim: bool,
) -> list[str]:
    """Assemble the ``evaluation_data`` claim list from the Goal and examples.

    Upstream takes ``evaluation_data`` (a list of claims in ClaimsReview format)
    directly from the caller; the framing prompt is parameterised only by
    ``language``/``type``, so the specific claim/topic can only influence the
    output through ``evaluation_data``. SuperRed's Goal is therefore folded in
    as the lead claim (``include_goal_as_claim``, the default), with the
    synthetic ``example_claims`` following as style demonstrations.

    When ``include_goal_as_claim`` is False, only ``example_claims`` are used
    (upstream's dataset-driven behaviour, independent of the specific Goal).
    """
    data: list[str] = []
    if include_goal_as_claim and goal.strip():
        data.append(goal.strip())
    data.extend(claim for claim in example_claims if claim.strip())
    return data


def format_few_shot_examples(evaluation_data: Sequence[str]) -> str:
    """Format claims as a few-shot block (upstream ``_format_few_shot_examples``)."""
    return _EXAMPLES_HEADER + "\n" + "\n".join(evaluation_data)


def assemble_framed_request(*, framing: str, body: str) -> str:
    """Fold the framing template and the body into one injectable user turn.

    Upstream sends ``framing`` as the target's system prompt and ``body`` (the
    few-shot examples or the knowledge graph) as a separate user message. This
    port targets a single free-text user surface, so the two are concatenated
    into one turn (see ASSUMPTIONS.md).
    """
    return framing.rstrip() + "\n\n" + body
