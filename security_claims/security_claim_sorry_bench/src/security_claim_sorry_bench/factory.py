"""Factory for the SORRY-Bench :class:`SecurityClaim`.

``sorry_bench_claim(...)`` is the user-facing entry point. It loads the
gated ``question.jsonl`` (either from a caller-supplied path or via HF
Hub lazy-download), filters rows per the optional ``categories`` /
``prompts_per_category`` / ``question_ids`` / ``subset`` kwargs, and
produces one :class:`SorryBenchTask` per surviving row, all bundled
into a :class:`SecurityClaim`.

The dataset license (custom non-redistributable) forbids vendoring the
prompts inside this package, so loading is **always** at runtime from
the user's HuggingFace authentication or a pre-downloaded path.

Verified end-to-end against the gated 202503 dataset, commit
``612a4e1f45db8adf884fa62318ddf9fa1c6e75e9`` (pinned in
:data:`DATASET_REVISION`).
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal, cast

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.target import Target
from superred.core.interfaces.task import Task
from superred.core.types.llm import LLMConfig

from security_claim_sorry_bench.categories import CATEGORY_NAMES
from security_claim_sorry_bench.judge import Judge, LLMAsJudge
from security_claim_sorry_bench.system_prompts import sorry_bench_system_prompt
from security_claim_sorry_bench.task import SorryBenchTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset pinning. The 202503 release was last modified 2025-02-28.
# Pinning the commit guards against silent upstream drift.
# Override via the ``revision`` kwarg if a newer revision is desired.
# ---------------------------------------------------------------------------

DATASET_REPO_ID: str = "sorry-bench/sorry-bench-202503"
DATASET_REVISION: str = "612a4e1f45db8adf884fa62318ddf9fa1c6e75e9"
DATASET_FILENAME: str = "question.jsonl"
DEFAULT_JUDGE_MODEL: str = "gpt-4-1106-preview"

# ---------------------------------------------------------------------------
# Subset pinning. Two disjoint halves of the 440-prompt benchmark, each
# evenly spread across all 44 categories (5 prompts/category each).
#
# The split uses ``question_id`` parity. The dataset invariant
# ``(question_id - 1) // 10 + 1 == category`` (asserted in
# ``test_factory.TestRowSchemaInvariant``) guarantees that each category
# contributes exactly 5 odd and 5 even q_ids, so both subsets are
# perfectly stratified by category and (by extension) by the 4 high-level
# domains in ``CATEGORY_DOMAINS``.
#
# Use case: run subset A first as a cheaper pilot (220 prompts, ~half
# the judge spend), inspect, then run subset B for the rest of the
# benchmark with no overlap.
# ---------------------------------------------------------------------------

SUBSET_A_QUESTION_IDS: frozenset[int] = frozenset(range(1, 441, 2))
SUBSET_B_QUESTION_IDS: frozenset[int] = frozenset(range(2, 441, 2))

Subset = Literal["a", "b"]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def sorry_bench_claim(
    *,
    target_model_id: str,
    dataset_path: str | None = None,
    dataset_revision: str = DATASET_REVISION,
    judge: Judge | None = None,
    judge_llm_config: LLMConfig | None = None,
    categories: list[int] | None = None,
    prompts_per_category: int | None = None,
    question_ids: list[int] | None = None,
    subset: Subset | None = None,
) -> SecurityClaim[Target]:
    """Build a :class:`SecurityClaim` evaluating SORRY-Bench against a chatbot.

    Args:
        target_model_id: The litellm-style model id of the
            :class:`ChatbotTarget` under test (e.g. ``"gpt-4o-2024-05-13"``).
            Used to look up the SORRY-Bench-faithful system prompt; see
            :func:`sorry_bench_system_prompt` for the lookup table.
        dataset_path: Absolute path to a downloaded ``question.jsonl``.
            If omitted, the dataset is fetched from HuggingFace at
            ``dataset_revision`` (gated; the user must accept the gate
            and authenticate via ``hf auth login`` or set ``HF_TOKEN``).
        dataset_revision: HuggingFace dataset commit hash to pin.
            Defaults to :data:`DATASET_REVISION` (a known-good commit).
        judge: Pluggable :class:`Judge`. If ``None``, an
            :class:`LLMAsJudge` is built from ``judge_llm_config``.
        judge_llm_config: Required if ``judge`` is ``None``. The judge
            uses its own :class:`LLMClient` (separate from the
            optimizer's), so judge cost is NOT tracked in the
            controller's ``RunResult.llm_usage`` — it surfaces in the
            evaluation rationale instead.
        categories: 1-indexed category ids to keep (e.g. ``[1, 5, 12]``).
            If ``None``, all 44 categories are included.
        prompts_per_category: Limit prompts per category (after the
            category and subset filters). If ``None``, all surviving
            prompts per category are kept.
        question_ids: 1-indexed prompt ids to keep. If provided, this
            overrides ``categories`` / ``prompts_per_category`` /
            ``subset`` and returns only the listed prompts.
        subset: Selects one of two disjoint halves, each evenly spread
            across all 44 categories (5 prompts/category, 220 total).
            ``"a"`` keeps odd ``question_id`` rows, ``"b"`` keeps even
            ones; together they reconstruct the full 440-prompt
            benchmark with no overlap. Composes with ``categories``
            (the subset is applied to the surviving categories) and
            ``prompts_per_category`` (the truncation is applied after).
            Intended use: run subset ``"a"`` as a cheaper pilot, then
            run ``"b"`` to cover the rest. Ignored when ``question_ids``
            is supplied. See :data:`SUBSET_A_QUESTION_IDS` /
            :data:`SUBSET_B_QUESTION_IDS` for the exact id lists.

    Returns:
        A :class:`SecurityClaim` bundling one :class:`SorryBenchTask`
        per surviving row.

    Raises:
        ValueError: If neither ``judge`` nor ``judge_llm_config`` is
            provided, if the resulting task list is empty, or if a row
            in the dataset has an out-of-range category.
        RuntimeError: If the dataset cannot be loaded (no path and no
            HF auth).
    """
    if judge is None:
        if judge_llm_config is None:
            raise ValueError(
                "Must provide either `judge` or `judge_llm_config`. "
                f"Pass `judge_llm_config=LLMConfig(model={DEFAULT_JUDGE_MODEL!r}, ...)` "
                "for the canonical GPT-4 judge."
            )
        judge = LLMAsJudge.from_config(judge_llm_config)

    rows = _load_dataset(dataset_path, revision=dataset_revision)
    rows = _filter_rows(
        rows,
        categories=categories,
        prompts_per_category=prompts_per_category,
        question_ids=question_ids,
        subset=subset,
    )
    _warn_if_not_base_dataset(rows)

    system_prompt = sorry_bench_system_prompt(target_model_id)
    tasks: list[SorryBenchTask] = []
    for row in rows:
        try:
            cat_id = int(row["category"])
            q_id = int(row["question_id"])
            question = row["turns"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed row %r: %s", row, exc)
            continue

        if not 1 <= cat_id <= 44:
            logger.warning(
                "Skipping row q_id=%s with out-of-range category=%r",
                q_id,
                row["category"],
            )
            continue

        tasks.append(
            SorryBenchTask(
                question_id=q_id,
                question=question,
                category_id=cat_id,
                category_name=CATEGORY_NAMES[cat_id - 1],
                system_prompt=system_prompt,
                judge=judge,
            )
        )

    if not tasks:
        raise ValueError(
            "No tasks produced. Check that the dataset_path / HF lazy-load "
            "returned rows and that the filter kwargs are not over-restrictive."
        )

    # Cast widens SecurityClaim[ChatbotTarget] to SecurityClaim[Target] so the
    # Controller (which takes SecurityClaim[Target]) accepts it. The underlying
    # tasks remain Task[ChatbotTarget]; only the variance annotation widens.
    return SecurityClaim.from_tasks(cast(list[Task[Target]], tasks))


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def _load_dataset(
    path: str | None, *, revision: str = DATASET_REVISION
) -> list[dict[str, Any]]:
    """Load ``question.jsonl`` from a local path or fall back to HF Hub.

    The SORRY-Bench HF dataset is published as raw JSONL files in a git
    repo (the upstream README instructs ``git clone <repo>``); it is NOT
    structured as a ``datasets``-library Dataset with named splits. We
    therefore use ``huggingface_hub.hf_hub_download`` to fetch
    ``question.jsonl`` directly, NOT ``datasets.load_dataset``.

    Authentication: ``huggingface_hub`` auto-detects either ``HF_TOKEN``
    or ``~/.cache/huggingface/token`` (set by ``hf auth login``). The
    user must have accepted the gate at
    https://huggingface.co/datasets/sorry-bench/sorry-bench-202503 first.

    Verified end-to-end against the gated 202503 dataset.
    """
    if path is not None:
        return _read_jsonl(path)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover — `huggingface_hub` is a hard dep
        raise RuntimeError(
            "huggingface_hub is required for HF lazy-load. Install with "
            "`pip install huggingface_hub>=0.20`, or pass an explicit "
            "`dataset_path=...`."
        ) from exc

    try:
        local = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=DATASET_FILENAME,
            repo_type="dataset",
            revision=revision,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {DATASET_REPO_ID}/{DATASET_FILENAME}@{revision} "
            "from HuggingFace. Either pass `dataset_path=...` to a downloaded "
            "copy, or accept the gate at "
            f"https://huggingface.co/datasets/{DATASET_REPO_ID} and authenticate "
            "via `hf auth login` (or set HF_TOKEN)."
        ) from exc

    return _read_jsonl(local)


def _read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts; skip blank lines."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------


def _filter_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    categories: list[int] | None,
    prompts_per_category: int | None,
    question_ids: list[int] | None,
    subset: Subset | None,
) -> list[dict[str, Any]]:
    """Apply the four optional filters from ``sorry_bench_claim`` kwargs.

    Filter precedence:

    1. ``question_ids`` is exclusive — if provided, only rows with a
       matching id survive (other filters ignored).
    2. Otherwise, in order: ``categories`` drops rows whose category is
       not in the set; ``subset`` keeps the half whose ``question_id``
       parity matches (``"a"`` = odd, ``"b"`` = even); finally
       ``prompts_per_category`` truncates each surviving category by
       the lowest ``question_id``.

    Note: the canonical dataset stores ``category`` as a numeric **string**
    (``"1"``..``"44"``); ``question_id`` is an int. This filter casts both
    defensively.
    """
    rows_list: list[dict[str, Any]] = [dict(r) for r in rows]

    if question_ids is not None:
        wanted_ids = set(question_ids)
        return [r for r in rows_list if int(r["question_id"]) in wanted_ids]

    if categories is not None:
        wanted_cats = set(categories)
        rows_list = [r for r in rows_list if int(r["category"]) in wanted_cats]

    if subset is not None:
        wanted_subset_ids: frozenset[int]
        if subset == "a":
            wanted_subset_ids = SUBSET_A_QUESTION_IDS
        elif subset == "b":
            wanted_subset_ids = SUBSET_B_QUESTION_IDS
        else:
            raise ValueError(f"subset must be 'a', 'b', or None, got {subset!r}")
        rows_list = [r for r in rows_list if int(r["question_id"]) in wanted_subset_ids]

    if prompts_per_category is not None:
        if prompts_per_category < 0:
            raise ValueError(
                f"prompts_per_category must be >= 0, got {prompts_per_category}"
            )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows_list:
            grouped[int(r["category"])].append(r)
        out: list[dict[str, Any]] = []
        for cat_id in sorted(grouped):
            ordered = sorted(grouped[cat_id], key=lambda r: int(r["question_id"]))
            out.extend(ordered[:prompts_per_category])
        return out

    return rows_list


def _warn_if_not_base_dataset(rows: Iterable[Mapping[str, Any]]) -> None:
    """Warn loudly if the loaded dataset is one of the 20 mutated variants.

    The canonical 202503 release publishes 20 paraphrased prompt files
    (e.g. ``question_caesar.jsonl``, ``question_translate-fr.jsonl``)
    alongside the base ``question.jsonl``. Each row has a ``prompt_style``
    field — ``"base"`` for the canonical 440 prompts, otherwise the
    mutation suffix.

    v1 of this module supports only the base prompts. If the user
    accidentally points ``dataset_path=`` at a mutated file, we'd
    silently produce 440 tasks whose internal state looks identical to
    base prompts but whose ``question`` is, say, Caesar-cipher text.
    Without post-decoding the model's response (the upstream
    ``data/sorry_bench/mutate/decode.py`` pipeline), the judge would
    score every prompt 0. That's a baseline-incomparable result.

    We therefore warn — once, with a count — when any row's
    ``prompt_style`` is not ``"base"``. Mutated-dataset support is v2.
    """
    non_base: dict[str, int] = defaultdict(int)
    for row in rows:
        style = row.get("prompt_style")
        if isinstance(style, str) and style != "base":
            non_base[style] += 1
    if non_base:
        styles = ", ".join(f"{k}={v}" for k, v in sorted(non_base.items()))
        logger.warning(
            "Loaded dataset contains non-base prompt_style values "
            "(%s). v1 supports only base prompts; mutated prompts may "
            "produce baseline-incomparable scores because no response "
            "post-decoding is performed. Consider passing "
            "dataset_path=<path to question.jsonl> instead of a "
            "question_<mutation>.jsonl variant.",
            styles,
        )


__all__ = [
    "sorry_bench_claim",
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "DATASET_FILENAME",
    "DEFAULT_JUDGE_MODEL",
    "SUBSET_A_QUESTION_IDS",
    "SUBSET_B_QUESTION_IDS",
    "Subset",
]
