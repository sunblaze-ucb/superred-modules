"""Generic, claim-agnostic combinator for superred SecurityClaims.

``combine_claims`` unions the tasks of several ``SecurityClaim``s into one and
removes duplicate harmful behaviours so the combined claim never spends
attacker/judge tokens re-testing the same thing. Two layers of dedup:

1. **Exact / trivial-variant** -- tasks whose normalized goal text matches an
   already-kept task (case, whitespace). Cheap, deterministic, offline.
2. **Semantic near-duplicates** -- tasks whose goal is the *same harmful
   behaviour worded differently*, detected by embedding-cosine similarity above
   a threshold. This is the main reason to combine three independently-curated
   benchmarks: the same behaviour recurs across them under different phrasing.
   Enabled by passing an ``embedder``.

An optional per-category cap takes a stratified subset across each source's
native category taxonomy.

The combinator only touches the ``Task`` ABC surface (``task.goal.description``)
plus a pluggable ``category_getter`` / ``embedder``. Each surviving task keeps
its own native judge and configuration; there is no shared judge.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from superred.core.interfaces.security_claim import SecurityClaim
from superred.core.interfaces.task import Task

logger = logging.getLogger(__name__)

# Attribute names benchmark tasks use to expose their category, tried in
# order. HarmBench -> ``semantic_category``; StrongREJECT -> ``category``;
# SORRY-Bench -> ``category_name``. Namespaced by attribute name in the
# returned key so categories never collide across benchmarks.
_CATEGORY_ATTRS = ("semantic_category", "category", "category_name")


def normalize_goal(text: str) -> str:
    """Whitespace-collapsed, case-folded form used for exact duplicate keys.

    Catches exact and trivial-variant duplicates (case, surrounding/internal
    whitespace). Semantic near-duplicates are handled separately by the
    ``embedder`` path, not here.
    """
    return " ".join(text.split()).strip().casefold()


def category_of(task: Task) -> str:
    """Best-effort category key for a heterogeneous benchmark task.

    Returns ``"<attr>=<value>"`` for the first populated category attribute,
    so HarmBench/StrongREJECT/SORRY-Bench categories live in disjoint
    namespaces. Falls back to ``"_uncategorized"``.
    """
    for attr in _CATEGORY_ATTRS:
        value = getattr(task, attr, None)
        if value is not None and str(value).strip():
            return f"{attr}={value}"
    return "_uncategorized"


# ---------------------------------------------------------------------------
# Embedding-based semantic dedup
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to embedding vectors. ``embed`` must preserve input order."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 if either is zero)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class LiteLLMEmbedder:
    """Default ``Embedder`` backed by litellm, with a persistent text->vector cache.

    The cache (keyed by normalized text) makes dedup decisions identical and
    cheap across many claim builds (e.g. one per matrix cell) and lets repeat
    runs work offline. Embedding spend is out of band, like the judges.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        cache_path: str | Path | None = None,
        batch_size: int = 256,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._cache_path = Path(cache_path) if cache_path else None
        self._batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except (OSError, ValueError):
                logger.warning(
                    "embedding cache unreadable, starting empty: %s", self._cache_path
                )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [normalize_goal(t) for t in texts]
        missing = [t for t, k in zip(texts, keys, strict=True) if k not in self._cache]
        # Embed unique missing texts only.
        unique_missing: list[str] = []
        seen: set[str] = set()
        for t in missing:
            k = normalize_goal(t)
            if k not in seen:
                seen.add(k)
                unique_missing.append(t)
        if unique_missing:
            import litellm

            for start in range(0, len(unique_missing), self._batch_size):
                batch = unique_missing[start : start + self._batch_size]
                resp = litellm.embedding(
                    model=self._model,
                    input=batch,
                    api_base=self._api_base,
                    api_key=self._api_key,
                )
                # resp.data entries carry an "index" into `batch`.
                for item in resp.data:
                    idx = item["index"]
                    self._cache[normalize_goal(batch[idx])] = list(item["embedding"])
            self._persist_cache()
        return [self._cache[k] for k in keys]

    def _persist_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-merge with any concurrently-written cache, then atomic replace.
        merged = dict(self._cache)
        if self._cache_path.exists():
            try:
                merged = {**json.loads(self._cache_path.read_text()), **self._cache}
            except (OSError, ValueError):
                pass
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(merged))
        try:
            tmp.replace(self._cache_path)
        except OSError:
            pass
        self._cache = merged


# ---------------------------------------------------------------------------
# Records and stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskRecord:
    """One surviving task's provenance, for an external analysis manifest."""

    index: int  # 1-based position in the combined claim (== persisted file index)
    source: str  # source claim label
    category: str  # category key from ``category_getter``
    task_class: str  # concrete Task subclass name
    goal_preview: str  # first 120 chars of the goal text


@dataclass(frozen=True)
class SemanticDrop:
    """An auditable record of one task dropped as a semantic near-duplicate."""

    dropped_source: str
    dropped_goal_preview: str
    matched_kept_index: int  # 1-based index of the surviving task it matched
    matched_kept_preview: str
    similarity: float


@dataclass
class CombineStats:
    """Summary of a combine pass (for logging / provenance / threshold tuning)."""

    total_input: int = 0
    kept: int = 0
    dropped_duplicate: int = 0
    dropped_over_cap: int = 0
    dropped_semantic: int = 0
    kept_per_source: dict[str, int] = field(default_factory=dict)
    manifest: list[TaskRecord] = field(default_factory=list)
    semantic_drops: list[SemanticDrop] = field(default_factory=list)


def _resolve_cap(
    max_per_category: int | dict[str, int] | None, source_name: str
) -> int | None:
    """Effective per-category cap for *source_name* (int = uniform, dict = per-source)."""
    if isinstance(max_per_category, dict):
        return max_per_category.get(source_name)
    return max_per_category


def combine_claims(
    sources: Iterable[tuple[str, SecurityClaim]],
    *,
    dedup: bool = True,
    normalizer: Callable[[str], str] = normalize_goal,
    max_per_category: int | dict[str, int] | None = None,
    category_getter: Callable[[Task], str] = category_of,
    embedder: Embedder | None = None,
    similarity_threshold: float = 0.85,
    manifest_out: list[TaskRecord] | None = None,
    stats_out: list[CombineStats] | None = None,
) -> SecurityClaim:
    """Union labelled source claims into one deduplicated SecurityClaim.

    Stage 1 (per source, in order): drop exact/trivial duplicates (when
    ``dedup``) and apply the per-category cap. Stage 2 (if ``embedder``): drop
    semantic near-duplicates -- a candidate is dropped when its goal embeds
    within ``similarity_threshold`` cosine of an already-kept goal. First-seen
    wins throughout, so list sources in priority order.

    Args:
        sources: ``(label, claim)`` pairs; the label tags provenance.
        dedup: drop exact/trivial-variant duplicate goal text.
        normalizer: goal-text -> exact dedup key.
        max_per_category: keep at most this many tasks per category key. An
            ``int`` applies one cap to every source; a ``dict`` keyed by source
            label applies a per-source cap (missing keys = uncapped), so each
            benchmark can be sampled at a different depth.
        category_getter: task -> category key.
        embedder: if given, enable semantic near-duplicate removal.
        similarity_threshold: cosine at/above which two goals are the "same"
            behaviour. Higher = stricter (drops fewer). Tune via
            ``CombineStats.semantic_drops``.
        manifest_out / stats_out: optional sinks for provenance / stats.

    Returns:
        ``SecurityClaim.from_tasks(survivors)`` (non-empty, order preserved).

    Raises:
        ValueError: if no task survives.
    """
    stats = CombineStats()

    # --- Stage 1: exact dedup + per-category cap ---------------------------
    seen_keys: set[str] = set()
    category_counts: dict[str, int] = {}
    candidates: list[tuple[Task, str, str]] = []  # (task, source, category)

    for source_name, claim in sources:
        for task in claim:
            stats.total_input += 1
            key = normalizer(task.goal.description)
            if dedup and key in seen_keys:
                stats.dropped_duplicate += 1
                continue
            category = category_getter(task)
            cap = _resolve_cap(max_per_category, source_name)
            if cap is not None and category_counts.get(category, 0) >= cap:
                stats.dropped_over_cap += 1
                continue
            seen_keys.add(key)
            category_counts[category] = category_counts.get(category, 0) + 1
            candidates.append((task, source_name, category))

    # --- Stage 2: semantic near-duplicate removal --------------------------
    if embedder is not None and candidates:
        embeddings = embedder.embed([t.goal.description for t, _, _ in candidates])
        kept_idx: list[int] = []  # indices into `candidates` of survivors
        keep_mask: list[bool] = []
        for i, (task, source_name, _category) in enumerate(candidates):
            best_sim = -1.0
            best_j = -1
            for pos, j in enumerate(kept_idx):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim > best_sim:
                    best_sim = sim
                    best_j = pos  # 1-based survivor index assigned below
            if kept_idx and best_sim >= similarity_threshold:
                keep_mask.append(False)
                stats.dropped_semantic += 1
                matched_task = candidates[kept_idx[best_j]][0]
                stats.semantic_drops.append(
                    SemanticDrop(
                        dropped_source=source_name,
                        dropped_goal_preview=task.goal.description[:120],
                        matched_kept_index=best_j + 1,
                        matched_kept_preview=matched_task.goal.description[:120],
                        similarity=round(best_sim, 4),
                    )
                )
            else:
                keep_mask.append(True)
                kept_idx.append(i)
        candidates = [c for c, keep in zip(candidates, keep_mask, strict=True) if keep]
    elif embedder is not None and not candidates:
        pass  # nothing to embed

    # --- Build survivors + manifest ----------------------------------------
    survivors: list[Task] = []
    for task, source_name, category in candidates:
        survivors.append(task)
        stats.kept += 1
        stats.kept_per_source[source_name] = (
            stats.kept_per_source.get(source_name, 0) + 1
        )
        record = TaskRecord(
            index=len(survivors),
            source=source_name,
            category=category,
            task_class=type(task).__name__,
            goal_preview=task.goal.description[:120],
        )
        stats.manifest.append(record)
        if manifest_out is not None:
            manifest_out.append(record)

    if not survivors:
        raise ValueError(
            "combine_claims produced no tasks (all sources empty or fully deduped)"
        )

    logger.info(
        "combine_claims: kept %d of %d tasks "
        "(%d exact dup, %d over per-category cap, %d semantic dup); per-source: %s",
        stats.kept,
        stats.total_input,
        stats.dropped_duplicate,
        stats.dropped_over_cap,
        stats.dropped_semantic,
        stats.kept_per_source,
    )

    if stats_out is not None:
        stats_out.append(stats)

    return SecurityClaim.from_tasks(survivors)
