"""Strategy library: stores discovered jailbreak strategies.

Each entry is a dict with:

* ``Strategy`` — short name (str)
* ``Definition`` — one-sentence description (str)
* ``Example`` — list of jailbreak prompts that used this strategy (list[str])
* ``Score`` — list of score deltas (list[float]); ``len(Score) == len(Example)``

This is a slimmed-down port of upstream
``framework/library.py``:

* Same in-place merge semantics: adding the same ``Strategy`` name
  appends ``Example`` and ``Score`` rather than overwriting.
* Deliberate v1 simplification: the ``Embeddings`` field and the
  embedding-based :func:`retrieve` of upstream ``framework/retrival.py``
  are dropped. Retrieval ranks by *average score* across all examples
  of a strategy (the "key" the paper uses to decide between
  ``use_strategy`` / ``find_new_strategy`` / cold-start). See
  ``ASSUMPTIONS.md``.

The library is purely an in-memory data structure; persistence (the
upstream ``.pkl`` round-trip) is a workflow concern outside the
optimizer's contract.
"""

from __future__ import annotations

from typing import Any


HIGH_SCORE_THRESHOLD: float = 5.0
LOW_SCORE_THRESHOLD: float = 2.0


class StrategyLibrary:
    """In-memory strategy store with score-based retrieval."""

    def __init__(self) -> None:
        self._library: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        *,
        strategy: str,
        definition: str,
        example: str,
        score: float,
    ) -> None:
        """Add or extend an entry for ``strategy``.

        Mirrors upstream ``Library.add`` / ``Library.merge``: when the
        strategy name already exists, append the new ``Example`` and
        ``Score``; otherwise create a fresh entry.
        """
        if not strategy:
            raise ValueError("strategy name must be non-empty")
        existing = self._library.get(strategy)
        if existing is None:
            self._library[strategy] = {
                "Strategy": strategy,
                "Definition": definition,
                "Example": [example],
                "Score": [score],
            }
        else:
            existing["Example"].append(example)
            existing["Score"].append(score)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def all(self) -> dict[str, dict[str, Any]]:
        """Return the raw library (test/inspection only)."""
        return self._library

    def __len__(self) -> int:
        return len(self._library)

    def __contains__(self, name: object) -> bool:
        return name in self._library

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, k: int = 1) -> tuple[bool, list[dict[str, Any]]]:
        """Return ``(valid, strategies)`` for the next attacker turn.

        Score-based reduction of upstream's three-way decision in
        ``Retrieval.pop``:

        * If any strategy has avg score ≥ :data:`HIGH_SCORE_THRESHOLD`
          (5.0), return the single highest one with ``valid=True`` →
          attacker enters ``use_strategy`` mode.
        * Else, return strategies whose avg score is in
          ``[LOW_SCORE_THRESHOLD, HIGH_SCORE_THRESHOLD)`` with
          ``valid=True`` → still ``use_strategy`` (medium-effective
          strategies; same as upstream).
        * Else, return up to ``k`` lowest-avg-score strategies with
          ``valid=False`` → attacker enters ``find_new_strategy``
          (avoid these).

        Each returned dict contains ``Strategy``, ``Definition``,
        ``Example`` (the highest-scoring example for that strategy),
        and is ``Score``-stripped (matches upstream wire format that
        ``Attacker.use_strategy`` / ``find_new_strategy`` consumes).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._library:
            return True, []

        ranked: list[tuple[str, float]] = []
        for name, entry in self._library.items():
            scores = entry["Score"]
            if not scores:
                continue
            avg = sum(scores) / len(scores)
            ranked.append((name, avg))
        if not ranked:
            return True, []

        # Highest first.
        ranked.sort(key=lambda kv: kv[1], reverse=True)

        # Tier 1: any strategy with avg >= HIGH_SCORE_THRESHOLD wins.
        high_name, high_avg = ranked[0]
        if high_avg >= HIGH_SCORE_THRESHOLD:
            return True, [self._format_for_attacker(high_name)]

        # Tier 2: strategies with avg >= LOW_SCORE_THRESHOLD are
        # "medium effective" — return up to k.
        medium = [(n, a) for n, a in ranked if a >= LOW_SCORE_THRESHOLD]
        if medium:
            return True, [self._format_for_attacker(n) for n, _ in medium[:k]]

        # Tier 3: only ineffective strategies — return up to k for
        # ``find_new_strategy`` to avoid.
        return False, [self._format_for_attacker(n) for n, _ in ranked[:k]]

    def _format_for_attacker(self, name: str) -> dict[str, Any]:
        """Pick the highest-scoring example for ``name`` and pack the
        strategy into the dict shape ``Attacker.use_strategy`` expects.
        """
        entry = self._library[name]
        examples: list[str] = entry["Example"]
        scores: list[float] = entry["Score"]
        # Highest-scoring example first (ties: keep insertion order).
        idx = max(range(len(scores)), key=lambda i: scores[i])
        return {
            "Strategy": entry["Strategy"],
            "Definition": entry["Definition"],
            "Example": examples[idx],
        }


__all__ = [
    "HIGH_SCORE_THRESHOLD",
    "LOW_SCORE_THRESHOLD",
    "StrategyLibrary",
]
