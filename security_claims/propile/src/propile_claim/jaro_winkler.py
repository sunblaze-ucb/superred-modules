"""Jaro-Winkler string similarity — a dependency-free reimplementation.

Garak's ProPILE detector imports ``jaro_winkler_similarity`` from
``nltk.metrics.distance``. Rather than pull in nltk (a heavy dependency whose only
use here is this one pure function), this module reimplements the exact same
algorithm. It is verified behaviourally identical to nltk's implementation across
40,000+ random, near-match, PII-shaped, and edge-case string pairs
(``tests/test_jaro_winkler_differential.py``, opt-in against nltk).

Standard Jaro-Winkler (Winkler 1990): prefix scaling ``p = 0.1`` and prefix cap
``max_l = 4``.
"""

from __future__ import annotations


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity of two strings, matching ``nltk.metrics.distance.jaro_similarity``."""
    if s1 == s2:
        return 1.0
    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    match_bound = max(len_s1, len_s2) // 2 - 1
    matches = 0
    transpositions = 0
    flagged_1: list[int] = []  # positions in s1 that matched
    matched_2: set[int] = set()  # positions in s2 that matched (set -> O(1) membership)

    for i in range(len_s1):
        upperbound = min(i + match_bound, len_s2 - 1)
        lowerbound = max(0, i - match_bound)
        for j in range(lowerbound, upperbound + 1):
            if s1[i] == s2[j] and j not in matched_2:
                matches += 1
                flagged_1.append(i)
                matched_2.add(j)
                break

    flagged_2 = sorted(matched_2)
    for i, j in zip(flagged_1, flagged_2):
        if s1[i] != s2[j]:
            transpositions += 1

    if matches == 0:
        return 0.0
    return (1 / 3) * (
        matches / len_s1 + matches / len_s2 + (matches - transpositions // 2) / matches
    )


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Jaro-Winkler similarity, matching ``nltk.metrics.distance.jaro_winkler_similarity``.

    ``jaro_winkler = jaro + l * p * (1 - jaro)`` where ``l`` is the common-prefix
    length capped at ``max_l``.
    """
    jaro_sim = jaro_similarity(s1, s2)
    prefix = 0
    for a, b in zip(s1, s2):
        if a == b:
            prefix += 1
        else:
            break
        if prefix == max_l:
            break
    return jaro_sim + (prefix * p * (1 - jaro_sim))


__all__ = ["jaro_similarity", "jaro_winkler_similarity"]
