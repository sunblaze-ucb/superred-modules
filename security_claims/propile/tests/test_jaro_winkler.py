"""Jaro-Winkler tests: canonical reference values + opt-in differential vs nltk."""

from __future__ import annotations

import random
import string

import pytest

from propile_claim.jaro_winkler import jaro_similarity, jaro_winkler_similarity


def test_identity_and_empty() -> None:
    assert jaro_winkler_similarity("abc", "abc") == 1.0
    assert jaro_similarity("abc", "abc") == 1.0
    assert jaro_winkler_similarity("", "abc") == 0.0
    assert jaro_winkler_similarity("abc", "") == 0.0
    assert jaro_winkler_similarity("", "") == 1.0  # identical (both empty)


def test_reference_values() -> None:
    # From Winkler (1990) / the census comparator table used by nltk's tests.
    assert round(jaro_winkler_similarity("martha", "marhta"), 4) == 0.9611
    assert round(jaro_winkler_similarity("dwayne", "duane"), 4) == 0.8400
    assert round(jaro_winkler_similarity("dixon", "dicksonx"), 4) == 0.8133


def test_no_common_prefix_equals_jaro() -> None:
    # prefix length 0 -> jaro_winkler == jaro
    a, b = "abcxyz", "zyxcba"
    assert jaro_winkler_similarity(a, b) == jaro_similarity(a, b)


# -- opt-in differential against nltk (skipped if nltk is not installed) ------
def test_matches_nltk_over_random_corpus() -> None:
    nltk_distance = pytest.importorskip("nltk.metrics.distance")
    jw_nltk = nltk_distance.jaro_winkler_similarity

    random.seed(7)
    alpha = string.ascii_lowercase + string.digits + ".@ "
    mismatches = 0
    for _ in range(3000):
        a = "".join(random.choice(alpha) for _ in range(random.randint(0, 16)))
        b = "".join(random.choice(alpha) for _ in range(random.randint(0, 16)))
        if abs(jaro_winkler_similarity(a, b) - jw_nltk(a, b)) > 1e-9:
            mismatches += 1
    assert mismatches == 0
