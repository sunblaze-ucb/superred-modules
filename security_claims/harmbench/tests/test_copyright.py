"""Copyright matcher tests: dispatch logic, error paths, structural sanity.

We do NOT test the MinHash + Jaccard math itself (that's a verbatim port
from upstream's eval_utils.py and re-testing it would just reimplement
the algorithm). We test the wiring: tag-based dispatch, helpful errors,
and that the bundled reference files are loadable.
"""

from __future__ import annotations

import pytest

from harmbench_claim import copyright as cp


def test_invalid_tags_raise() -> None:
    with pytest.raises(ValueError, match="must have 'book' or 'lyrics' tag"):
        cp.evaluate_copyright(
            behavior_id="anything", generation="x", tags={"hash_check"},
        )


def test_invalid_tags_rejects_unrelated() -> None:
    with pytest.raises(ValueError, match="must have 'book' or 'lyrics' tag"):
        cp.evaluate_copyright(
            behavior_id="anything", generation="x", tags={"context"},
        )


def test_window_step_must_be_positive() -> None:
    """Defensive: window_size <= overlap would loop forever."""
    with pytest.raises(ValueError, match="window_size .* must exceed overlap"):
        cp._compute_hashes("hello world", window_size=10, overlap=10)


def test_unknown_behavior_raises_filenotfound() -> None:
    """Looking up a non-existent reference .pkl should fail loudly."""
    with pytest.raises(FileNotFoundError):
        cp.evaluate_copyright(
            behavior_id="not_a_real_behavior_id_at_all",
            generation="x",
            tags={"book", "hash_check"},
        )


def test_bundled_reference_pkls_exist_for_book_tag() -> None:
    """Spot-check one bundled passage_*.pkl loads cleanly."""
    refs = cp._load_reference("passage_crawdads_sing")
    # Should be a non-empty list of MinHash-shaped objects.
    assert isinstance(refs, list)
    assert len(refs) > 0
    # Duck-typing: each reference should expose .jaccard().
    assert callable(getattr(refs[0], "jaccard", None))


def test_bundled_reference_pkls_exist_for_lyrics_tag() -> None:
    """Spot-check one bundled lyrics_*.pkl loads cleanly."""
    refs = cp._load_reference("lyrics_99_problems")
    assert isinstance(refs, list)
    assert len(refs) > 0
    assert callable(getattr(refs[0], "jaccard", None))
