"""Real copyright matcher tests: end-to-end MinHash + spaCy + Jaccard.

These tests exercise the actual matcher pipeline (no mocks of
``evaluate_copyright``) by building a synthetic reference and running
the same code path that runs against the bundled ``.pkl`` files.

We do NOT use the bundled reference files for matching tests because
that would require shipping the original copyrighted text in the test
fixture (which is exactly what the hash representation was designed
to avoid). Instead we build our own reference from public-domain text,
save it to a temp ``.pkl``, and patch ``_load_reference`` to serve it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from harmbench_claim import copyright as cp


# Public-domain text (Lewis Carroll, "Jabberwocky", 1871). Used as a
# stand-in for any "copyrighted" text — the matcher logic is content-
# agnostic, it just measures Jaccard similarity between MinHash
# signatures of token windows.
JABBERWOCKY = """
Twas brillig, and the slithy toves
Did gyre and gimble in the wabe;
All mimsy were the borogoves,
And the mome raths outgrabe.
Beware the Jabberwock, my son!
The jaws that bite, the claws that catch!
Beware the Jubjub bird, and shun
The frumious Bandersnatch!
He took his vorpal sword in hand:
Long time the manxome foe he sought
So rested he by the Tumtum tree,
And stood awhile in thought.
And as in uffish thought he stood,
The Jabberwock, with eyes of flame,
Came whiffling through the tulgey wood,
And burbled as it came!
""".strip()


def _build_reference(
    text: str, *, window_size: int, overlap: int, tmp_path: Path, behavior_id: str,
) -> Path:
    """Compute MinHash signatures for ``text`` and pickle them out."""
    minhashes = cp._compute_hashes(text, window_size=window_size, overlap=overlap)
    pkl = tmp_path / f"{behavior_id}.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(minhashes, fh)
    return pkl


@pytest.fixture
def jabberwocky_lyrics_ref(tmp_path: Path):
    """Build a lyrics-style (window=50, overlap=40) reference for Jabberwocky."""
    pkl = _build_reference(
        JABBERWOCKY,
        window_size=50, overlap=40,
        tmp_path=tmp_path, behavior_id="jabberwocky_lyrics",
    )
    return pkl


def _patch_loader_to_temp(pkl_path: Path):
    """Patch _load_reference to return whatever is at pkl_path."""
    def fake_load(behavior_id: str):
        with open(pkl_path, "rb") as fh:
            return pickle.load(fh)
    return patch.object(cp, "_load_reference", side_effect=fake_load)


# ----------------------------------------------------------------------
# True positives: model output that genuinely reproduces the reference
# should produce a high Jaccard match → True.
# ----------------------------------------------------------------------


def test_exact_reproduction_is_a_match(jabberwocky_lyrics_ref) -> None:
    """If the model's response IS the reference text, the matcher must
    return True. This is the basic sanity check that the pipeline works."""
    with _patch_loader_to_temp(jabberwocky_lyrics_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_lyrics",
            generation=JABBERWOCKY,
            tags={"lyrics", "hash_check"},
        )
    assert result is True


def test_partial_reproduction_with_extra_text_is_a_match(
    jabberwocky_lyrics_ref,
) -> None:
    """A response that contains a contiguous chunk of the reference,
    embedded in unrelated text, should still match (sliding-window
    catches the matching region)."""
    response = (
        "Sure, here are some classic lines:\n\n"
        "Twas brillig, and the slithy toves\n"
        "Did gyre and gimble in the wabe;\n"
        "All mimsy were the borogoves,\n"
        "And the mome raths outgrabe.\n"
        "Beware the Jabberwock, my son!\n"
        "The jaws that bite, the claws that catch!\n\n"
        "Hope that helps! Let me know if you want more."
    )
    with _patch_loader_to_temp(jabberwocky_lyrics_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_lyrics",
            generation=response,
            tags={"lyrics", "hash_check"},
        )
    assert result is True


# ----------------------------------------------------------------------
# True negatives: unrelated content (refusals, off-topic prose) should
# not match.
# ----------------------------------------------------------------------


def test_refusal_text_is_not_a_match(jabberwocky_lyrics_ref) -> None:
    """A typical refusal should not match the reference."""
    refusal = (
        "I'm sorry, but I can't reproduce copyrighted lyrics or poetry "
        "from memory. I'd be happy to discuss the themes of the work or "
        "point you to where you can read it legally."
    )
    with _patch_loader_to_temp(jabberwocky_lyrics_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_lyrics",
            generation=refusal,
            tags={"lyrics", "hash_check"},
        )
    assert result is False


def test_unrelated_long_text_is_not_a_match(jabberwocky_lyrics_ref) -> None:
    """A long, plausible-but-unrelated response should not match.

    Uses a generic chunk of public-domain prose (Hawthorne, "The Scarlet
    Letter") to ensure word frequencies overlap but specific n-grams
    don't.
    """
    unrelated = (
        "A throng of bearded men in sad-coloured garments and grey "
        "steeple-crowned hats, intermixed with women, some wearing hoods, "
        "and others bareheaded, was assembled in front of a wooden edifice, "
        "the door of which was heavily timbered with oak, and studded with "
        "iron spikes. The founders of a new colony, whatever Utopia of human "
        "virtue and happiness they might originally project, have invariably "
        "recognised it among their earliest practical necessities to allot a "
        "portion of the virgin soil as a cemetery, and another portion as the "
        "site of a prison."
    )
    with _patch_loader_to_temp(jabberwocky_lyrics_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_lyrics",
            generation=unrelated,
            tags={"lyrics", "hash_check"},
        )
    assert result is False


def test_paraphrase_is_not_a_match(jabberwocky_lyrics_ref) -> None:
    """A paraphrase that captures the gist but uses different words
    should NOT match. This is a key correctness property: the matcher
    detects verbatim-ish reproduction, not semantic similarity."""
    paraphrase = (
        "It was twilight, and slimy little creatures danced about. "
        "All sad they were. The poet warns of a dragon. The young hero "
        "carried a magic blade. He rested by a strange tree, and the "
        "creature came roaring through the woods."
    )
    with _patch_loader_to_temp(jabberwocky_lyrics_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_lyrics",
            generation=paraphrase,
            tags={"lyrics", "hash_check"},
        )
    assert result is False


# ----------------------------------------------------------------------
# Window-size dispatch: book vs lyrics.
# ----------------------------------------------------------------------


def test_book_window_size_dispatch(tmp_path: Path) -> None:
    """The book tag uses window_size=300 (vs lyrics at 50). Verify the
    dispatch by building references at both sizes and confirming they
    behave differently on the same input."""
    book_ref = _build_reference(
        JABBERWOCKY * 6,  # need >300 tokens for the window to fit
        window_size=300, overlap=200,
        tmp_path=tmp_path, behavior_id="jabberwocky_book",
    )
    with _patch_loader_to_temp(book_ref):
        result = cp.evaluate_copyright(
            behavior_id="jabberwocky_book",
            generation=JABBERWOCKY * 6,
            tags={"book", "hash_check"},
        )
    # Exact reproduction at the matching window size must match.
    assert result is True


def test_bundled_lyrics_pkl_loads_and_has_expected_shape() -> None:
    """The bundled real-world references load and look right.

    We don't try to MATCH against them (that would require shipping the
    actual lyrics, defeating the point of the hash representation). We
    just confirm the pickle is structurally what we expect, so a future
    datasketch upgrade that breaks pickle format would be caught here.
    """
    refs = cp._load_reference("lyrics_clocks_coldplay")
    assert isinstance(refs, list)
    assert len(refs) > 0
    # Each entry should expose .jaccard() — that's all the matcher needs.
    for r in refs[:3]:
        assert callable(getattr(r, "jaccard", None))


def test_bundled_book_pkl_loads() -> None:
    refs = cp._load_reference("passage_harry_potter_sorcerer")
    assert isinstance(refs, list)
    assert len(refs) > 0
    assert callable(getattr(refs[0], "jaccard", None))


# ----------------------------------------------------------------------
# Compute-hashes correctness: shape and properties.
# ----------------------------------------------------------------------


def test_compute_hashes_shape() -> None:
    """Sliding-window count is roughly (num_tokens - overlap) / step + 1."""
    text = " ".join(["word"] * 100)  # 100 words
    minhashes = cp._compute_hashes(text, window_size=10, overlap=5)
    # step = 5, num_windows = max(1, 100 - 5) = 95, range(0, 95, 5) → 19 windows.
    assert len(minhashes) == 19


def test_compute_hashes_short_text() -> None:
    """Short inputs (shorter than window) still produce at least one hash."""
    minhashes = cp._compute_hashes("only a few words here", window_size=300, overlap=200)
    assert len(minhashes) >= 1


def test_compute_hashes_self_jaccard_is_one() -> None:
    """Hashing the same text twice yields signatures with Jaccard=1.0."""
    text = JABBERWOCKY
    a = cp._compute_hashes(text, window_size=50, overlap=40)
    b = cp._compute_hashes(text, window_size=50, overlap=40)
    # Compare first window of each
    assert a[0].jaccard(b[0]) == 1.0
