"""CSV loader behaviour: encoding, normalisation, filtering."""

from __future__ import annotations

from strongreject_claim.loader import (
    StrongREJECTRow,
    load_full_rows,
    load_small_rows,
    rows_for_category,
)


def test_load_full_returns_typed_rows() -> None:
    rows = load_full_rows()
    first: StrongREJECTRow = rows[0]
    assert set(first.keys()) == {"category", "source", "forbidden_prompt"}
    assert isinstance(first["category"], str)
    assert isinstance(first["source"], str)
    assert isinstance(first["forbidden_prompt"], str)


def test_load_small_returns_typed_rows() -> None:
    rows = load_small_rows()
    assert len(rows) == 60
    assert set(rows[0].keys()) == {"category", "source", "forbidden_prompt"}


def test_loader_decodes_utf8_for_non_ascii_row() -> None:
    rows = load_full_rows()
    non_ascii = [r for r in rows if not r["forbidden_prompt"].isascii()]
    assert len(non_ascii) == 1, "exactly one upstream row contains non-ASCII text"
    # Round-trip must be lossless: re-encoding produces no replacement chars
    assert "�" not in non_ascii[0]["forbidden_prompt"]


def test_loader_preserves_embedded_newlines() -> None:
    rows = load_full_rows()
    multi = [r for r in rows if "\n" in r["forbidden_prompt"]]
    assert len(multi) == 2, "exactly two upstream rows contain embedded newlines"


def test_loader_lowercases_categories_at_load_time() -> None:
    full_categories = {r["category"] for r in load_full_rows()}
    small_categories = {r["category"] for r in load_small_rows()}
    # Both files normalise to the same six lowercase keys, dissolving the
    # upstream "Disinformation and Deception" vs "Disinformation and
    # deception" inconsistency.
    expected = {
        "disinformation and deception",
        "hate, harassment and discrimination",
        "illegal goods and services",
        "non-violent crimes",
        "sexual content",
        "violence",
    }
    assert full_categories == expected
    assert small_categories == expected


def test_rows_for_violence_returns_54_rows() -> None:
    assert len(rows_for_category("violence")) == 54


def test_rows_for_unknown_category_returns_empty() -> None:
    assert rows_for_category("nonexistent_category") == []


def test_rows_for_category_is_case_insensitive() -> None:
    """The argument is lowercased; "Violence" and "violence" both work."""
    assert len(rows_for_category("Violence")) == 54
    assert len(rows_for_category("VIOLENCE")) == 54


def test_no_braces_in_forbidden_prompts() -> None:
    """str.format substitution in judge.py is safe iff prompts have no braces.

    Verified once at investigation time; this test enforces the
    invariant for any future dataset revision.
    """
    for r in load_full_rows():
        assert "{" not in r["forbidden_prompt"]
        assert "}" not in r["forbidden_prompt"]
    for r in load_small_rows():
        assert "{" not in r["forbidden_prompt"]
        assert "}" not in r["forbidden_prompt"]
