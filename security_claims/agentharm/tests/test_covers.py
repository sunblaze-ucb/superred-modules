"""Benign cover loader + data-completeness (offline, no LLM)."""

from __future__ import annotations

from security_claim_agentharm.covers import (
    EXCLUDED_BASES,
    base_id,
    cover_for,
    covered_base_ids,
    is_excluded,
)

# The 41 base behaviours reframable as IPI = 44 in harmful test_public minus the three
# structurally-unwinnable bases 21, 22, 49 (see EXCLUDED_BASES for why each).
_EXPECTED_BASES = frozenset(
    {
        "1",
        "2",
        "7",
        "9",
        "10",
        "11",
        "13",
        "14",
        "17",
        "19",
        "20",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "33",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "42",
        "43",
        "44",
        "45",
        "46",
        "50",
        "52",
        "53",
        "54",
        "58",
        "61",
        "62",
        "63",
        "64",
        "66",
    }
)


def test_base_id_strips_augmentation() -> None:
    assert base_id("7-3") == "7"
    assert base_id("66-1") == "66"
    assert base_id("1-4") == "1"


def test_every_base_behaviour_has_a_cover() -> None:
    assert covered_base_ids() == _EXPECTED_BASES


def test_covers_are_nonempty_strings() -> None:
    for base in _EXPECTED_BASES:
        cover = cover_for(f"{base}-1")
        assert isinstance(cover, str) and cover.strip(), f"empty cover for base {base}"


def test_all_four_augmentations_share_the_base_cover() -> None:
    # The cover is keyed by base, so every augmentation of a base resolves to it.
    for base in _EXPECTED_BASES:
        covers = {cover_for(f"{base}-{aug}") for aug in (1, 2, 3, 4)}
        assert len(covers) == 1, f"augmentations of base {base} disagree on cover"


def test_unknown_base_has_no_cover() -> None:
    assert cover_for("9999-1") is None


def test_structurally_unwinnable_bases_excluded() -> None:
    # All three are unwinnable (no injection reaches score==1.0), for distinct reasons:
    # 22 no injectable surface; 49 the cover's benign order poisons an all()-address check;
    # 21 an upstream grader bug makes correct_illegal_offers always False (both modes).
    assert EXCLUDED_BASES == frozenset({"21", "22", "49"})
    for base in ("21", "22", "49"):
        assert base not in covered_base_ids()
        assert cover_for(f"{base}-1") is None
        assert is_excluded(f"{base}-3") is True
    assert is_excluded("7-1") is False
