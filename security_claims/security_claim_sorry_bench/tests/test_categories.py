"""Tests for the 44-class taxonomy module."""

from __future__ import annotations

import pytest

from sorry_bench_claim.categories import (
    CATEGORY_DOMAINS,
    CATEGORY_NAMES,
    CATEGORY_NAMES_SHORTEST,
    category_domain,
    category_name,
    category_name_shortest,
)


class TestCategoryListShape:
    def test_long_form_has_44_entries(self) -> None:
        assert len(CATEGORY_NAMES) == 44

    def test_shortest_form_has_44_entries(self) -> None:
        assert len(CATEGORY_NAMES_SHORTEST) == 44

    def test_domains_has_44_entries(self) -> None:
        assert len(CATEGORY_DOMAINS) == 44

    def test_no_duplicate_long_names(self) -> None:
        assert len(set(CATEGORY_NAMES)) == 44

    def test_no_duplicate_shortest_names(self) -> None:
        assert len(set(CATEGORY_NAMES_SHORTEST)) == 44

    def test_all_long_names_nonempty(self) -> None:
        assert all(name for name in CATEGORY_NAMES)


class TestDomainGrouping:
    """The 4 high-level domains group categories per paper §2.2 / Table 5."""

    def test_hate_speech_covers_categories_1_to_5(self) -> None:
        for cat_id in range(1, 6):
            assert CATEGORY_DOMAINS[cat_id - 1] == "Hate Speech Generation"

    def test_crimes_or_torts_covers_categories_6_to_24(self) -> None:
        for cat_id in range(6, 25):
            assert CATEGORY_DOMAINS[cat_id - 1] == "Assistance with Crimes or Torts"

    def test_inappropriate_topics_covers_categories_25_to_39(self) -> None:
        for cat_id in range(25, 40):
            assert CATEGORY_DOMAINS[cat_id - 1] == "Potentially Inappropriate Topics"

    def test_unqualified_advice_covers_categories_40_to_44(self) -> None:
        for cat_id in range(40, 45):
            assert CATEGORY_DOMAINS[cat_id - 1] == "Potentially Unqualified Advice"


class TestCanonicalDifferenceFromPaperTable5:
    """Spot-check the 8 entries that differ from paper Table 5 (page 19).

    The dataset's ``meta_info.py`` is canonical. These tests guard
    against accidentally reverting to the paper's strings.
    """

    @pytest.mark.parametrize(
        ("cat_id", "expected"),
        [
            (3, "Threatening Words"),  # paper: "...and Blackmails"
            (5, "Libelous Words"),  # paper: "...and Defamation"
            (19, "Terrorism"),  # paper: "Terrorism Promotion and Advice"
            (21, "Animal-related Crimes"),  # paper: "Animal-Related" (capital R)
            (34, "Religion Promotion"),  # paper: "...(Missionary)"
        ],
    )
    def test_truncation_diffs_from_paper(self, cat_id: int, expected: str) -> None:
        assert CATEGORY_NAMES[cat_id - 1] == expected

    @pytest.mark.parametrize("cat_id", [12, 24, 39])
    def test_uses_ampersand_not_and(self, cat_id: int) -> None:
        # Categories 12, 24, 39 use "&" in canonical, "and" in paper.
        assert " & " in CATEGORY_NAMES[cat_id - 1]


class TestCategoryNameLookup:
    def test_returns_correct_name_for_id_1(self) -> None:
        assert category_name(1) == "Personal Insulting Words"

    def test_returns_correct_name_for_id_44(self) -> None:
        assert category_name(44) == "Dangerous Machinery Operation Advice"

    def test_returns_correct_name_for_id_8(self) -> None:
        # #8 Harassment is one of the "most refused" categories per paper §K.2.
        assert category_name(8) == "Harassment"

    @pytest.mark.parametrize("bad_id", [0, -1, 45, 100])
    def test_raises_on_out_of_range_id(self, bad_id: int) -> None:
        with pytest.raises(ValueError, match="must be in 1..44"):
            category_name(bad_id)


class TestShortestNameLookup:
    def test_id_1_is_personal_insult_words(self) -> None:
        # Note: shortest form uses "Insult" not "Insulting" (per meta_info.py).
        assert category_name_shortest(1) == "Personal Insult Words"

    def test_id_34_is_just_religion(self) -> None:
        # Figure 4 axis label form.
        assert category_name_shortest(34) == "Religion"

    @pytest.mark.parametrize("bad_id", [0, 45])
    def test_raises_on_out_of_range_id(self, bad_id: int) -> None:
        with pytest.raises(ValueError, match="must be in 1..44"):
            category_name_shortest(bad_id)


class TestDomainLookup:
    @pytest.mark.parametrize(
        ("cat_id", "expected"),
        [
            (1, "Hate Speech Generation"),
            (5, "Hate Speech Generation"),
            (6, "Assistance with Crimes or Torts"),
            (24, "Assistance with Crimes or Torts"),
            (25, "Potentially Inappropriate Topics"),
            (39, "Potentially Inappropriate Topics"),
            (40, "Potentially Unqualified Advice"),
            (44, "Potentially Unqualified Advice"),
        ],
    )
    def test_id_to_domain_mapping(self, cat_id: int, expected: str) -> None:
        assert category_domain(cat_id) == expected

    @pytest.mark.parametrize("bad_id", [0, 45])
    def test_raises_on_out_of_range_id(self, bad_id: int) -> None:
        with pytest.raises(ValueError, match="must be in 1..44"):
            category_domain(bad_id)
