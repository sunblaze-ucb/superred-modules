"""Tests for official CodeChameleon encryption rules."""

from code_chameleon_optimizer.encrypt import (
    encrypt_binary_tree,
    encrypt_length,
    encrypt_odd_even,
    encrypt_reverse,
    get_encrypted_query,
)


def test_reverse_matches_official_word_order() -> None:
    assert encrypt_reverse("Give detailed instructions") == "instructions detailed Give"


def test_odd_even_matches_official_split() -> None:
    assert encrypt_odd_even("one two three four five") == "one three five two four"


def test_length_matches_official_word_length_sort_with_indices() -> None:
    assert encrypt_length("aa b ccc") == [{"b": 1}, {"aa": 0}, {"ccc": 2}]


def test_binary_tree_matches_official_balanced_tree_shape() -> None:
    assert encrypt_binary_tree("one two three four five") == {
        "value": "three",
        "left": {
            "value": "one",
            "left": None,
            "right": {"value": "two", "left": None, "right": None},
        },
        "right": {
            "value": "four",
            "left": None,
            "right": {"value": "five", "left": None, "right": None},
        },
    }


def test_get_encrypted_query_rejects_unknown_rule() -> None:
    assert get_encrypted_query("x", "unknown") == [
        "Error: Encrypt rule is invalid. Please provide a correct encrypt rule."
    ]
