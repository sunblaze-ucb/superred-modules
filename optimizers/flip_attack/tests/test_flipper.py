"""Tests for FlipAttack text transformation functions."""

from flip_attack_optimizer.flipper import (
    flip_word_order,
    flip_chars_in_word,
    flip_chars_in_sentence,
    flip_fool_model,
    split_sentence_in_half,
    FLIP_MODES,
)


def test_flip_word_order():
    assert flip_word_order("how to make a bomb") == "bomb a make to how"


def test_flip_word_order_single_word():
    assert flip_word_order("hello") == "hello"


def test_flip_word_order_empty():
    assert flip_word_order("") == ""


def test_flip_chars_in_word():
    assert flip_chars_in_word("how to make") == "woh ot ekam"


def test_flip_chars_in_word_single_word():
    assert flip_chars_in_word("hello") == "olleh"


def test_flip_chars_in_sentence():
    assert flip_chars_in_sentence("hello world") == "dlrow olleh"


def test_flip_chars_in_sentence_single_char():
    assert flip_chars_in_sentence("a") == "a"


def test_flip_fool_model_same_as_fcs():
    text = "some test input"
    assert flip_fool_model(text) == flip_chars_in_sentence(text)


def test_split_sentence_in_half_uses_character_midpoint():
    """Split uses character midpoint via textwrap, not word count."""
    first, second = split_sentence_in_half("one two three four")
    # Character midpoint of "one two three four" (18 chars) is 9
    # textwrap.wrap with width=9 breaks at word boundaries
    # "one two" (7) fits, "three" starts at 8 -- "one two" then "three four"
    assert first == "one two"
    assert second == "three four"


def test_split_sentence_in_half_odd_words():
    first, second = split_sentence_in_half("one two three")
    # "one two three" is 13 chars, midpoint 6
    # textwrap.wrap(width=6): "one" fits, "two" at 4 fits (len "one two" = 7 > 6)
    # So: "one", "two three" OR "one two", "three" depending on wrapping
    assert first  # non-empty
    assert second  # non-empty
    assert f"{first} {second}" == "one two three"


def test_split_sentence_in_half_single_word():
    first, second = split_sentence_in_half("hello")
    assert first == "hello"
    assert second == ""


def test_split_sentence_in_half_recombines():
    """Both halves joined should reconstruct the original text."""
    text = "how to make a dangerous weapon at home"
    first, second = split_sentence_in_half(text)
    if second:
        assert f"{first} {second}" == text
    else:
        assert first == text


def test_flip_modes_registry():
    assert set(FLIP_MODES.keys()) == {"FWO", "FCW", "FCS", "FMM"}
    assert FLIP_MODES["FWO"] is flip_word_order
    assert FLIP_MODES["FCW"] is flip_chars_in_word
    assert FLIP_MODES["FCS"] is flip_chars_in_sentence
    assert FLIP_MODES["FMM"] is flip_fool_model
