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


def test_split_sentence_in_half():
    first, second = split_sentence_in_half("one two three four")
    assert first == "one two"
    assert second == "three four"


def test_split_sentence_in_half_odd_words():
    first, second = split_sentence_in_half("one two three")
    assert first == "one"
    assert second == "two three"


def test_flip_modes_registry():
    assert set(FLIP_MODES.keys()) == {"FWO", "FCW", "FCS", "FMM"}
    assert FLIP_MODES["FWO"] is flip_word_order
    assert FLIP_MODES["FCW"] is flip_chars_in_word
    assert FLIP_MODES["FCS"] is flip_chars_in_sentence
    assert FLIP_MODES["FMM"] is flip_fool_model
