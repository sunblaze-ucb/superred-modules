"""Tests for bijection construction, encoding and decoding."""

from __future__ import annotations

import random
import string

import pytest

from bijection_optimizer.bijection import (
    ALPHABET,
    Bijection,
    _DEFAULT_RNG,
    generate_bijection,
)


def _ascii_lc_alphabet() -> set[str]:
    return set(string.ascii_lowercase)


# ---------------------------------------------------------------------------
# Letter codomain
# ---------------------------------------------------------------------------


class TestLetterBijection:
    def test_letter_mapping_has_all_26_letters_and_no_duplicates(self) -> None:
        b = generate_bijection(
            codomain="letter", fixed_size=0, rng=random.Random(0),
        )
        assert set(b.mapping.keys()) == _ascii_lc_alphabet()
        assert len(set(b.mapping.values())) == 26

    @pytest.mark.parametrize("fixed_size", [0, 1, 13, 25, 26])
    def test_letter_fixed_size_count_is_at_least_requested(
        self, fixed_size: int
    ) -> None:
        # The construction guarantees ``fixed_size`` letters that map to
        # themselves; movable letters may also coincidentally map to
        # themselves under shuffle, so the actual count is >= fixed_size.
        b = generate_bijection(
            codomain="letter",
            fixed_size=fixed_size,
            rng=random.Random(123),
        )
        fixed_count = sum(1 for k, v in b.mapping.items() if k == v)
        assert fixed_count >= fixed_size

    def test_letter_encoding_roundtrip(self) -> None:
        b = generate_bijection(
            codomain="letter", fixed_size=5, rng=random.Random(42),
        )
        text = "the quick brown fox jumps over the lazy dog"
        encoded = b.encode(text)
        assert b.decode(encoded) == text

    def test_letter_encoding_preserves_non_alpha(self) -> None:
        b = generate_bijection(
            codomain="letter", fixed_size=0, rng=random.Random(1),
        )
        encoded = b.encode("hello, world! 123")
        # Non-alpha (comma, space, !, digits) are preserved verbatim.
        assert "," in encoded and "!" in encoded and "1" in encoded
        assert b.decode(encoded) == "hello, world! 123"

    def test_letter_encoding_lowercases_input(self) -> None:
        b = generate_bijection(
            codomain="letter", fixed_size=0, rng=random.Random(7),
        )
        # Upstream encoder lowercases first; decoded output is lowercase.
        decoded = b.decode(b.encode("HELLO"))
        assert decoded == "hello"


# ---------------------------------------------------------------------------
# Digit codomain
# ---------------------------------------------------------------------------


class TestDigitBijection:
    def test_digit_mapping_uses_unique_n_digit_numbers(self) -> None:
        b = generate_bijection(
            codomain="digit",
            fixed_size=0,
            num_digits=2,
            delimiter="  ",
            rng=random.Random(0),
        )
        # Every non-fixed letter -> a unique 2-digit string in [10,100).
        numerics = [v for v in b.mapping.values() if v.isdigit()]
        assert len(numerics) == 26  # fixed_size=0 -> all letters movable
        assert len(set(numerics)) == 26
        for n in numerics:
            assert len(n) == 2
            assert 10 <= int(n) < 100

    @pytest.mark.parametrize("num_digits", [2, 3, 4])
    def test_digit_roundtrip_with_empty_delimiter(self, num_digits: int) -> None:
        # Upstream's decoder pre-pends the delimiter and leaks it at the
        # start when delim is non-empty (see ASSUMPTIONS.md). With an
        # empty delimiter the roundtrip is exact.
        b = generate_bijection(
            codomain="digit",
            fixed_size=8,
            num_digits=num_digits,
            delimiter="",
            rng=random.Random(9),
        )
        text = "the harbor was busy in early autumn"
        assert b.decode(b.encode(text)) == text

    def test_digit_roundtrip_with_double_space_delim_leaks_leading_delim(
        self,
    ) -> None:
        # Verbatim port of upstream behavior: with non-empty delim the
        # decoder leaks the prepended delim at the start. The rest of
        # the text must still round-trip cleanly.
        b = generate_bijection(
            codomain="digit",
            fixed_size=8,
            num_digits=2,
            delimiter="  ",
            rng=random.Random(13),
        )
        text = "the harbor was busy in early autumn"
        decoded = b.decode(b.encode(text))
        assert decoded.lstrip(" ") == text
        # Confirm the original text is still recoverable by the same
        # whitespace-strip step upstream's pipeline applies.
        assert decoded.strip() == text

    def test_digit_delimiter_only_inserted_before_substituted_tokens(
        self,
    ) -> None:
        # Encoder sanity check: delimiter precedes substituted tokens
        # but never identity-mapped letters.
        rng = random.Random(11)
        b = generate_bijection(
            codomain="digit",
            fixed_size=25,  # only 1 movable letter
            num_digits=2,
            delimiter="X",
            rng=rng,
        )
        movable = next(k for k, v in b.mapping.items() if k != v)
        fixed_letter = "a" if movable != "a" else "b"
        encoded = b.encode(fixed_letter + movable)
        assert encoded == fixed_letter + "X" + b.mapping[movable]
        # Encoding two fixed letters in a row never inserts a delimiter.
        encoded_pair_fixed = b.encode(fixed_letter + fixed_letter)
        assert "X" not in encoded_pair_fixed

    def test_digit_rejects_too_few_digits_for_alphabet(self) -> None:
        # num_digits=1 would only give us tokens 1..9 (9 tokens), but
        # a 0-fixed bijection needs 26 unique tokens.
        with pytest.raises(ValueError):
            generate_bijection(
                codomain="digit",
                fixed_size=0,
                num_digits=1,
                rng=random.Random(0),
            )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_unknown_codomain_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_bijection(codomain="word", fixed_size=10)

    @pytest.mark.parametrize("fixed_size", [-1, 27])
    def test_out_of_range_fixed_size_raises(self, fixed_size: int) -> None:
        with pytest.raises(ValueError):
            generate_bijection(codomain="letter", fixed_size=fixed_size)

    def test_zero_num_digits_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_bijection(
                codomain="digit", fixed_size=0, num_digits=0,
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_gives_same_mapping(self) -> None:
        rng_a = random.Random(2024)
        rng_b = random.Random(2024)
        a = generate_bijection(codomain="letter", fixed_size=10, rng=rng_a)
        b = generate_bijection(codomain="letter", fixed_size=10, rng=rng_b)
        assert a.mapping == b.mapping

    def test_consecutive_calls_differ(self) -> None:
        rng = random.Random(0)
        a = generate_bijection(codomain="letter", fixed_size=10, rng=rng)
        b = generate_bijection(codomain="letter", fixed_size=10, rng=rng)
        # Same RNG, two draws -> different bijections (with high probability).
        assert a.mapping != b.mapping


# ---------------------------------------------------------------------------
# Bijection dataclass
# ---------------------------------------------------------------------------


class TestBijectionType:
    def test_inverse_is_built_from_mapping(self) -> None:
        b = Bijection(
            codomain="letter",
            mapping={c: c for c in ALPHABET},
            fixed_size=26,
        )
        # Identity bijection: encode/decode are pass-through.
        assert b.encode("hello") == "hello"
        assert b.decode("hello") == "hello"

    def test_explicit_mapping_roundtrip(self) -> None:
        # Hand-built tiny letter bijection (a<->b, rest identity).
        mapping = {c: c for c in ALPHABET}
        mapping["a"], mapping["b"] = "b", "a"
        b = Bijection(codomain="letter", mapping=mapping, fixed_size=24)
        assert b.encode("aabb") == "bbaa"
        assert b.decode("bbaa") == "aabb"

    def test_letter_decode_lowercases_input_matching_upstream(self) -> None:
        # Upstream's ``permute_string`` is reused for both encode and
        # decode and lowercases unconditionally. We mirror that so a
        # downstream judge wrapper feeding raw model output through the
        # decoder gets the upstream-equivalent text (identity bijection
        # on mixed-case input collapses to lowercase, not preserves
        # case).
        b = Bijection(
            codomain="letter",
            mapping={c: c for c in ALPHABET},
            fixed_size=26,
        )
        assert b.decode("MIXED") == "mixed"


# ---------------------------------------------------------------------------
# Default RNG
# ---------------------------------------------------------------------------


class TestDefaultRng:
    def test_module_default_rng_is_random_random_instance(self) -> None:
        # The module-level default is a real ``random.Random`` instance
        # (not the ``random`` module itself), so the parameter type
        # ``random.Random | None`` stays well-formed under strict
        # type-checking.
        assert isinstance(_DEFAULT_RNG, random.Random)

    def test_default_rng_path_produces_a_valid_bijection(self) -> None:
        # Without an ``rng`` argument the function must use
        # ``_DEFAULT_RNG`` and still return a well-formed bijection.
        b = generate_bijection(codomain="letter", fixed_size=5)
        assert isinstance(b, Bijection)
        assert b.codomain == "letter"
        assert set(b.mapping.keys()) == _ascii_lc_alphabet()
        # Encode/decode roundtrip works on lowercase input.
        assert b.decode(b.encode("hello world")) == "hello world"
