"""Bijection construction, encoding, and decoding.

Verbatim ports of ``BijectionLanguage`` and ``DigitBijectionLanguage``
from the upstream ``haizelabs/bijection-learning``
(``that_good_infra/transformations.py``). We keep ``letter`` and
``digit`` codomains — the two evaluated in the paper's main results
table (Huang et al., ICLR 2025, Table 1).

Each ``Bijection`` is a single random map from ``a..z`` to a codomain,
constructed under a fixed-points budget (``fixed_size``) that controls
the paper's complexity dimension ``dispersion = 26 - fixed_size``.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

ALPHABET: str = string.ascii_lowercase

# Default RNG used by ``generate_bijection`` when no ``rng`` is passed.
# A real ``random.Random`` instance (not the ``random`` module itself)
# so the parameter type stays ``random.Random | None`` cleanly under
# strict type-checking.
_DEFAULT_RNG: random.Random = random.Random()


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bijection:
    """One concrete bijection from ``a..z`` to a codomain of strings.

    Attributes:
        codomain: Either ``"letter"`` or ``"digit"``. Selects which
            decoder to use; encoders are unified by ``mapping`` /
            ``delimiter`` alone.
        mapping: ``dict[str, str]`` mapping each lowercase letter to
            its encoded form. Identity entries are kept explicitly so
            ``mapping`` can also serve as the "alphabet table" shown
            to the model in the system prompt.
        fixed_size: Number of letters that map to themselves. The
            paper's dispersion is ``26 - fixed_size``.
        delimiter: Inserted before each numeric token in the encoded
            output. Only used when ``codomain == "digit"``; ignored
            for ``"letter"``. Paper default for digit: two spaces.
        num_digits: Length of each numeric encoding (only meaningful
            when ``codomain == "digit"``). Paper default: 2.
    """

    codomain: str
    mapping: dict[str, str]
    fixed_size: int
    delimiter: str = ""
    num_digits: int = 2
    _inverse: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # `frozen=True` blocks attribute assignment, so build the
        # inverse via object.__setattr__ (same trick stdlib uses).
        object.__setattr__(
            self,
            "_inverse",
            {v: k for k, v in self.mapping.items()},
        )

    # -------------------------------------------------------------------
    # Encode / decode
    # -------------------------------------------------------------------

    def encode(self, text: str) -> str:
        """Apply the bijection to ``text``.

        Lowercases first (the upstream encoder also calls ``.lower()``).
        Characters outside ``a..z`` pass through unchanged. For digit
        bijections, the delimiter is inserted before each *substituted*
        numeric token (not before identity-mapped letters), matching
        upstream ``DigitBijectionLanguage._f``.
        """
        if self.codomain == "letter":
            return self._encode_letter(text)
        return self._encode_digit(text)

    def _encode_letter(self, text: str) -> str:
        return "".join(self.mapping.get(c, c) for c in text.lower())

    def _encode_digit(self, text: str) -> str:
        out_parts: list[str] = []
        for c in text.lower():
            replacement = self.mapping.get(c, c)
            if replacement.isnumeric() and replacement != c:
                out_parts.append(self.delimiter + replacement)
            else:
                out_parts.append(replacement)
        return "".join(out_parts)

    def decode(self, text: str) -> str:
        """Invert the bijection on ``text``.

        For ``letter`` codomain: lowercases first then inverse-maps
        char-by-char (upstream's ``permute_string`` is the same
        function for encode and decode and lowercases unconditionally;
        we match that exactly so a downstream judge wrapper feeding
        raw model output through gets the upstream-equivalent text).
        For ``digit`` codomain: verbatim port of the upstream
        ``permute_digits_to_string`` walker — peel off ``delimiter``
        then ``num_digits`` digits at each potential token boundary.
        """
        if self.codomain == "letter":
            return "".join(self._inverse.get(c, c) for c in text.lower())

        # digit codomain
        delim = self.delimiter
        n = self.num_digits
        # Upstream prepends delimiter + strip(); we mirror that exactly
        # so the first numeric token at offset 0 is recognised.
        s = delim + text.strip()
        out: list[str] = []
        i = 0
        L = len(s)
        while i < L:
            if (
                i + n + len(delim) <= L
                and s[i : i + len(delim)] == delim
                and all(s[j].isdigit() for j in range(i + len(delim), i + len(delim) + n))
            ):
                num = s[i + len(delim) : i + len(delim) + n]
                out.append(self._inverse.get(num, num))
                i += len(delim) + n
            else:
                out.append(s[i])
                i += 1
        return "".join(out)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _draw_fixed_points(rng: random.Random, fixed_size: int) -> list[str]:
    if fixed_size < 0 or fixed_size > 26:
        raise ValueError("fixed_size must be in [0, 26]")
    return rng.sample(ALPHABET, k=fixed_size)


def _make_letter_bijection(rng: random.Random, fixed_size: int) -> dict[str, str]:
    """Random alphabet permutation with ``fixed_size`` fixed points.

    Verbatim port of upstream ``BijectionLanguage.create_bijective_mapping``.
    """
    fixed = _draw_fixed_points(rng, fixed_size)
    movable = [c for c in ALPHABET if c not in fixed]
    shuffled = movable[:]
    rng.shuffle(shuffled)
    permuted: dict[str, str] = {c: c for c in fixed}
    for src, dst in zip(movable, shuffled):
        permuted[src] = dst
    return permuted


def _make_digit_bijection(
    rng: random.Random, fixed_size: int, num_digits: int
) -> dict[str, str]:
    """Map each non-fixed letter to a unique ``num_digits``-digit number.

    Verbatim port of upstream
    ``DigitBijectionLanguage.create_bijective_mapping_digits``: numbers
    are sampled without replacement from ``[10^(n-1), 10^n)`` so every
    encoding has exactly ``num_digits`` digits.
    """
    if num_digits < 1:
        raise ValueError("num_digits must be >= 1")

    fixed = _draw_fixed_points(rng, fixed_size)
    movable = [c for c in ALPHABET if c not in fixed]

    low = int("1" + "0" * (num_digits - 1)) if num_digits > 1 else 1
    high = int("1" + "0" * num_digits)
    pool = list(range(low, high))
    if len(pool) < len(movable):
        raise ValueError(
            f"num_digits={num_digits} cannot encode {len(movable)} letters "
            f"(only {len(pool)} unique tokens available)"
        )
    sampled = rng.sample(pool, k=len(movable))

    permuted: dict[str, str] = {c: c for c in fixed}
    for letter, num in zip(movable, sampled):
        permuted[letter] = str(num)
    return permuted


def generate_bijection(
    *,
    codomain: str,
    fixed_size: int,
    num_digits: int = 2,
    delimiter: str = "",
    rng: random.Random | None = None,
) -> Bijection:
    """Generate one random bijection.

    Args:
        codomain: ``"letter"`` (alphabet permutation) or ``"digit"``
            (each letter → unique ``num_digits``-digit number).
        fixed_size: Number of letters fixed under the bijection;
            paper's dispersion is ``26 - fixed_size``.
        num_digits: Encoding length for ``digit`` codomain. Ignored
            when ``codomain == "letter"``.
        delimiter: Prefix inserted before each substituted numeric
            token. Ignored when ``codomain == "letter"``.
        rng: Optional ``random.Random`` for deterministic generation.
            Defaults to the module-level random instance, matching
            upstream behaviour.
    """
    rng = rng if rng is not None else _DEFAULT_RNG
    if codomain == "letter":
        mapping = _make_letter_bijection(rng, fixed_size)
        return Bijection(
            codomain="letter",
            mapping=mapping,
            fixed_size=fixed_size,
        )
    if codomain == "digit":
        mapping = _make_digit_bijection(rng, fixed_size, num_digits)
        return Bijection(
            codomain="digit",
            mapping=mapping,
            fixed_size=fixed_size,
            delimiter=delimiter,
            num_digits=num_digits,
        )
    raise ValueError(
        f"unknown codomain {codomain!r}; expected 'letter' or 'digit'"
    )


__all__ = ["ALPHABET", "Bijection", "generate_bijection"]
