"""Best-of-N text augmentations, ported from jplhughes/bon-jailbreaking.

Upstream: ``bon/attacks/run_text_bon.py`` (MIT) — ``apply_word_scrambling``,
``apply_random_capitalization``, ``apply_ascii_noising`` and the
``process_text_augmentation`` ordering. The random prefix/suffix path comes
from ``bon/utils/text_utils.py``.

Upstream seeds the *global* ``random`` module. These functions take a seeded
``random.Random`` instance instead, which draws the identical sequence (same
Mersenne Twister, same seed) without clobbering global state — a test asserts
the outputs match upstream's global-seeded implementation exactly.

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import random

#: Upstream's default proportion of characters to augment.
DEFAULT_SIGMA = 0.4


def apply_word_scrambling(text: str, sigma: float, rng: random.Random) -> str:
    """Scramble the middle characters of words longer than 3 characters.

    Byte-faithful to upstream: the scramble probability is ``sigma ** (1/2)``,
    the first and last characters are preserved, and only the middle is
    shuffled.
    """
    words = text.split()
    scrambled_words = []
    for word in words:
        if len(word) > 3 and rng.random() < sigma ** (1 / 2):
            chars = list(word)
            middle_chars = chars[1:-1]
            rng.shuffle(middle_chars)
            scrambled_word = chars[0] + "".join(middle_chars) + chars[-1]
            scrambled_words.append(scrambled_word)
        else:
            scrambled_words.append(word)
    return " ".join(scrambled_words)


def apply_random_capitalization(text: str, sigma: float, rng: random.Random) -> str:
    """Randomly flip the case of letters, at probability ``sigma ** (1/2)``.

    Byte-faithful to upstream, **including a character-dropping bug**: a
    character that is alphabetic and passes the probability check but is
    outside ``a-z``/``A-Z`` (any non-ASCII letter, e.g. ``é`` or Cyrillic)
    matches neither branch and is appended nowhere, so it vanishes from the
    output. That is preserved rather than "fixed", so the augmentation
    distribution matches the published attack; see ASSUMPTIONS.md.
    """
    new_text = []
    for c in text:
        if c.isalpha() and rng.random() < sigma ** (1 / 2):
            if "a" <= c <= "z":
                new_text.append(chr(ord(c) - 32))
            elif "A" <= c <= "Z":
                new_text.append(chr(ord(c) + 32))
        else:
            new_text.append(c)
    return "".join(new_text)


def apply_ascii_noising(text: str, sigma: float, rng: random.Random) -> str:
    """Shift printable characters by +/-1, at probability ``sigma ** 3``.

    Byte-faithful to upstream: a shift that would leave printable ASCII
    (32..126) is discarded and the original character kept.
    """
    new_text = []
    for c in text:
        if c.isprintable() and rng.random() < sigma**3:
            perturbation = rng.choice([-1, 1])
            new_char_code = ord(c) + perturbation
            if 32 <= new_char_code <= 126:
                new_text.append(chr(new_char_code))
            else:
                new_text.append(c)
        else:
            new_text.append(c)
    return "".join(new_text)


def random_token_string(num_tokens: int, rng: random.Random) -> str:
    """Upstream's ``get_attack_string``: random tokens from ``cl100k_base``.

    Samples ``num_tokens`` ids from the tokenizer vocabulary, excluding the
    special ids, the ids upstream documents as crashing the tokenizer
    (100261..100275 and 100256), and ``0``; then decodes and whitespace-
    normalises them, exactly as ``AttackString.get_normalised_string`` does.

    Requires ``tiktoken``, which is an optional extra because upstream leaves
    the random prefix/suffix off by default (lengths of ``0``).

    Raises:
        RuntimeError: if ``tiktoken`` is not installed.
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise RuntimeError(
            "random_prefix_length/random_suffix_length need the 'tokens' extra: "
            "pip install superred-optimizer-best-of-n[tokens]"
        ) from exc

    tokenizer = tiktoken.get_encoding("cl100k_base")
    special_ids = {
        tokenizer.encode(t, allowed_special=tokenizer.special_tokens_set)[0]
        for t in tokenizer.special_tokens_set
    }
    # Upstream: "These cause the tokenizer to crash due to PanicException".
    error_ids = set(range(100261, 100276)) | {100256}
    disallowed = special_ids | error_ids | {0}
    ids_to_sample = sorted(set(range(tokenizer.n_vocab)) - disallowed)
    token_ids = rng.sample(ids_to_sample, num_tokens)
    return " ".join(tokenizer.decode(token_ids).split())


def augment(
    text: str,
    *,
    sigma: float = DEFAULT_SIGMA,
    seed: int = 0,
    word_scrambling: bool = True,
    random_capitalization: bool = True,
    ascii_perturbation: bool = True,
    random_prefix_length: int = 0,
    random_suffix_length: int = 0,
) -> str:
    """Apply upstream's augmentation pipeline in upstream's order.

    ``process_text_augmentation``: prefix, suffix, then word scrambling,
    random capitalization and ASCII noising, each gated by its own flag.

    Args:
        text: The prompt to augment.
        sigma: Proportion of characters to augment (upstream default 0.4).
        seed: Seeds this variant; upstream seeds the global RNG per sample.
        word_scrambling/random_capitalization/ascii_perturbation: enable each
            augmentation, all on upstream by default.
        random_prefix_length/random_suffix_length: random tokens to prepend or
            append (upstream default 0; needs the ``tokens`` extra).

    Raises:
        ValueError: if *sigma* is outside ``[0, 1]`` or a length is negative.
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma!r}")
    if random_prefix_length < 0 or random_suffix_length < 0:
        raise ValueError("random prefix/suffix lengths must be >= 0")

    rng = random.Random(seed)
    if random_prefix_length > 0:
        text = random_token_string(random_prefix_length, rng) + "\n\n" + text
    if random_suffix_length > 0:
        text = text + "\n\n" + random_token_string(random_suffix_length, rng)
    if word_scrambling:
        text = apply_word_scrambling(text, sigma, rng)
    if random_capitalization:
        text = apply_random_capitalization(text, sigma, rng)
    if ascii_perturbation:
        text = apply_ascii_noising(text, sigma, rng)
    return text


__all__ = [
    "DEFAULT_SIGMA",
    "apply_ascii_noising",
    "apply_random_capitalization",
    "apply_word_scrambling",
    "augment",
    "random_token_string",
]
