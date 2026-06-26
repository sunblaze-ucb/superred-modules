"""Byte-identical asset tests for the MUZZLE optimizer port.

Checks four classes of claim:

1. Vendored YAML prompt files are byte-identical to the pinned upstream muzzle revision.
2. Vendored injection JSON files are byte-identical to the pinned upstream muzzle revision.
3. ``muzzle_optimizer.constants`` values match the values documented in ASSUMPTIONS.md.
4. Vendored PAIR primitives equal the reviewed ``pair_optimizer`` source, allowing only the
   module-name change on import lines that reference ``pair_optimizer`` /
   ``muzzle_optimizer.pair``.
"""

from __future__ import annotations

import pathlib

import pytest

import muzzle_optimizer.constants as constants

# ---------------------------------------------------------------------------
# Absolute paths
# ---------------------------------------------------------------------------

_MUZZLE_PKG = pathlib.Path(
    "/Users/simonsure/research/superred/.worktrees/muzzle-optimizer"
    "/optimizers/muzzle/src/muzzle_optimizer"
)
_UPSTREAM_ROOT = pathlib.Path("/Users/simonsure/.claude/jobs/2e94194c/tmp/muzzle")
_PAIR_REF_DIR = pathlib.Path(
    "/Users/simonsure/research/superred/superred-modules/optimizers/pair/src/pair_optimizer"
)

_VENDORED_YAML_DIR = _MUZZLE_PKG / "data" / "prompts"
_UPSTREAM_YAML_DIR = _UPSTREAM_ROOT / "muzzle" / "prototype" / "agents" / "prompts"

_VENDORED_JSON_DIR = _MUZZLE_PKG / "data" / "injections"
_UPSTREAM_JSON_DIR = _UPSTREAM_ROOT / "configs" / "injections"

_MUZZLE_PAIR_DIR = _MUZZLE_PKG / "pair"

# ---------------------------------------------------------------------------
# File name lists
# ---------------------------------------------------------------------------

_YAML_STEMS = ["summarizer", "grafter", "prompter", "judge", "dispatcher"]
_JSON_STEMS = [
    "generic_plain_text",
    "generic_url_injection",
    "goal_hijacking_plain_text",
    "goal_hijacking_url_injection",
]

# PAIR files that must be byte-identical (no import-name differences).
_PAIR_BYTE_IDENTICAL = ["prompts.py", "json_utils.py"]

# PAIR files that are identical except for module-prefix on import lines.
_PAIR_IMPORT_NORMALISED = ["attacker.py", "evaluator.py"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _normalise_imports(text: str) -> str:
    """Rewrite module-name differences on PAIR import lines before comparison.

    Any line that contains ``import`` and references either ``pair_optimizer``
    or ``muzzle_optimizer.pair`` has ``muzzle_optimizer.pair.`` replaced by
    ``pair_optimizer.`` so the two copies compare equal everywhere except the
    package they live in.
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if "import" in line and ("pair_optimizer" in line or "muzzle_optimizer.pair" in line):
            line = line.replace("muzzle_optimizer.pair.", "pair_optimizer.")
        out.append(line)
    return "".join(out)


# ---------------------------------------------------------------------------
# 1. YAML prompt files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", _YAML_STEMS)
def test_yaml_prompt_byte_identical_to_upstream(stem: str) -> None:
    """Vendored ``data/prompts/<stem>.yaml`` equals the upstream muzzle YAML."""
    vendored = _VENDORED_YAML_DIR / f"{stem}.yaml"
    upstream = _UPSTREAM_YAML_DIR / f"{stem}.yaml"
    assert vendored.exists(), f"Vendored YAML not found: {vendored}"
    assert upstream.exists(), f"Upstream YAML not found: {upstream}"
    assert vendored.read_bytes() == upstream.read_bytes(), (
        f"data/prompts/{stem}.yaml differs from upstream muzzle"
    )


# ---------------------------------------------------------------------------
# 2. Injection JSON files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", _JSON_STEMS)
def test_injection_json_byte_identical_to_upstream(stem: str) -> None:
    """Vendored ``data/injections/<stem>.json`` equals the upstream muzzle JSON."""
    vendored = _VENDORED_JSON_DIR / f"{stem}.json"
    upstream = _UPSTREAM_JSON_DIR / f"{stem}.json"
    assert vendored.exists(), f"Vendored injection JSON not found: {vendored}"
    assert upstream.exists(), f"Upstream injection JSON not found: {upstream}"
    assert vendored.read_bytes() == upstream.read_bytes(), (
        f"data/injections/{stem}.json differs from upstream muzzle"
    )


# ---------------------------------------------------------------------------
# 3. Constants
# ---------------------------------------------------------------------------


def test_constant_placeholder_instr() -> None:
    """PLACEHOLDER_INSTR is the PROBE marker planted into the victim context."""
    assert constants.PLACEHOLDER_INSTR == "[INSTRUCTION_PLACEHOLDER]"


def test_constant_placeholder_alias() -> None:
    """PLACEHOLDER_ALIAS is the legacy PAIR splice token."""
    assert constants.PLACEHOLDER_ALIAS == "[PLACEHOLDER]"


def test_constant_target_str() -> None:
    """TARGET_STR is the PAIR target-string default (calibrant convention)."""
    assert constants.TARGET_STR == "The next step is to "


def test_constant_upstream_muzzle_sha() -> None:
    """UPSTREAM_MUZZLE_SHA pins the audited upstream revision."""
    assert constants.UPSTREAM_MUZZLE_SHA == "ed611c0de448caf3dc50542b0a90023424774bf7"


# ---------------------------------------------------------------------------
# 4. Vendored PAIR primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fname", _PAIR_BYTE_IDENTICAL)
def test_pair_primitive_byte_identical_to_pair_optimizer(fname: str) -> None:
    """``muzzle_optimizer/pair/<fname>`` is byte-identical to ``pair_optimizer/<fname>``."""
    muzzle_file = _MUZZLE_PAIR_DIR / fname
    pair_ref = _PAIR_REF_DIR / fname
    if not pair_ref.exists():
        pytest.skip(f"pair_optimizer reference absent: {pair_ref}")
    assert muzzle_file.exists(), f"Muzzle PAIR file missing: {muzzle_file}"
    assert muzzle_file.read_bytes() == pair_ref.read_bytes(), (
        f"muzzle_optimizer/pair/{fname} differs from pair_optimizer/{fname}"
    )


@pytest.mark.parametrize("fname", _PAIR_IMPORT_NORMALISED)
def test_pair_primitive_identical_except_package_imports(fname: str) -> None:
    """``pair/{fname}`` equals ``pair_optimizer/{fname}`` after normalising import prefixes.

    Only lines that contain ``import`` and reference ``pair_optimizer`` or
    ``muzzle_optimizer.pair`` may differ; all other lines must be byte-equal.
    """
    muzzle_file = _MUZZLE_PAIR_DIR / fname
    pair_ref = _PAIR_REF_DIR / fname
    if not pair_ref.exists():
        pytest.skip(f"pair_optimizer reference absent: {pair_ref}")
    assert muzzle_file.exists(), f"Muzzle PAIR file missing: {muzzle_file}"
    muzzle_text = _normalise_imports(muzzle_file.read_text(encoding="utf-8"))
    pair_text = _normalise_imports(pair_ref.read_text(encoding="utf-8"))
    assert muzzle_text == pair_text, (
        f"muzzle_optimizer/pair/{fname} differs from pair_optimizer/{fname} "
        f"beyond import-name normalisation"
    )
