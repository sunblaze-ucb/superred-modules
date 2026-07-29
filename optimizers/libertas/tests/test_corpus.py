"""Parity and Unicode-preservation tests for the pinned L1B3RT4S corpus."""

from __future__ import annotations

import hashlib
import os
import subprocess
import unicodedata
from pathlib import Path

import pytest

from libertas_optimizer.corpus import (
    UPSTREAM_COMMIT,
    detect_provider,
    load_manifest,
    load_prompt_templates,
    load_source_bytes,
    render_prompt,
    verify_bundled_corpus,
)


def test_bundled_snapshot_matches_manifest() -> None:
    assert verify_bundled_corpus() == []


def test_manifest_records_every_upstream_root_file() -> None:
    manifest = load_manifest()
    assert manifest["commit"] == UPSTREAM_COMMIT
    assert len(manifest["files"]) == 44
    assert sum(entry["bundled"] for entry in manifest["files"]) == 43

    omitted = {
        entry["source_path"]
        for entry in manifest["files"]
        if not entry["bundled"]
    }
    assert omitted == {"LICENSE"}
    assert all(
        entry["sha256"] and entry["size"] > 0 for entry in manifest["files"]
    )


def test_special_character_filenames_are_portable_but_bytes_are_exact() -> None:
    manifest = load_manifest()
    stored = {
        entry["source_path"]: entry["stored_path"] for entry in manifest["files"]
    }
    assert stored["*SPECIAL_TOKENS.json"] == "SPECIAL_TOKENS.json"
    assert stored["#MOTHERLOAD.txt"] == "MOTHERLOAD.txt"
    assert stored["!SHORTCUTS.json"] == "SHORTCUTS.json"

    for source_file in (
        "*SPECIAL_TOKENS.json",
        "#MOTHERLOAD.txt",
        "!SHORTCUTS.json",
    ):
        entry = next(
            item for item in manifest["files"] if item["source_path"] == source_file
        )
        content = load_source_bytes(source_file)
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_every_bundled_source_round_trips_strict_utf8() -> None:
    for entry in load_manifest()["files"]:
        if not entry["bundled"]:
            continue
        raw = load_source_bytes(entry["source_path"])
        assert raw.decode("utf-8", errors="strict").encode("utf-8") == raw


def test_hidden_and_combining_unicode_is_not_normalized() -> None:
    raw = load_source_bytes("README.md")
    text = raw.decode("utf-8", errors="strict")

    # Upstream's README includes variation selectors and combining marks.  A
    # normalizing read would change this exact string and invalidate the attack.
    assert "\ufe0e" in text or "\ufe0f" in text
    assert unicodedata.normalize("NFC", text) != text
    assert text.encode("utf-8") == raw


def test_prompt_body_hashes_match_exact_utf8_slices() -> None:
    templates = load_prompt_templates(
        provider=None,
        include_system_templates=True,
    )
    assert len(templates) == 45
    assert len({template.id for template in templates}) == len(templates)

    for template in templates:
        body_bytes = template.raw_template.encode("utf-8")
        assert hashlib.sha256(body_bytes).hexdigest() == template.raw_sha256
        assert template.raw_sha256[:16] in template.id


def test_strict_render_changes_only_exact_upstream_goal_markers() -> None:
    goal = "GØAL\u200d🧪\ufe0f"
    templates = load_prompt_templates(
        provider=None,
        include_system_templates=True,
    )
    assert templates

    for template in templates:
        expected = template.raw_template
        for marker in template.goal_markers:
            expected = expected.replace(marker, goal)
        rendered = render_prompt(template, goal)
        assert rendered == expected
        assert goal in rendered


def test_untemplated_sections_are_opt_in_and_append_without_normalizing() -> None:
    strict = load_prompt_templates(provider="anthropic")
    adapted = load_prompt_templates(provider="anthropic", include_untemplated=True)
    assert len(adapted) > len(strict)

    template = next(item for item in adapted if not item.is_templated)
    with pytest.raises(ValueError, match="no explicit upstream goal marker"):
        render_prompt(template, "goal")

    rendered = render_prompt(template, "goal", append_untemplated=True)
    assert rendered.startswith(template.raw_template)
    assert rendered.endswith("goal")


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("openai/gpt-5.4-mini", "openai"),
        ("azure/my-gpt-4o", "openai"),
        ("anthropic/claude-opus-4-6", "anthropic"),
        ("vertex_ai/gemini-3-pro", "google"),
        ("xai/grok-4", "xai"),
        ("meta/llama-4-maverick", "meta"),
        ("deepseek/deepseek-v3", "deepseek"),
        ("dashscope/qwen3", "alibaba"),
        ("bedrock/amazon.nova-pro", "amazon"),
        ("zai/glm-4.7", "zai"),
        ("moonshot/kimi-k2", "moonshot"),
    ],
)
def test_provider_detection(model: str, provider: str) -> None:
    assert detect_provider(model) == provider


def test_unknown_provider_detection_is_conservative() -> None:
    assert detect_provider("private/my-model") is None
    assert detect_provider("") is None


def test_optional_live_checkout_matches_every_manifest_entry() -> None:
    checkout_value = os.environ.get("L1B3RT4S_CHECKOUT")
    if checkout_value is None:
        pytest.skip("set L1B3RT4S_CHECKOUT for byte-for-byte source comparison")
    checkout = Path(checkout_value)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert commit == UPSTREAM_COMMIT

    for entry in load_manifest()["files"]:
        upstream = (checkout / entry["source_path"]).read_bytes()
        assert len(upstream) == entry["size"]
        assert hashlib.sha256(upstream).hexdigest() == entry["sha256"]
        if entry["bundled"]:
            assert load_source_bytes(entry["source_path"]) == upstream
