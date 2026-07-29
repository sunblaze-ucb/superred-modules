"""Parity and Unicode-preservation tests for the pinned L1B3RT4S corpus."""

from __future__ import annotations

import hashlib
import os
import subprocess
import unicodedata
from pathlib import Path

import pytest

import libertas_optimizer.corpus as corpus_module
from libertas_optimizer.corpus import (
    UPSTREAM_COMMIT,
    PromptTemplate,
    detect_provider,
    load_manifest,
    load_prompt_templates,
    load_source_bytes,
    render_prompt,
    verify_bundled_corpus,
)


def test_bundled_snapshot_matches_manifest() -> None:
    assert verify_bundled_corpus() == []


def test_source_loader_detects_tampering_after_an_initial_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = "cache-parity-fixture.mkd"
    original = "exact λ bytes\n".encode()
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    stored = upstream_root / source_file
    stored.write_bytes(original)
    manifest = {
        "repository": corpus_module.UPSTREAM_REPOSITORY,
        "commit": corpus_module.UPSTREAM_COMMIT,
        "files": [
            {
                "source_path": source_file,
                "stored_path": source_file,
                "sha256": hashlib.sha256(original).hexdigest(),
                "size": len(original),
                "bundled": True,
                "reason": None,
            }
        ],
    }
    monkeypatch.setattr(corpus_module, "_data_root", lambda: tmp_path)
    monkeypatch.setattr(corpus_module, "load_manifest", lambda: manifest)

    assert load_source_bytes(source_file) == original
    stored.write_bytes(original[::-1])

    with pytest.raises(ValueError, match="sha256"):
        load_source_bytes(source_file)


def test_manifest_records_every_upstream_root_file() -> None:
    manifest = load_manifest()
    assert manifest["commit"] == UPSTREAM_COMMIT
    assert len(manifest["files"]) == 44
    assert sum(entry["bundled"] for entry in manifest["files"]) == 43

    omitted = {entry["source_path"] for entry in manifest["files"] if not entry["bundled"]}
    assert omitted == {"LICENSE"}
    assert all(entry["sha256"] and entry["size"] > 0 for entry in manifest["files"])


def test_module_license_matches_the_upstream_manifest_entry() -> None:
    entry = next(item for item in load_manifest()["files"] if item["source_path"] == "LICENSE")
    license_bytes = (Path(__file__).resolve().parent.parent / "LICENSE").read_bytes()

    assert len(license_bytes) == entry["size"]
    assert hashlib.sha256(license_bytes).hexdigest() == entry["sha256"]


def test_special_character_filenames_and_bytes_are_exact() -> None:
    manifest = load_manifest()
    stored = {entry["source_path"]: entry["stored_path"] for entry in manifest["files"]}
    assert stored["*SPECIAL_TOKENS.json"] == "*SPECIAL_TOKENS.json"
    assert stored["#MOTHERLOAD.txt"] == "#MOTHERLOAD.txt"
    assert stored["!SHORTCUTS.json"] == "!SHORTCUTS.json"
    assert stored["-MISCELLANEOUS-.mkd"] == "-MISCELLANEOUS-.mkd"
    assert stored["README.md"] == "README.md"

    for source_file in (
        "*SPECIAL_TOKENS.json",
        "#MOTHERLOAD.txt",
        "!SHORTCUTS.json",
    ):
        entry = next(item for item in manifest["files"] if item["source_path"] == source_file)
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
    assert len(templates) == 46
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


def test_structural_user_query_xml_is_not_a_goal_marker() -> None:
    templates = load_prompt_templates(provider="anthropic")

    assert all(template.heading != "Usage notes" for template in templates)


def test_predictive_reasoning_variable_is_a_strict_goal_marker() -> None:
    templates = load_prompt_templates(source_files=("-MISCELLANEOUS-.mkd",))
    template = next(item for item in templates if item.heading == "PREDICTIVE REASONING")

    rendered = render_prompt(template, "TARGET GOAL")

    assert "<VARIABLE Z = ________>" not in rendered
    assert "TARGET GOAL" in rendered


def test_rendering_does_not_replace_marker_text_inside_the_goal() -> None:
    raw = "{user_input/query} then {Z}"
    template = PromptTemplate(
        id="fixture:000",
        source_file="fixture.mkd",
        section_index=0,
        heading="Fixture",
        provider=None,
        delivery="user",
        raw_template=raw,
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        goal_markers=("{user_input/query}", "{Z}"),
    )

    rendered = render_prompt(template, "explain {Z} literally")

    assert rendered == "explain {Z} literally then explain {Z} literally"


def test_nvidia_level_two_model_headings_are_distinct_prompts() -> None:
    templates = load_prompt_templates(source_files=("NVIDIA.mkd",))

    assert [template.heading for template in templates] == [
        "LLAMA-3.1-NEMOTRON-70B",
        "NVIDIA NEMOTRON-4-340B",
    ]


@pytest.mark.parametrize(
    ("source_file", "expected_headings"),
    [
        ("AMAZON.mkd", {"AMAZON NOVA MODELS", "AMAZON RUFUS JAILBREAK\u2028\u2028"}),
        ("GOOGLE.mkd", {"GEMINI 1.5 PRO", "GEMINI 1.0 PRO"}),
        ("GRAYSWAN.mkd", {"Cygnet 1.0"}),
        ("META.mkd", {"LLAMA 4 MAVERICK ", "LLAMA-3.1-405B", "LLAMA-3-70B"}),
        ("PERPLEXITY.mkd", {"Updated"}),
    ],
)
def test_reviewed_level_two_model_headings_are_metadata(
    source_file: str,
    expected_headings: set[str],
) -> None:
    templates = load_prompt_templates(
        source_files=(source_file,),
        include_untemplated=True,
        include_system_templates=True,
    )

    headings = {template.heading for template in templates}
    assert expected_headings <= headings


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
