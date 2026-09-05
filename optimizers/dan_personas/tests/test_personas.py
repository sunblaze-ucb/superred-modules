"""Faithfulness tests for the vendored DAN corpus.

Pins the vendored data and the placeholder substitution to upstream, so drift
in either fails the suite rather than a scan.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest
from dan_personas_optimizer.personas import (
    GENERATOR_NAME_PLACEHOLDER,
    PERSONA_NAMES,
    Persona,
    load_personas,
)

UPSTREAM = "dan_personas_optimizer.data.upstream"


def test_all_thirteen_personas_load() -> None:
    personas = load_personas()
    assert len(personas) == 13
    assert [p.name for p in personas] == list(PERSONA_NAMES)


def test_preambles_are_byte_identical_to_vendored_json() -> None:
    """The loader must not reformat upstream text beyond stripping."""
    root = resources.files(UPSTREAM)
    for persona in load_personas():
        raw = json.loads(root.joinpath(f"{persona.name}.json").read_text("utf-8"))
        assert persona.preamble == str(raw[0]).strip()


def test_every_preamble_is_substantial() -> None:
    # shortest upstream persona (DUDE) is ~500 chars; guards against empty files
    for persona in load_personas():
        assert len(persona.preamble) > 400, persona.name


def test_no_preamble_carries_a_goal_slot() -> None:
    """Upstream personas are primers, not templates: the only placeholder is
    the generator name. A goal slot would mean the port is misreading them."""
    for persona in load_personas():
        assert "{goal}" not in persona.preamble
        for token in ("{0}", "{}"):
            assert token not in persona.preamble


def test_render_substitutes_generator_name() -> None:
    persona = Persona(name="x", preamble=f"Hello {GENERATOR_NAME_PLACEHOLDER}.")
    assert persona.render("Claude") == "Hello Claude."


def test_render_defaults_to_chatgpt() -> None:
    persona = Persona(name="x", preamble=f"Hello {GENERATOR_NAME_PLACEHOLDER}.")
    assert persona.render(None) == "Hello ChatGPT."


def test_render_leaves_prompts_without_placeholder_untouched() -> None:
    """Mirrors upstream, which only formats when the placeholder is present."""
    persona = Persona(name="x", preamble="No placeholder here {weird}.")
    assert persona.render("Claude") == "No placeholder here {weird}."


def test_render_is_applied_to_the_real_corpus() -> None:
    rendered = [p.render("Claude") for p in load_personas()]
    assert all(GENERATOR_NAME_PLACEHOLDER not in text for text in rendered)


def test_subset_selection_preserves_requested_order() -> None:
    personas = load_personas(["STAN", "DUDE"])
    assert [p.name for p in personas] == ["STAN", "DUDE"]


def test_unknown_persona_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown persona"):
        load_personas(["NotAPersona"])


def test_image_markdown_persona_is_not_vendored() -> None:
    """Excluded deliberately: it is the multimodal probe."""
    assert "ChatGPT_Image_Markdown" not in PERSONA_NAMES
    assert not resources.files(UPSTREAM).joinpath("ChatGPT_Image_Markdown.json").is_file()
