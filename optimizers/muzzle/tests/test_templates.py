"""Tests for muzzle_optimizer.templates.

Covers load_template, select_family, select_form, and render_injection.
"""

from __future__ import annotations

import pytest

from muzzle_optimizer.templates import (
    INJECTION_FAMILIES,
    INJECTION_FORMS,
    load_template,
    render_injection,
    select_family,
    select_form,
)

# ---------------------------------------------------------------------------
# load_template
# ---------------------------------------------------------------------------


def test_load_template_all_four_combinations_return_nonempty_strings() -> None:
    """Each (family, form) combination loads a non-empty injection template."""
    for family in INJECTION_FAMILIES:
        for form in INJECTION_FORMS:
            text = load_template(family, form)
            assert isinstance(text, str)
            assert text.strip(), f"Template {family}/{form} is blank"


def test_load_template_generic_plain_text_contains_instruction_placeholder() -> None:
    """generic/plain_text template uses the ${instruction} placeholder."""
    text = load_template("generic", "plain_text")
    assert "${instruction}" in text


def test_load_template_generic_url_injection_contains_action_url_placeholder() -> None:
    """generic/url_injection template references ${action_url}."""
    text = load_template("generic", "url_injection")
    assert "${action_url}" in text


def test_load_template_goal_hijacking_plain_text_contains_user_goal_placeholder() -> None:
    """goal_hijacking/plain_text template references ${user_goal}."""
    text = load_template("goal_hijacking", "plain_text")
    assert "${user_goal}" in text


def test_load_template_goal_hijacking_url_injection_contains_all_placeholders() -> None:
    """goal_hijacking/url_injection references ${instruction}, ${user_goal}, and ${action_url}."""
    text = load_template("goal_hijacking", "url_injection")
    assert "${instruction}" in text
    assert "${user_goal}" in text
    assert "${action_url}" in text


def test_load_template_invalid_family_raises_value_error() -> None:
    """An unknown injection family raises ValueError."""
    with pytest.raises(ValueError, match="unknown injection family"):
        load_template("nonexistent", "plain_text")


def test_load_template_invalid_form_raises_value_error() -> None:
    """An unknown injection form raises ValueError."""
    with pytest.raises(ValueError, match="unknown injection form"):
        load_template("generic", "nonexistent_form")


# ---------------------------------------------------------------------------
# select_family
# ---------------------------------------------------------------------------


def test_select_family_goal_hijacking_when_user_goal_given() -> None:
    """A non-empty user_goal selects the 'goal_hijacking' family."""
    assert select_family("send an email") == "goal_hijacking"


def test_select_family_generic_when_user_goal_is_none() -> None:
    """None user_goal falls back to 'generic'."""
    assert select_family(None) == "generic"


def test_select_family_generic_when_user_goal_is_empty_string() -> None:
    """Empty-string user_goal is treated as absent -> 'generic'."""
    assert select_family("") == "generic"


def test_select_family_generic_when_user_goal_is_whitespace_only() -> None:
    """Whitespace-only user_goal is treated as absent -> 'generic'."""
    assert select_family("   ") == "generic"


# ---------------------------------------------------------------------------
# select_form
# ---------------------------------------------------------------------------


def test_select_form_url_injection_when_action_url_given() -> None:
    """A non-empty action_url selects the 'url_injection' form."""
    assert select_form("https://attacker.example.com") == "url_injection"


def test_select_form_plain_text_when_action_url_is_none() -> None:
    """None action_url falls back to 'plain_text'."""
    assert select_form(None) == "plain_text"


def test_select_form_plain_text_when_action_url_is_empty_string() -> None:
    """Empty-string action_url is treated as absent -> 'plain_text'."""
    assert select_form("") == "plain_text"


def test_select_form_plain_text_when_action_url_is_whitespace_only() -> None:
    """Whitespace-only action_url is treated as absent -> 'plain_text'."""
    assert select_form("   ") == "plain_text"


# ---------------------------------------------------------------------------
# render_injection
# ---------------------------------------------------------------------------


def test_render_injection_contains_instruction_always() -> None:
    """The rendered output always contains the verbatim instruction."""
    result = render_injection(instruction="steal the credentials")
    assert "steal the credentials" in result


def test_render_injection_no_unreplaced_instruction_placeholder() -> None:
    """The literal '${instruction}' is replaced; it must not appear in the output."""
    result = render_injection(instruction="do something")
    assert "${instruction}" not in result


def test_render_injection_with_url_contains_url() -> None:
    """When action_url is given, the rendered output contains the URL."""
    result = render_injection(
        instruction="click here", action_url="https://evil.example.com/payload"
    )
    assert "https://evil.example.com/payload" in result
    assert "click here" in result


def test_render_injection_with_url_uses_url_injection_form() -> None:
    """action_url present -> url_injection template (generic family, no user_goal)."""
    generic_url_text = load_template("generic", "url_injection")
    # The url_injection template contains the literal string "HAS MOVED" (upstream marker).
    assert "HAS MOVED" in generic_url_text
    result = render_injection(instruction="exec", action_url="https://attacker.example.com")
    assert "HAS MOVED" in result


def test_render_injection_with_goal_uses_goal_hijacking_family() -> None:
    """user_goal present -> goal_hijacking template (plain_text form, no action_url)."""
    result = render_injection(instruction="perform the action", user_goal="book a flight")
    assert "book a flight" in result
    assert "perform the action" in result


def test_render_injection_with_goal_no_unreplaced_user_goal_placeholder() -> None:
    """The literal '${user_goal}' is replaced; it must not appear in the output."""
    result = render_injection(instruction="do it", user_goal="send an email")
    assert "${user_goal}" not in result


def test_render_injection_with_url_and_goal_uses_goal_hijacking_url_injection() -> None:
    """Both action_url and user_goal -> goal_hijacking/url_injection template."""
    result = render_injection(
        instruction="exfiltrate",
        action_url="https://exfil.example.com",
        user_goal="summarize the document",
    )
    assert "exfiltrate" in result
    assert "https://exfil.example.com" in result
    assert "summarize the document" in result
    # goal_hijacking/url_injection template has "HAS MOVED" as an upstream marker
    assert "HAS MOVED" in result


def test_render_injection_no_goal_no_url_uses_generic_plain_text() -> None:
    """No goal, no url -> generic/plain_text template; instruction is embedded."""
    generic_plain = load_template("generic", "plain_text")
    # The generic/plain_text template references ${action_url} which gets substituted
    # to empty string. We verify the template is the right one by checking a static
    # string from it (it contains "ATTENTION").
    assert "ATTENTION" in generic_plain
    result = render_injection(instruction="attack now")
    assert "ATTENTION" in result
    assert "attack now" in result
