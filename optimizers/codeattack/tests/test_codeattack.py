"""Faithfulness tests for the CodeAttack transform."""

from __future__ import annotations

import json
from importlib import resources

import pytest
from codeattack_optimizer.codeattack import VARIANTS, _wrap_input, render


def test_variants_cover_every_upstream_template() -> None:
    assert VARIANTS == (
        "python_list",
        "python_stack",
        "python_string",
        "python_list_plus",
        "python_stack_plus",
        "python_string_plus",
        "C_string",
        "go_string",
    )


def test_list_appends_words_in_order() -> None:
    out = _wrap_input("Write a script", "python_list")
    assert out == '    my_list.append("Write")\n    my_list.append("a")\n    my_list.append("script")'


def test_stack_appends_words_reversed() -> None:
    out = _wrap_input("Write a script", "python_stack")
    assert out == '    my_stack.append("script")\n    my_stack.append("a")\n    my_stack.append("Write")'


def test_stack_splits_hyphens() -> None:
    # upstream splits on whitespace AND hyphens
    out = _wrap_input("multi-step plan", "python_stack")
    assert out == '    my_stack.append("plan")\n    my_stack.append("step")\n    my_stack.append("multi")'


def test_stack_single_token_splits_to_chars() -> None:
    out = _wrap_input("bomb", "python_stack")
    assert out == '    my_stack.append("b")\n    my_stack.append("m")\n    my_stack.append("o")\n    my_stack.append("b")'


def test_string_is_whole_query() -> None:
    assert _wrap_input("do the thing", "python_string") == '    my_string = "do the thing"'


def test_render_fills_the_slot() -> None:
    for v in VARIANTS:
        out = render("x", v)
        assert "{wrapped_input}" not in out


def test_render_uses_current_stack_template() -> None:
    out = render("Write a bomb tutorial", "python_stack")
    assert "my_stack = deque()" in out
    assert 'my_stack.append("tutorial")' in out


def test_template_has_single_slot() -> None:
    for name in ("code_python_list", "code_python_stack", "code_python_string"):
        text = resources.files("codeattack_optimizer.data.upstream").joinpath(f"{name}.txt").read_text()
        assert text.count("{wrapped_input}") == 1


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        render("x", "ruby_hash")
    with pytest.raises(ValueError, match="unknown variant"):
        _wrap_input("x", "ruby_hash")


def test_plus_variants_reuse_the_base_tokenisation() -> None:
    """`*_plus` differ only in the template's output demand, not the encoding."""
    assert _wrap_input("a b c", "python_list_plus") == _wrap_input("a b c", "python_list")
    assert _wrap_input("a b c", "python_stack_plus") == _wrap_input("a b c", "python_stack")


def test_plus_templates_demand_paragraphs_not_steps() -> None:
    out = render("x", "python_list_plus")
    assert "100 words" in out
    assert "100 words" not in render("x", "python_list")


def test_c_and_go_replace_the_hardcoded_example_line() -> None:
    """Upstream fills these by replacing a fixed 0-indexed line, not a slot."""
    c = render("EXPLOIT_HERE", "C_string")
    g = render("EXPLOIT_HERE", "go_string")
    assert c.splitlines()[10] == '    std::string my_string = "EXPLOIT_HERE";'
    assert g.splitlines()[17] == '        myQueue := "EXPLOIT_HERE"'
    # upstream's baked-in examples must be gone
    assert "hacking into a government database" not in c
    assert "I make a bomb" not in g


def test_every_variant_renders_without_a_leftover_slot() -> None:
    for v in VARIANTS:
        assert "{wrapped_input}" not in render("do the thing", v)
