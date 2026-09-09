"""CodeAttack transforms, ported from renqibing/CodeAttack.

Upstream: https://github.com/renqibing/CodeAttack (MIT). The three Python code
templates under ``data/upstream/`` are byte-identical to CodeAttack's
``prompt_templates/code_python_{list,stack,string}.txt``; ``wrap`` reproduces
the tokenisation from ``data_preparation.py`` ``CodeAttack.wrap``.

Reference: Ren et al., "Exploring Safety Generalization Challenges of Large
Language Models via Code Completion" (arXiv:2403.07865).

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

#: Every template upstream ships. The ``*_plus`` variants demand 100-word
#: paragraphs per step instead of terse steps; ``C_string``/``go_string`` are
#: the non-Python scaffolds.
VARIANTS: tuple[str, ...] = (
    "python_list",
    "python_stack",
    "python_string",
    "python_list_plus",
    "python_stack_plus",
    "python_string_plus",
    "C_string",
    "go_string",
)

_TEMPLATE_FILE = {
    "python_list": "code_python_list.txt",
    "python_stack": "code_python_stack.txt",
    "python_string": "code_python_string.txt",
    "python_list_plus": "code_python_list_plus.txt",
    "python_stack_plus": "code_python_stack_plus.txt",
    "python_string_plus": "code_python_string_plus.txt",
    "C_string": "code_C_string.txt",
    "go_string": "code_go_string.txt",
}

#: Upstream fills the non-Python templates by *replacing* a fixed 0-indexed
#: line (``CodeAttack.wrap``'s ``replacement_line``) rather than substituting a
#: ``{wrapped_input}`` slot, because those scaffolds carry a hardcoded example.
_REPLACEMENT_LINE = {"C_string": 10, "go_string": 17}


@lru_cache(maxsize=None)
def _template(variant: str) -> str:
    return (
        resources.files("codeattack_optimizer.data.upstream")
        .joinpath(_TEMPLATE_FILE[variant])
        .read_text(encoding="utf-8")
    )


def _wrap_input(query: str, variant: str) -> str:
    """Tokenise the query into the variant's code fragment.

    Byte-faithful to upstream ``CodeAttack.wrap``:

    - ``python_list``: ``query.split()`` -> ``my_list.append("word")`` in order.
    - ``python_stack``: split on whitespace/hyphens, reverse, ->
      ``my_stack.append("word")`` (a stack pops in reverse). A single token is
      split into characters, as upstream does for scriptless languages.
    - ``python_string``: the whole query as one ``my_string = "..."``.

    No trailing newline is appended: upstream's shipped ``data_python_{list,
    string}_full.json`` are byte-identical to the current template filled this
    way (the repo's ``wrap`` adds a ``\n`` that its own materialised data does
    not carry -- an upstream inconsistency; the shipped data is the ground
    truth for what CodeAttack sends).
    """
    variant = variant[:-5] if variant.endswith("_plus") else variant
    if variant == "python_list":
        words = query.split()
        return "\n".join(f'    my_list.append("{w}")' for w in words)
    if variant == "python_stack":
        words = re.split(r"[\s\-]+", query)
        if len(words) == 1:
            words = list(words[0])
        words = words[::-1]
        return "\n".join(f'    my_stack.append("{w}")' for w in words)
    if variant == "python_string":
        return f'    my_string = "{query}"'
    if variant == "C_string":
        return f'    std::string my_string = "{query}";\n'
    if variant == "go_string":
        return f'        myQueue := "{query}"\n'
    raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")


def render(query: str, variant: str = "python_stack") -> str:
    """Wrap *query* as a code-completion prompt for the given variant.

    Faithful to upstream: tokenise via :func:`_wrap_input`, then inject into the
    template's ``{wrapped_input}`` slot. ``python_stack`` is upstream's headline
    variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    wrapped = _wrap_input(query, variant)
    line_no = _REPLACEMENT_LINE.get(variant)
    if line_no is None:
        return _template(variant).format(wrapped_input=wrapped)
    # Byte-faithful to upstream ``CodeAttack.replace``: emit the replacement
    # instead of the line at that 0-indexed position, appending it if the
    # template turned out to be shorter than the index.
    out = ""
    index = -1
    for index, line in enumerate(_template(variant).splitlines(keepends=True)):
        out += wrapped if index == line_no else line
    if line_no > index:
        out += wrapped
    return out
