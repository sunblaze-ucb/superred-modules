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

# The three clean Python variants (upstream's C_string/go_string use a
# line-index replace marked TODO upstream; not ported -- see ASSUMPTIONS.md).
VARIANTS: tuple[str, ...] = ("python_list", "python_stack", "python_string")

_TEMPLATE_FILE = {
    "python_list": "code_python_list.txt",
    "python_stack": "code_python_stack.txt",
    "python_string": "code_python_string.txt",
}


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
    raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")


def render(query: str, variant: str = "python_stack") -> str:
    """Wrap *query* as a code-completion prompt for the given variant.

    Faithful to upstream: tokenise via :func:`_wrap_input`, then inject into the
    template's ``{wrapped_input}`` slot. ``python_stack`` is upstream's headline
    variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    return _template(variant).format(wrapped_input=_wrap_input(query, variant))
