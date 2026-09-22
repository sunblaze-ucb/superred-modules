"""Tests for the vendored-helper loader (structure, hermeticity, torch-free).

These assert *shape* only -- counts, names, the ``<>`` nesting slot -- and never
reproduce any vendored prompt/scenario/judge body.
"""

from __future__ import annotations

import sys

from renellm_optimizer import vendored


def test_load_exposes_six_operations_in_upstream_order() -> None:
    v = vendored.load()
    names = [op.__name__ for op in v.operations]
    assert names == [
        "shortenSentence",
        "misrewriteSentence",
        "changeOrder",
        "addChar",
        "languageMix",
        "styleChange",
    ]


def test_load_exposes_three_scenarios_with_nesting_slot() -> None:
    v = vendored.load()
    assert len(v.scenarios) == 3
    assert all("<>" in scenario for scenario in v.scenarios)


def test_judge_and_prompt_constant_present() -> None:
    v = vendored.load()
    assert callable(v.harmful_classification)
    assert isinstance(v.harm_judge_prompt, str) and v.harm_judge_prompt


def test_load_is_cached() -> None:
    assert vendored.load() is vendored.load()


def test_load_does_not_leak_utils_into_sys_modules() -> None:
    vendored.load()
    leaked = [k for k in sys.modules if k == "utils" or k.startswith("utils.")]
    assert leaked == []


def test_load_is_torch_free() -> None:
    vendored.load()
    for mod in ("torch", "transformers", "nltk", "sentence_transformers"):
        assert mod not in sys.modules
