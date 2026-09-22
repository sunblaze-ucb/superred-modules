"""Access to the byte-identically-vendored PyRIT Anecdoctor prompt templates.

The upstream prompt bodies are never retyped into this package. They live under
``_vendor/pyrit/datasets/executors/anecdoctor/`` exactly as shipped by PyRIT at
the pinned commit (see ``scripts/sync_upstream.py``) and are read from there at
runtime.

Upstream ``AnecdoctorGenerator._load_prompt_from_yaml`` resolves each template
under ``EXECUTOR_SEED_PROMPT_PATH / "anecdoctor" / <file>`` and returns the
``value`` key. :func:`load_prompt` reproduces exactly that resolution against
the vendored copy, so the loader logic lives here and no template text is
reproduced in authored code. Keeping the bodies in ``_vendor/`` lets the sync
checker byte-compare them against upstream and honours the content rule.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Any

import yaml

_PKG = "anecdoctor_optimizer"
#: Path of the vendored anecdoctor dataset dir relative to the package root,
#: mirroring the upstream ``pyrit/datasets/executors/anecdoctor/`` layout.
_VENDOR_ROOT = ("_vendor", "pyrit", "datasets", "executors", "anecdoctor")

#: Vendored template filenames (upstream ``AnecdoctorGenerator`` constants).
BUILD_KG_YAML = "anecdoctor_build_knowledge_graph.yaml"
USE_KG_YAML = "anecdoctor_use_knowledge_graph.yaml"
USE_FEWSHOT_YAML = "anecdoctor_use_fewshot.yaml"


def _read_vendored(yaml_filename: str) -> str:
    """Return the text of a vendored anecdoctor YAML file."""
    resource = files(_PKG)
    for part in (*_VENDOR_ROOT, yaml_filename):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


@cache
def load_yaml(yaml_filename: str) -> dict[str, Any]:
    """Load a vendored anecdoctor YAML file as a dict."""
    return yaml.safe_load(_read_vendored(yaml_filename))


@cache
def load_prompt(yaml_filename: str) -> str:
    """Return the ``value`` prompt template from a vendored anecdoctor YAML file.

    Mirrors upstream ``AnecdoctorGenerator._load_prompt_from_yaml``: read the
    YAML file and return the ``value`` key as a string.

    Raises:
        KeyError: If the ``value`` key is missing from the YAML data.
    """
    return str(load_yaml(yaml_filename)["value"])
