"""Access to the byte-identically-vendored PyRIT Skeleton Key seed prompts.

The upstream prompt bodies are never retyped into this package. They live under
``_vendor/pyrit/`` exactly as shipped by Microsoft PyRIT at the pinned commit
(see ``scripts/sync_upstream.py``) and are read from there at runtime.

The two Skeleton Key ``.prompt`` files are PyRIT ``SeedDataset`` YAML documents:
a top-level ``seeds`` list whose entries each carry a ``value`` (plus metadata
such as ``data_type`` and ``description``). Upstream's
``SkeletonKeyAttack`` reads them with
``SeedDataset.from_yaml_file(path).prompts[0].value`` -- the first prompt-typed
seed's ``value``. :func:`load_seed_prompt_value` reproduces exactly that access
with the standard library plus PyYAML, so none of PyRIT's Pydantic model stack
(or any heavy dependency) is imported.

Keeping the bodies in ``_vendor/`` means the sync checker can byte-compare them
against upstream and the content rule is honoured (this file contains loader
logic only, no prompt text).
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Any

import yaml

_PKG = "skeleton_key_optimizer"
#: The vendored Skeleton Key dataset directory, mirroring the upstream repo
#: layout ``pyrit/datasets/executors/skeleton_key/`` under ``_vendor/pyrit/``.
_VENDOR_ROOT = ("_vendor", "pyrit", "datasets", "executors", "skeleton_key")

#: File names as shipped upstream.
SKELETON_KEY_PROMPT_FILE = "skeleton_key.prompt"
SKELETON_KEY_ACCEPTANCE_FILE = "skeleton_key_acceptance.prompt"


def _read_vendored(name: str) -> str:
    """Return the text of a vendored ``.prompt`` file by its file name."""
    resource = files(_PKG)
    for part in (*_VENDOR_ROOT, name):
        resource = resource.joinpath(part)
    return resource.read_text(encoding="utf-8")


def _is_prompt_seed(seed: dict[str, Any], dataset_default: str) -> bool:
    """Whether a raw seed dict is a *prompt*-typed seed (PyRIT's ``.prompts``).

    PyRIT defaults a seed's type to the dataset-level ``seed_type`` or, absent
    that, ``"prompt"`` (``SeedDataset._build_seeds``), and ``.prompts`` keeps
    only the ``SeedPrompt`` (prompt-typed) seeds. We mirror that so a dataset
    that mixes prompt and non-prompt seeds still selects the first prompt.
    """
    return str(seed.get("seed_type") or dataset_default) == "prompt"


@cache
def load_seed_prompt_value(name: str) -> str:
    """Return the first prompt seed's ``value`` from a vendored ``.prompt`` file.

    Equivalent to upstream
    ``SeedDataset.from_yaml_file(name).prompts[0].value`` but implemented with
    stdlib + PyYAML only.

    Args:
        name: The vendored file name (e.g. :data:`SKELETON_KEY_PROMPT_FILE`).

    Returns:
        The seed's ``value`` string, byte-for-byte as upstream wrote it.

    Raises:
        ValueError: If the document has no seeds.
        KeyError: If no prompt-typed seed carries a ``value``.
    """
    data = yaml.safe_load(_read_vendored(name))
    if not isinstance(data, dict) or not data.get("seeds"):
        raise ValueError(f"vendored seed file {name!r} has no seeds")
    dataset_default = str(data.get("seed_type") or "prompt")
    for seed in data["seeds"]:
        if isinstance(seed, dict) and _is_prompt_seed(seed, dataset_default) and "value" in seed:
            return str(seed["value"])
    raise KeyError(f"no prompt-typed seed with a value in {name!r}")


def load_skeleton_key_prompt() -> str:
    """The Skeleton Key preamble (upstream default ``skeleton_key.prompt``)."""
    return load_seed_prompt_value(SKELETON_KEY_PROMPT_FILE)


def load_skeleton_key_acceptance() -> str:
    """The simulated acceptance (upstream default ``skeleton_key_acceptance.prompt``)."""
    return load_seed_prompt_value(SKELETON_KEY_ACCEPTANCE_FILE)


__all__ = [
    "SKELETON_KEY_ACCEPTANCE_FILE",
    "SKELETON_KEY_PROMPT_FILE",
    "load_seed_prompt_value",
    "load_skeleton_key_acceptance",
    "load_skeleton_key_prompt",
]
