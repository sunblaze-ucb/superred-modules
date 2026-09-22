"""Load ReNeLLM's byte-identically-vendored, torch-free helpers.

Upstream ReNeLLM (``NJUNLP/ReNeLLM`` @ ``a61c39e``) ships its rewrite/nest/judge
logic under a top-level ``utils`` package and imports it with absolute paths
(e.g. ``from utils.data_utils import remove_number_prefix``). We vendor those
four torch-free modules byte-for-byte under ``_vendor/renellm/utils/`` (never
retyped -- see ``scripts/sync_upstream.py`` and ``_vendor/SHA256SUMS``) and
*execute* them, so the upstream rewrite/judge behaviour (including candidate
selection and number-prefix stripping) is faithful rather than re-implemented.

Because the vendored files cannot be edited, their ``from utils.X import Y``
lines must resolve at import time. :func:`load` therefore installs, for the
duration of one hermetic load, a private ``utils`` package pointing at the
vendored directory plus our :mod:`renellm_optimizer._shim` as
``utils.llm_completion_utils`` (replacing upstream's SDK helper). It imports the
four modules, captures direct references, and restores ``sys.modules`` -- the
imported modules keep working afterwards because their ``from utils.X import Y``
names are already bound. Nothing here reproduces any prompt/template body.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from renellm_optimizer import _shim

_VENDOR_ROOT = Path(__file__).resolve().parent / "_vendor" / "renellm"
_UTILS_DIR = _VENDOR_ROOT / "utils"

# Load order matters: modules with no intra-``utils`` imports first, so a later
# module's ``from utils.X import Y`` finds an already-registered X.
_UTILS_MODULE_NAMES = (
    "data_utils",
    "scenario_nest_utils",
    "harmful_classification_utils",
    "prompt_rewrite_utils",
)

_lock = threading.Lock()
_cache: Vendored | None = None


@dataclass(frozen=True)
class Vendored:
    """Direct references into the executed vendored helpers.

    Attributes:
        operations: The six rewrite operations, in upstream index order
            (``shortenSentence`` .. ``styleChange``). Each is ``(args, str) ->
            str`` and issues one rewrite LLM call through the shim.
        scenarios: The three nesting scenario templates (``<>`` is the goal
            slot). Loaded, never reproduced in source.
        harmful_classification: The binary LLM judge ``(args, str) -> str``
            returning ``"1"``/``"0"``; used both to keep a rewrite harmful and
            to score the target reply.
        harm_judge_prompt: The judge's instruction constant (exposed only so
            offline tests can route a mocked LLM by identity, not by text).
    """

    operations: tuple[Callable[[object, str], str], ...]
    scenarios: tuple[str, ...]
    harmful_classification: Callable[[object, str], str]
    harm_judge_prompt: str


def _load_module(name: str) -> ModuleType:
    path = _UTILS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"utils.{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build spec for vendored utils.{name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"utils.{name}"] = module
    spec.loader.exec_module(module)
    return module


def load() -> Vendored:
    """Import (once) and return references to the vendored ReNeLLM helpers."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache

        saved = {k: v for k, v in sys.modules.items() if k == "utils" or k.startswith("utils.")}
        for key in saved:
            del sys.modules[key]
        try:
            pkg = ModuleType("utils")
            pkg.__path__ = [str(_UTILS_DIR)]  # type: ignore[attr-defined]
            sys.modules["utils"] = pkg
            sys.modules["utils.llm_completion_utils"] = _shim
            mods = {name: _load_module(name) for name in _UTILS_MODULE_NAMES}
        finally:
            for key in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
                del sys.modules[key]
            sys.modules.update(saved)

        rewrite = mods["prompt_rewrite_utils"]
        nest = mods["scenario_nest_utils"]
        judge = mods["harmful_classification_utils"]
        _cache = Vendored(
            operations=(
                rewrite.shortenSentence,
                rewrite.misrewriteSentence,
                rewrite.changeOrder,
                rewrite.addChar,
                rewrite.languageMix,
                rewrite.styleChange,
            ),
            scenarios=tuple(nest.SCENARIOS),
            harmful_classification=judge.harmful_classification,
            harm_judge_prompt=judge.HARM_JUDGE_PROMPT,
        )
        return _cache
