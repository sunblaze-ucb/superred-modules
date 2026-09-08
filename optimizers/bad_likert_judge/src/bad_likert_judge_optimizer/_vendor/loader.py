"""Load DeepTeam's Bad Likert Judge templates without forking them.

``deepteam_blj/template.py`` is a byte-identical copy of upstream, so it keeps
upstream's absolute import::

    from deepteam.attacks.multi_turn.base_template import BaseMultiTurnTemplate

Rather than edit that line (which would fork the file and break
``sync_upstream.py``), the loader registers the vendored ``base_template`` under
exactly that module name before executing the template. Only the two module
names DeepTeam's import needs are registered, and only if nothing already
occupies them -- so a real ``deepteam`` installation in the same environment is
never shadowed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent / "deepteam_blj"
_BASE_MODULE = "deepteam.attacks.multi_turn.base_template"


def _base_is_resolvable() -> bool:
    """Whether the real ``deepteam`` can already satisfy the base import.

    Checks *install* state, not import state. ``_BASE_MODULE not in
    sys.modules`` only says deepteam has not been imported yet, and because
    imports are lazy an installed-but-unimported deepteam would then be
    replaced by an empty stub -- after which any real ``import deepteam.x``
    fails against that stub's empty ``__path__``. When a real deepteam is
    present this returns True and nothing is registered at all, so it is used
    as-is.
    """
    if _BASE_MODULE in sys.modules:
        return True
    try:
        return importlib.util.find_spec(_BASE_MODULE) is not None
    except (ImportError, ValueError):
        # A parent package that cannot be imported cannot provide the base.
        return False


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def bad_likert_judge_template() -> type:
    """Return upstream's ``BadLikertJudgeTemplate`` class, unmodified."""
    if not _base_is_resolvable():
        # Nothing can satisfy the vendored template's absolute import, so
        # provide it: the package chain it walks, then the vendored base.
        for part in ("deepteam", "deepteam.attacks", "deepteam.attacks.multi_turn"):
            if part not in sys.modules:
                pkg = types.ModuleType(part)
                pkg.__path__ = []  # marks it as a package
                sys.modules[part] = pkg
        sys.modules[_BASE_MODULE] = _load(
            "_superred_blj_base_template", _HERE / "base_template.py"
        )

    template = _load("_superred_blj_template", _HERE / "template.py")
    return template.BadLikertJudgeTemplate


__all__ = ["bad_likert_judge_template"]
