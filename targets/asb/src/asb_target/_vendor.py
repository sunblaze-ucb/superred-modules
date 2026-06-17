"""Make the vendored ASB packages (``pyopenagi``, ``aios``) importable.

The vendored ASB source under ``_asb_vendor/`` uses absolute imports
(``from pyopenagi...``, ``from aios...``), so that directory must be on
``sys.path`` for those names to resolve. We append it (rather than
insert at the front) so a genuinely installed package of the same name
would still win, and we only do it once.

This is the "scoped shim" for the vendored tree. It does introduce the
top-level names ``pyopenagi`` and ``aios`` into the import namespace;
nothing else in this repo uses those names. See ASSUMPTIONS.md.
"""

from __future__ import annotations

import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_asb_vendor")

#: Upstream ASB commit the vendored tree was copied from.
ASB_UPSTREAM_SHA = "1f561dccf92d55302368fa67679b4ba9d9c8fdc4"


def ensure_vendor_on_path() -> None:
    """Append the vendored ASB directory to ``sys.path`` (idempotent)."""
    if _VENDOR_DIR not in sys.path:
        sys.path.append(_VENDOR_DIR)


__all__ = ["ensure_vendor_on_path", "ASB_UPSTREAM_SHA"]
