"""Print `<path>[<extra>]` for a module's test extra, if it declares one.

Modules are inconsistent about the name — some call it ``dev``, others
``test`` — and twelve set a pytest ``addopts`` containing ``--cov``, so the
extra is where ``pytest-cov`` and other test-only requirements come from.
Printing nothing is fine and means the module declares neither.

Usage: python .github/scripts/extras_for.py targets/chatbot
"""

import pathlib
import sys
import tomllib

mod = sys.argv[1].rstrip("/")
extras = tomllib.load(open(pathlib.Path(mod) / "pyproject.toml", "rb"))["project"].get(
    "optional-dependencies", {}
)
for name in ("dev", "test"):
    if name in extras:
        print(f"{mod}[{name}]")
        break
