"""The installed claim must import without heavy/ML dependencies.

``pyarrow`` and ``huggingface_hub`` are present in the dev venv (they power
``scripts/sync_upstream.py``), so this checks in a clean subprocess that merely
importing ``aegis_claim`` and building a claim never pulls them — nor pandas,
torch, transformers, or datasets — into ``sys.modules``. The data loads through
the stdlib ``csv`` module.
"""

from __future__ import annotations

import subprocess
import sys

_HEAVY = ("pandas", "pyarrow", "torch", "transformers", "datasets")

_PROGRAM = f"""
import sys
import aegis_claim
from aegis_claim import aegis_claim as factory, RefusalHeuristicJudge, load_unsafe_user_prompts
claim = factory(judge=RefusalHeuristicJudge(), limit=3)
assert len(list(claim)) == 3
assert len(load_unsafe_user_prompts()) == 236
leaked = [m for m in {_HEAVY!r} if m in sys.modules]
assert not leaked, f"heavy deps imported: {{leaked}}"
print("ok")
"""


def test_imports_without_heavy_deps() -> None:
    # Inherit the parent's PYTHONPATH (the battery sets it) so the child can
    # resolve aegis_claim / superred / chatbot_target.
    proc = subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}"
    assert proc.stdout.strip().endswith("ok")
