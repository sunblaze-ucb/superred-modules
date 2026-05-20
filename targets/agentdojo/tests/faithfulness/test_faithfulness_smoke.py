"""Faithfulness verification smoke test (gated on real-LLM credentials).

Compares AgentDojo upstream's :meth:`TaskSuite.run_task_with_pipeline`
against our :class:`AgentDojoTarget` for a small set of pairs, driving
both through the same model.  The two sides must agree on:

- function-call trace function names (after stripping our ``{suite}__``
  prefix);
- the post-environment state (per-suite sub-env);
- the ``(utility, security)`` booleans from upstream's task methods.

For v1, this file contains the SCAFFOLDING and one smoke comparison
gated on credentials.  The full 629-case cross-product runner lives at
``scripts/run_faithfulness_full.py`` (TODO; not in v1) and uses
AgentDojo's published ``runs/`` JSON logs as a free oracle on the
upstream side.

The tests are skipped without ``LITELLM_API_KEY`` + ``LITELLM_API_BASE``
or ``OPENAI_API_KEY`` in the environment.
"""

from __future__ import annotations

import os

import pytest


def _has_creds() -> bool:
    if "LITELLM_API_KEY" in os.environ and "LITELLM_API_BASE" in os.environ:
        return True
    if "OPENAI_API_KEY" in os.environ:
        return True
    return False


pytestmark = pytest.mark.skipif(
    not _has_creds(),
    reason=(
        "Faithfulness smoke requires LLM credentials.  Set OPENAI_API_KEY "
        "or LITELLM_API_KEY+LITELLM_API_BASE."
    ),
)


@pytest.mark.faithfulness
@pytest.mark.asyncio
async def test_banking_ut1_it0_round_trip() -> None:
    """Smoke comparison for banking user_task_1 + injection_task_0.

    Drives both upstream and our port via the same model with the
    AgentDojo "Important instructions, no names" attack injected into
    the canonical slot.  The two sides must agree on
    ``(utility, security)``.

    TODO(v2): expand to the 12-pair smoke set + the full 629-case
    cross-product via a separate scripts/ runner.
    """
    pytest.skip(
        "v1 placeholder.  Implementation arrives with the full "
        "faithfulness runner; tracked in PLAN.md §P."
    )
