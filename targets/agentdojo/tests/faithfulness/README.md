# Faithfulness verification

Compares this port against AgentDojo upstream for selected
``(user_task, injection_task)`` pairs.  Two sides agree if:

- function-call traces have the same function-name sequence (after
  stripping our ``{suite}__`` prefix),
- the post-environment per-suite sub-env matches via
  ``WorkspaceDeepDiff``-style equality (computed-field mirrors excluded),
- the upstream task's ``(utility, security)`` booleans match.

## v1 scope

`test_faithfulness_smoke.py` — gated on real-LLM credentials.  v1
contains a placeholder for one pair (banking UT1 + IT0).  Extending to
the brief's 12-pair smoke set is mechanical: add additional
parametrised cases.

## Full 629-case cross-product (v2)

The user confirmed the full cross-product as the faithfulness baseline:
240 workspace + 105 slack + 140 travel + 144 banking = 629 cases.

Runner: `scripts/run_faithfulness_full.py` (NOT yet implemented).  Plan
of attack:

1. Iterate every ``(suite, user_task, injection_task)`` triple from
   the four AgentDojo suites (currently pinned to v1.2.2 via
   ``seed_loader._BENCHMARK_VERSION``).
2. On the upstream side, prefer AgentDojo's published ``runs/`` JSON
   logs (free oracle for ``gpt-4o-2024-05-13`` and friends).  When a
   log is absent, fall back to live invocation of
   ``task_suite.run_task_with_pipeline``.
3. On the port side, run via :class:`AgentDojoTarget` with a
   ``PassthroughInjectionOptimizer`` that injects each upstream slot
   value into the corresponding per-read Controllable when fired.
4. Compare results; emit a per-pair pass/fail JSON.

Cost estimate (gpt-4o-mini): ~$30-60 for one sweep.  Wall time at
concurrency 16: a few hours.  Recommended cadence: pre-release.

## How to run the smoke test

```bash
# Direct OpenAI
OPENAI_API_KEY=sk-... pytest -m faithfulness tests/faithfulness/

# litellm proxy
LITELLM_API_KEY=... LITELLM_API_BASE=https://... \
    pytest -m faithfulness tests/faithfulness/
```

Both markers are off by default in the package's pytest config.
