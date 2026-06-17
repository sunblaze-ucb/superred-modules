# Faithfulness verification

Compares this port against AgentDojo upstream for a fixed set of
``(suite, user_task, injection_task)`` pairs by driving the same prompt
and canonical injection through both sides at temperature 0 and checking
that their security verdicts agree.

The comparison is verdict-only: it compares the single win/lose security
boolean. The upstream side returns AgentDojo's ``security`` boolean (the
attack succeeded) from ``TaskSuite.run_task_with_pipeline``; the port
side runs ``Controller + AgentDojoTarget + Layer1Task`` and reads
``best_evaluation.primary_score.value`` (the attack-succeeded float),
coerced to bool. A pair matches when those two booleans are equal. There
is no environment-diff or function-call-trace comparison.

## What is implemented

`test_upstream_comparison.py` contains the test
``test_upstream_vs_port_security_verdict_match``, gated on real-LLM
credentials (``LITELLM_API_KEY`` + ``LITELLM_API_BASE`` or
``OPENAI_API_KEY``) and on the ``faithfulness_upstream`` opt-in marker
because it makes real LLM calls.

It runs a fixed list of 12 hand-picked pairs, defined in the
``FAITHFULNESS_PAIRS`` constant (3 each across banking, workspace, slack,
and travel). The port side uses a noop optimizer that never injects, so
the canonical AgentDojo injection defaults (already substituted into the
environment at seed-load time) are the only attack channel, matching
upstream's ``run_task_with_pipeline`` behaviour.

Because temperature 0 does not guarantee bit-for-bit reproducibility from
the OpenAI API, the test tolerates a small number of mismatches: it
passes when at least ``MATCH_FLOOR`` (= 10) of the 12 verdicts agree, so
one or two transient nondeterminism mismatches do not fail CI.

Cost is hard-capped at ``LLMConfig(max_cost=4.0)``; expected actual cost
at temperature 0 with ``gpt-4o-2024-05-13`` (or the LiteLLM substitution
``gpt-4-turbo-2024-04-09``) is about $1 to $3 for all 12 pairs together.
The benchmark version both sides use is imported from
``agentdojo_target.BENCHMARK_VERSION``, so the comparison tracks the
port's canonical version.

## How to run

```bash
# Direct OpenAI
OPENAI_API_KEY=sk-... pytest -m faithfulness_upstream tests/faithfulness/

# litellm proxy
LITELLM_API_KEY=... LITELLM_API_BASE=https://... \
    pytest -m faithfulness_upstream tests/faithfulness/
```

The model defaults to ``openai/gpt-4o-2024-05-13`` and can be overridden
via the ``AGENTDOJO_FAITHFULNESS_MODEL`` env var (the upstream side only
supports ``openai/*`` models and skips otherwise). The
``faithfulness_upstream`` marker is off by default in the package's
pytest config.

## Future work (not yet implemented)

The following are aspirational and have NOT been built:

### Full 629-case cross-product

A full cross-product over all four suites
(240 workspace + 105 slack + 140 travel + 144 banking = 629 cases) as a
broader faithfulness baseline.

Runner: `scripts/run_faithfulness_full.py` (does not exist yet). Plan of
attack:

1. Iterate every ``(suite, user_task, injection_task)`` triple from the
   four AgentDojo suites (currently pinned via the public
   ``agentdojo_target.BENCHMARK_VERSION`` constant).
2. On the upstream side, prefer AgentDojo's published ``runs/`` JSON
   logs (a free oracle for ``gpt-4o-2024-05-13`` and friends). When a
   log is absent, fall back to live invocation of
   ``task_suite.run_task_with_pipeline``.
3. On the port side, run via ``AgentDojoTarget`` with a passthrough
   optimizer that injects each upstream slot value into the
   corresponding per-read Controllable when fired.
4. Compare results; emit a per-pair pass/fail JSON.

Cost estimate (gpt-4o-mini): about $30 to $60 for one sweep. Wall time
at concurrency 16: a few hours. Recommended cadence: pre-release.
