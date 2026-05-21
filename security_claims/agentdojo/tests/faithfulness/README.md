# Faithfulness verification

Tests in this directory verify that our port of AgentDojo's task
semantics agrees with upstream's predicates.  They run with the
default `pytest` invocation (no special marker required).

## What's here

### `test_ground_truth_replay.py` (Option 1 in the design discussion)

For every AgentDojo v1 user task (97) and injection task (27):

- Replay the upstream `ground_truth(env)` function-call list through
  our `WrappedFunctionsRuntime`-equivalent runtime (a plain
  `FunctionsRuntime` over `tool_registry.ALL_FUNCTIONS` — same
  suite-prefixed catalogue, same `Depends` rebinding) against a
  composite env whose target suite holds the upstream-prepared
  `pre_environment` and the other three suites hold their defaults.
- Compare the post-env / trace / model_output against upstream's
  `utility` or `security` predicate (via `*_from_traces` when
  defined; falling back to the post-env variant; falling back to
  False with a warning on `NotImplementedError`).
- Assert the predicate returns True.

This validates, **at zero LLM cost**, that:

- Our composite env loads each suite's seed correctly.
- Suite-prefix rewriting + `Depends` rebinding route every tool to
  the right sub-env.
- `sync_initial_fields` makes the round-trip faithful (the
  predicate sees the agent's mutations).
- Upstream's predicates fire correctly against our post-env.
- The Layer-1 evaluate dispatch (`*_from_traces` → fallback) is
  consistent with what upstream would compute in-memory.

7 cases are `xfail`-marked because of **upstream v1 bugs** between
`ground_truth` and the matching predicate (e.g., banking UT5's GT
writes the wrong amount, workspace UT7's reschedule predicate
forgets that the tool also adjusts end_time).  These divergences
were fixed in upstream v1.1+; documenting them here keeps them
visible.

Total: 117 passing GT replays + 7 xfailed divergences.

## What's deferred (Option 2)

Replay AgentDojo's published `runs/*.json` log files through our
pipeline (no new LLM calls on the upstream side; recorded messages
become the oracle for ~12-20 representative pairs).  Not yet
implemented because:

1. The `runs/` directory is **not** included in the PyPI
   distribution of `agentdojo`; we'd have to git-fetch or vendor
   the JSONs.
2. Replaying upstream's messages through our pipeline requires a
   "messages-only LLM" pipeline element (returns recorded assistant
   turns instead of calling an LLM) — straightforward but new code.
3. The "compare verdicts" oracle requires mapping AgentDojo's
   static-slot injection model to our per-read controllable model
   so the right injection-vector content is presented to the
   correct read tool during replay.

Estimated work: 0.5-1 day to implement; ~$0.50-2 of LLM cost for
the side that does need a real LLM (if we run our port against a
real model and use the upstream JSONs as the oracle).  Marked as
v2 follow-up.

## What's already covered by the SecurityClaim smoke

`tests/smoke/run.py` is a credential-gated end-to-end run against a
real LLM, executing a Layer-1 paired task through `Controller +
AgentDojoTarget + passthrough optimizer`.  It complements the
zero-cost replay tests with real LLM behaviour but isn't a
faithfulness assertion (it doesn't compare to upstream).

## How to run

```bash
cd security_claims/agentdojo
pip install -e ".[dev]"

# Faithfulness tests (default: included in main run)
pytest tests/faithfulness/ -v

# Real-LLM smoke (gated on creds)
OPENAI_API_KEY=sk-... python tests/smoke/run.py
# OR
LITELLM_API_KEY=... LITELLM_API_BASE=... python tests/smoke/run.py
```
