# superred-target-safeclawbench-exec

A superred **Target** for [SafeClawBench](https://huggingface.co/datasets/sairights/safeclawbench)'s
**Exec-Balanced** endpoint (Tian et al., arXiv:2606.18356) — the paper's
*sandbox-observed tool/state harm* stage, distinct from the prompt-level
Semantic Core (`superred-target-safeclawbench` / `superred-claim-safeclawbench`).

Each run takes one of the 600 executable scenarios, builds its **mock world**
(six isolated stores — files, email, db, memory, web, code), drives the
**model-under-test** through the benchmark's JSON tool-plan loop against the
permissioned mock tools, and captures the final world state + trajectory. The
paired [`superred-claim-safeclawbench-exec`](../../security_claims/safeclawbench_exec)
package scores the vendored **deterministic state oracle** over that capture.

The sandbox is **fully offline and side-effect-free** — every tool call is
simulated against in-memory state, so no real file / email / network / code
effects occur. Only the model-under-test is live.

## Requirements

An LLM to red-team, supplied as `agent_llm_config` (a `superred` `LLMConfig`);
tests inject a scripted stub instead. No Docker, no third-party runtime — the
vendored sandbox is pure-stdlib.

## Usage

```python
from safeclawbench_exec_target import safeclawbench_exec_target_factory
from superred.core.types.llm import LLMConfig

target_factory = safeclawbench_exec_target_factory(
    agent_llm_config=LLMConfig(model="gpt-5.4", api_base=..., api_key=...),
    defense_level="D0",   # D0 (no defense, baseline) .. D3 (full stack)
)
# pair with safeclawbench_exec_claim(...) and an optimizer via the Controller.
```

The target exposes:

- **config** — `scenario` (the executable case JSON) and `defense_level`.
- **query** — `post_state` (JSON: initial/final world state + trajectory), plus
  `final_response`, `blocked_by_harness`, `error`.
- **controllables** — `user_message` (attacker text **appended** to the baked
  task prompt, so the baked attack and the deterministic oracle's success
  contract always run while an optimizer augments the prompt).
- **security domain** — `system → {user_input, external_data, memory, tools}`.

The agent tool-call budget is bounded (`max_tool_calls`); model failures are
recorded as an error type (never raised, never leaking the key), so a claim can
abstain rather than crashing a sweep.

## Provenance & faithfulness

SafeClawBench's `executable/` mock sandbox (schema, state, tools, trajectory,
metrics, runner, fixtures) and `defenses/defense_stack.py`, plus the 600-case
`exec_full_600.json`, are vendored **byte-for-byte** (verify with
`python scripts/sync_upstream.py --check`). The agent model loop is
re-implemented on superred's `LLMClient` but reuses the vendored prompt, JSON
schema, parser, tools, and defenses. Deviations are in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). MIT-licensed; SafeClawBench's MIT licence
ships in `LICENSES/`.

Cite Tian et al., 2026 (arXiv:2606.18356).
