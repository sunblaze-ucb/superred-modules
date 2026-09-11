# superred-target-injecagent

A superred **Target** that drives the agent-under-test through one
[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (Zhan et al., ACL
Findings 2024, arXiv:2403.02691) indirect-prompt-injection test case, over a
**fully simulated tool environment (no Docker)**. The poisoned tool observation
is baked into the case; the target builds the ReAct (or native function-calling)
prompt, calls the model, and — for a data-stealing case whose exfiltration step
succeeds — runs the second step with the exfiltration tool's response served
**cache-first** from the vendored cache (LLM fallback only on a genuine miss).

Pairs with the [`superred-claim-injecagent`](../../security_claims/injecagent)
package, whose pure, deterministic judge grades the captured outputs. This target
reuses that judge to gate its ds step 2, so the "did step 1 succeed?" decision
and the claim's final verdict can never disagree.

## Usage

```python
from injecagent_claim import injecagent_claim
from injecagent_target import injecagent_target_factory

claim = injecagent_claim()                                  # 1,054 base tasks
target = injecagent_target_factory(model="openai/gpt-4o-mini")

# function-calling instead of text ReAct:
ft = injecagent_target_factory(model="openai/gpt-4o-mini", mode="finetuned")

# deterministic (cache-only) ds scoring on the exfiltration step:
det = injecagent_target_factory(model="openai/gpt-4o-mini", only_first_step=True)
```

Construction knobs: `model` (agent under test), `mode` (`"prompted"` /
`"finetuned"`), `prompt_type` (`"InjecAgent"` / `"hwchase17_react"`),
`temperature` / `max_tokens` (InjecAgent uses `0.0` / `4096`), `only_first_step`
(score ds on the exfil step only — no simulation needed), and `sim_model` (model
for a ds step-2 cache miss; defaults to `gpt-4-0613` — upstream's simulator model —
override to reuse `model`).

The target exposes two injection surfaces an optimizer may drive: the injected
`attacker_instruction` (re-substituted into the poisoned tool observation, scoped
to the `external_data` trust boundary) and the benign `user_instruction` (scoped
to `user`). Left un-injected, the benchmark runs exactly as InjecAgent designed
it.

## Provenance & faithfulness

`tools.json` and `attacker_simulated_responses.json` are vendored **byte-for-byte**
from upstream (verify with `python scripts/sync_upstream.py --check`). The agent
prompts and ds-simulation prompt are vendored verbatim; the tool-dictionary
loader, prompt/message builders, and cache-first simulator are ported from
upstream. Deviations are documented in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

**Licensing.** Module code is MIT. `tools.json` is adapted by InjecAgent from
**ToolEmu** (Ruan et al., 2023, Apache-2.0); both licences are shipped in
`LICENSES/` and attributed in `NOTICE`.

End-to-end runs call an LLM (the agent under test, and — only on a ds step-2 cache
miss — the simulator); that path is not exercised in unit tests, which cover the
contract, the prompt/message construction, the cache-hit simulator, and the full
run orchestration with a stubbed model.
