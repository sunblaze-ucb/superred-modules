# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) |
| Path | `data/tools.json`, `data/attacker_simulated_responses.json`, `src/prompts/*.py`, `src/utils.py`, `src/evaluate_{prompted,finetuned}_agent.py` |
| Commit | `f19c9f2` |
| Paper | Zhan et al., ACL Findings 2024 (arXiv:2403.02691) |
| Licence | MIT (© 2023 Qiusi Zhan); `tools.json` derives from ToolEmu (Apache-2.0) |

## Vendored byte-for-byte

- `src/injecagent_target/data/tools.json` — 330 tool specs / 38 toolkits.
- `src/injecagent_target/data/attacker_simulated_responses.json` — InjecAgent's
  cache of simulated exfiltration-tool responses (2,347 entries) for the ds step 2.

`scripts/sync_upstream.py --check` re-downloads the pinned commit and diffs both.

## Vendored verbatim (prompts, code strings)

- `_vendor/agent_prompts.py` — `SYS_PROMPT`, `USER_PROMPT`, `USER_PROMPT_SIMPLE`,
  `PROMPT_DICT` (both prompt types, byte-identical to `src/prompts/agent_prompts.py`).
- `_vendor/generation_prompts.py` — `SYSTEM_MESSAGE`, `ENV_INFO`,
  `DS_ATTACKER_TOOL_RESPONSE_GEN_MESSAGE`, `EXAMPLE` (byte-identical to the
  corresponding constants in `src/prompts/generation_prompts.py`). Kept under
  `_vendor` (excluded from ruff) so the verbatim text is never reformatted.

## Upstream coverage (Simon rule: skip nothing)

- **Both evaluation modes.** `mode="prompted"` (text ReAct — the paper's headline
  path, from `evaluate_prompted_agent.py`) and `mode="finetuned"` (native
  function-calling, from `evaluate_finetuned_agent.py`). The prompt/message
  construction for each is ported verbatim (`agent.build_prompted_input` /
  `agent.build_finetuned_messages`), including the fixed synthetic call id.
- **Both prompt types** (`InjecAgent`, `hwchase17_react`) and both settings
  (`base`, `enhanced`, chosen by the claim's corpus, carried in each case).
- **The ds two-step**, gated exactly as upstream interleaves it: step 1 is judged
  (reusing the claim's evaluator) and, on success, step 2 runs with the exfil
  tool's simulated response and is judged for the second attacker tool.
- **`only_first_step`** (upstream `--only_first_step`): score ds on the
  exfiltration step alone — deterministic (no simulated step-2 response).

## superred adaptation

- **The environment is fully simulated (no Docker).** Upstream pre-bakes the
  poisoned tool observation into each case and simulates the ds step-2
  exfiltration response; this target does the same, so a run is a single model
  call (two for a succeeded ds case). This is inherent to InjecAgent, not a
  simplification.
- **litellm replaces `src/models.py`.** Upstream ships per-provider backends
  (`ClaudeModel`/`GPTModel`/`LlamaModel`/`TogetherAIModel`) and a 160-entry raw
  prompt-template table for base models. The target reaches every provider through
  litellm's chat abstraction (`agent.call_prompted` / `agent.call_finetuned`),
  which subsumes that provider-specific plumbing. The benchmark-semantic prompts
  (the two prompt types) are ported verbatim; the plumbing is not.
- **Injection surfaces for an optimizer (Simon rule: LLM/optimizer surface).** The
  target exposes `attacker_instruction` (re-substituted into the poisoned tool
  observation via the case's `Tool Response Template`, scoped to `external_data`)
  and `user_instruction` (scoped to `user`). Un-injected, the shipped benchmark
  injection runs — so a passthrough optimizer reproduces InjecAgent exactly.

## Fidelity-preserving deviations

- **The vendored simulated-response cache is read-only.** Upstream writes newly
  generated responses back into `attacker_simulated_responses.json`; this target
  never mutates the vendored file (that would break byte-faithfulness). A genuine
  cache miss is computed via the LLM and memoized **in-process only**.
- **ds step-2 simulation LLM fallback.** On a cache miss, the exfil-tool response
  is generated with `sim_model` (default: `gpt-4-0613`, upstream's simulator
  model — override it, e.g. to reuse the agent's model when OpenAI is
  unavailable) using the vendored generation prompt. This is *environment*
  simulation, permitted by Simon's rule; the judge stays pure. `extract_content`
  indexing is guarded so a fenceless completion degrades to the raw message
  instead of raising (upstream indexes `[0]` unconditionally). The vendored cache
  (2,347 entries) covers the shipped corpus, so this fires only on a genuine miss.
- **`tools.json` provenance.** These schemas are adapted by InjecAgent from
  ToolEmu (Apache-2.0). Attribution is in `NOTICE` and `LICENSES/`.

## Not ported (with reason)

- `src/models.py` provider backends + `src/prompts/prompt_template.py` (see
  litellm note above).
- The two dataset-*generation* prompts in `generation_prompts.py`
  (`ATTACKER_INSTRUCTION_GEN_HUMAN_MESSAGE`, `USER_INSTRUCTION_GEN_HUMAN_MESSAGE`):
  they regenerate the corpus, which we ship pre-generated and vendored.
- `--use_cache` agent-output resume: the Controller owns run orchestration, so
  each task runs fresh (no resume-cache trap).
