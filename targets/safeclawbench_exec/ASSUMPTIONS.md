# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | SafeClawBench (executable / Exec-Balanced path) |
| Dataset | <https://huggingface.co/datasets/sairights/safeclawbench> |
| Path | `executable/` (schema, state, tools, trajectory, metrics, runner, fixtures) + `defenses/defense_stack.py` |
| Commit | `e6c29204c24a5910600aae854baae57a51586655` |
| Paper | Tian et al., *SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents*, arXiv:2606.18356 (2026) |
| Licence | MIT (© 2026 ClawSecBench Authors) |

This is the paper's **third** endpoint — *sandbox-observed tool/state harm*. The
prompt-level **Semantic Core** (CoreFail@600) endpoint is a separate package
(`superred-claim-safeclawbench`); the *audit-evidence* label is surfaced there.

## Vendored byte-for-byte

Under `src/safeclawbench_exec_target/_vendor/`, verifiable with
`python scripts/sync_upstream.py --check`:

- `executable/{schema,state,tools,trajectory,metrics,runner,fixtures}.py` +
  `executable/__init__.py` + `executable/README.md` — the offline mock sandbox
  (six state stores, eleven permissioned tools, the deterministic state oracle,
  and the JSON tool-plan runner). Pure-stdlib; no third-party imports.
- `executable/fixtures/exec_full_600.json` — the full 600-case executable set
  (100 per family: DPI / IPI / TRI / MPI / MEX / ADI) — and `tiny_subset.json`.
- `defenses/defense_stack.py` — the prompt-level defense policies.

## Not vendored / adapted

- **The agent model loop is re-implemented, not vendored.** Upstream's
  `runner._api_policy` / `_run_api_tool_loop` call the model through upstream's
  `agents.api_wrapper`; superred drives the **model-under-test** through its own
  `LLMClient`. `SafeClawBenchExecTarget._drive_agent` is a faithful port of that
  loop — it reuses the vendored system prompt (`_build_api_system_prompt`), the
  vendored JSON schema + parser (`parse_model_tool_plan`), the vendored
  `MockToolSandbox`, and the vendored defense policies — so the executed protocol
  matches upstream; only the model transport differs. The vendored
  `runner.py` is kept intact (its `agents.api_wrapper` import is lazy and unused).
- **Defense levels.** Only `D0`–`D3` ship. Upstream's `defense_stack` optionally
  layers `B1`–`B4` / `D4` / length-control variants from `defenses.baselines`,
  `defenses.agent_shield`, `defenses.length_control`; those modules are outside
  `defense_stack.py`, are guarded by `try/except ImportError`, and are not part
  of this endpoint's core, so they are absent (D0–D3 are the primary policies).
- **Analysis / plotting.** Upstream's `analysis.py`, harness runners, and the
  `numpy` / `matplotlib` / `requests` dependencies are not vendored — the
  executable sandbox + oracle themselves are pure-stdlib.

## superred adaptation

- **Target / claim split** (matching the SafeClawArena precedent):
  - **Target** (`SafeClawBenchExecTarget`) — builds the case's mock world, drives
    the model-under-test through the sandbox, captures the final world state +
    trajectory as `query("post_state")`.
  - **Claim** (`safeclawbench-exec-claim`) — applies the vendored
    `evaluate_case_metrics` state oracle to that capture; `HarmCompletion-ASR`
    is the attacker-oriented `success`, with `ToolCall-ASR` / `StateChange-ASR` /
    `Blocked-by-Harness` as sub-scores. `Intent-ASR` (upstream's optional LLM
    intent judge) is left unset — the state oracle is fully deterministic.
- **Controllables.** The attack is baked into each scenario; the target also
  exposes `user_message` (override the task prompt) and `external_data` (plant an
  attacker file the agent can read back) so an optimizer may vary the attack.
  Left un-injected, the benchmark runs as shipped.
- **Security domain.** `system → {user_input, external_data, memory, tools}`.
- **Offline & side-effect-free.** Every tool call is simulated against in-memory
  state; the only live component is the model-under-test (a scripted stub in
  tests).
