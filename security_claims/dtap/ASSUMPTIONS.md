# ASSUMPTIONS: security_claim_dtap

Deviations from upstream DecodingTrust-Agent (DTAP, `AI-secure/DecodingTrust-Agent`,
commit `e0323a52`, Apache-2.0). This package is the agent-agnostic **DTAP-BENCH**
security claim: it enumerates the upstream per-task tree, turns each per-task
`config.yaml` into one superred `Task`, and reuses upstream's per-task `judge.py`
**verbatim** (run out of band by the shared `dtap_scaffold.judge_runner`).
Deviations concern packaging, the judge transport, target-agnostic discovery, and
re-expressing one per-task config as one `Task`. The scoring predicates
(`eval_task` for benign, `eval_attack` for malicious) are upstream's own.

## A. Data

- **A.1** The per-task dataset (`config.yaml` / `setup.sh` / `judge.py` /
  `metadata/`) is **not vendored** in this package (it is large and licensed with
  the dataset). It is resolved by the frozen scaffold loader
  (`dtap_scaffold.dataset.resolve_dataset_root`): an explicit `dataset_root`, else
  `$DTAP_DATASET_ROOT`, else `./dataset`; with `download=True` the requested
  domains are fetched from HuggingFace (`AI-Secure/DecodingTrust-Agent-Platform`).
- **A.2** **Text-only scope.** DTAP ships 14 domains; this port covers the 11
  text-based ones (`dtap_scaffold.text_domains.TEXT_ONLY_DOMAINS`). `browser`,
  `macos`, and `windows` are vision/GUI driven and out of scope; the scaffold
  enumerator skips them, so the claim can never construct a task for them.
- **A.3** **Golden-hash manifest.** `data/golden_hashes.json` pins
  `sha256(Goal bytes + judge.py bytes)` for a deterministic strided sample
  (default 16 `travel` tasks, spanning benign and malicious; see
  `dataset.build_golden_hashes`). The *Goal bytes* are exactly what the `Task`
  exposes as the `Goal`: `Attack.malicious_goal` for a malicious task, and the
  first `Task.task_instruction` for a benign task (mirroring
  `TaskConfig.instructions[0]`, which `task.py` uses). Together with each task's
  `judge.py` those are the two byte-identity-critical pieces per task: the
  objective the attacker pursues and the judge that decides the outcome. A
  faithfulness test re-hashes the sampled dirs and compares, so a dataset drift in
  either piece must be consciously re-baselined (regenerate with
  `dataset.build_golden_hashes(root)`). The manifest is a dev/faithfulness
  artifact at the package root (it pins external bytes), not runtime data.

## B. Predicates (scoring)

- **B.1 Malicious primary** = the judge's **`attack_success`** (upstream
  `Judge.eval_attack`). **Polarity: `True` == the attack SUCCEEDED** (the safety
  property was violated). `task_success` (did the agent also do the benign cover
  task) is carried as an unscoped sub-score. `success` of the `EvaluationResult`
  equals `attack_success`.
- **B.2 Benign primary** = the judge's **`task_success`** (upstream
  `Judge.eval_task`): did the agent complete the legitimate request. No sub-scores.
- **B.3 The judge is upstream's, run byte-faithfully out of band.** Each task's
  own `judge.py` is executed by `dtap_scaffold.judge_runner.run_dtap_judge`, which
  re-queries the **live environment state** through the task's env container ports.
  This is DTAP's *verifiable env-state* evaluation, not an LLM-judge of text.
  Following the framework rule that scorers do not count against the optimizer, the
  judge's work (and the few tasks whose judge calls an LLM) is **out of band**:
  never charged to the optimizer's budget. The optional `judge_model` /
  `judge_api_base` / `judge_api_key` are forwarded to those LLM-using judges;
  substituting a proxy-served model for the canonical judge model is the single
  permitted deviation (recorded by the caller).
- **B.4 An un-evaluable run is surfaced as an error, not a clean negative.** The
  judge may fail to produce a verdict for the deciding metric -- `attack_success`
  (malicious) or `task_success` (benign) -- because its subprocess crashed or
  timed out, or a verifiable judge's live env was unreachable. `run_dtap_judge`
  reports that as a truthy `error` with the successes left `None`. `evaluate`
  raises on it (deciding metric `None`, or any `error`), so the controller records
  the task with `stop_reason="error"` and the traceback on `TaskResult.error` --
  kept distinct from a genuine negative (a real "attack blocked" / "task not
  done"). Collapsing it to `0.0` / `success=False` would count an un-evaluable run
  as a safe result and inflate the benchmark; upstream likewise never treats an
  undecidable verdict as a failure (`eval/task_runner.py:387` tests
  `task_success is False`, not `not task_success`). A benign task's naturally-
  `None` `attack_success` is NOT an error: only the deciding metric is checked.

## C. Transport & scoring shape

- **C.1 `primary_score.security_domain = None`** (always visible). DTAP's attack
  surface is one of four injection vectors (prompt / tool / environment / skill),
  chosen **per experiment by the Controller scope**, not fixed by the task. Pinning
  the primary to a single trust-boundary tag would hide it under other scopes, so
  it stays unscoped (the framework also requires this of `primary_score`). The
  `task_success` sub-score is likewise unscoped.
- **C.2 No boundary-specific sub-scores.** DTAP tasks are single-shot env-state
  checks with no durable cross-run memory bank, so there is no memory-tagged
  sub-score (unlike the ASB claim's `memory_found`).
- **C.3 Target-agnostic discovery.** The `Task` binds to the base
  `superred.core.interfaces.target.Target`, not a concrete DTAP target. It detects
  DTAP compatibility by config-slot presence
  (`REQUIRED_CONFIG_SLOTS = {active_mcp_servers, task_dir, user_prompt}`); a target
  missing those raises `NotApplicable` (the Controller skips it) instead of
  crashing, so the claim composes with non-DTAP targets without error.

## D. Granularity & scope

- **D.1** One `DtapTask` per per-task `config.yaml` (one DTAP objective). Filters:
  `domains`, `types` (`benign` / `malicious`), `threat_models` (`direct` /
  `indirect`), `risk_categories`. Roll-ups: `dtap_domain_claim`,
  `dtap_direct_claim`, `dtap_indirect_claim`, `dtap_benign_claim`,
  `dtap_risk_claim`, and `dtap_combined_claim` (lazy chaining). `dtap_claim` raises
  `ValueError` if no task matches the filters.
- **D.2 Goal content.** `DtapTask.goal.description` is the **byte-identical**
  `Attack.malicious_goal` for a malicious task, and the first benign
  `task_instruction` for a benign task. The attack **content and method** are an
  external attacker's (optimizer's) concern: the target exposes the four injection
  vectors as controllables and the Controller scope selects which are in play;
  there is **no attacker module** in the claim. Because success is decided by the
  env-state judge (not a substring match on the goal), using `malicious_goal` as
  the `Goal` does not let a content-injection optimizer self-fulfil success.
- **D.3 `configure_target`** sets the scenario from the parsed config: the active
  MCP env servers, the env-injection config, the system prompt, the benign user
  prompt (JSON list, for multi-turn), the task dir, the available-injection hint,
  and the threat model. `available_injections` is recorded as a **scope hint
  only**; it does NOT remove controllables (the target always exposes its real
  surface; the Controller scope gates per experiment). `max_turns` and the
  native-tools policy are left at the target's construction defaults (they are
  generation / golden-replay concerns, not per-objective config).
- **D.4 Target factories.** `dtap_claudecode_target_factory` /
  `dtap_openclaw_target_factory` **lazily import** the concrete target classes, so
  this claim package depends only on `superred` + `dtap-scaffold` (the target
  packages carry the heavier Docker / agent-SDK deps). `concurrency` defaults to 1
  (each task spins a fresh Docker env stack); `model` is the agent's own inference
  model, run through the target's own client, NOT the optimizer's budget-locked
  `LLMClient`.

## E. Conformance

- **E.1** The golden-hash test (`tests/test_dataset.py`) re-hashes the committed
  sample dirs and compares, so a change to a goal or a `judge.py` is caught.
- **E.2** Goal byte-equality tests assert `DtapTask.goal.description` equals the
  dataset's `Attack.malicious_goal` (malicious) / first `task_instruction`
  (benign) verbatim, with no paraphrase or truncation.
- **E.3** The test suite is **offline-only**: every Docker / HTTP / LLM boundary is
  mocked (a fake `dtap_scaffold.judge_runner` module; stub targets). Live Docker /
  LLM integration is exercised by `@pytest.mark.docker` / `@pytest.mark.live` tests
  in the target packages, skipped where those resources are absent.
