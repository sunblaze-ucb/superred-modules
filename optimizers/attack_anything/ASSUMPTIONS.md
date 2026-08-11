# Attack Anything Optimizer Assumptions

A faithfulness ledger for the superred port of **Attack Anything: Recursive
Self-Evolving Attack Tree Search for Multi-Turn LLM Red-Teaming** (SEATS).

## Source and provenance

- **Reference implementation:** provided by the user as an anonymous, under-review
  research submission (paper author "Anonymous"; status "Under review"). It carries
  **no LICENSE file and no public repository or commit SHA**. The user asserts it is
  Apache-2.0; it is redistributed here under that license (`LICENSE`, `NOTICE`,
  `LICENSES/AttackAnything-Apache-2.0.txt`).
- Because there is no upstream commit to pin, faithfulness is anchored to a
  **SHA-256 of each vendored file**, recorded in `constants.VENDORED_SHA256` and
  checked by `tests/test_assets_byte_identical.py`.

## What Attack Anything is

An automated multi-turn chatbot jailbreak. Its headline method layers four
building blocks: **decomposition** (a helper LLM splits the harmful goal into ~4
innocuous technical sub-tasks), **feedback** (a short multi-turn conversation per
sub-task that reads each refusal and rewrites the next message, PAIR/TAP-style),
**tree search** (candidates form a UCT tree grown by WizardLM depth/breadth/
crossover operators), and a **cross-goal elite archive** (winning attacks transfer
between goals). The upstream ships this plus an ablation ladder of the same engine
with blocks removed, plus a v2 frontier extension.

## Byte-identical assets

The entire upstream SEATS engine is vendored verbatim under
`src/attack_anything_optimizer/_vendor/` and pinned by SHA-256:

- `tree.py` — `AttackNode`, `AttackTree` (UCT selection / backprop).
- `archive.py` — `EliteArchive`, `EliteEntry` (bounded, per-goal cap, weighted sampling).
- `operators.py` — `RULE_WRAPPERS` (6 wrappers), depth/breadth/crossover operators,
  `PROBE_STRATEGIES` / `get_probe`, personas — every prompt/template string.
- `utils.py` — the three response judges (`llm_classify_reply` permissive/strict/
  calibrated + the `_REFUSAL_PREFIX_PHRASES` pre-filter), `classify_reply`,
  `AFFIRM/REFUSAL_PATTERNS`, `Turn`, `normalize_ws`, and the (unused) urllib client.
- `seats.py` — `SEATSConfig` (all defaults), `generate_seed_prompts`, the reward math.
- `seats_decon.py` — `DeconNode`, the decomposition prompt (`_DECOMPOSE_SYS`),
  `_PERSONA_LIST`, `_AVOID_HINT_TPL`, `decompose_goal`, `select_best_decomposition`,
  `assign_wrappers`, `wrap_subtask`, and the five decon operators.
- `seats_feedback.py` — `FeedbackGuidedRefine`, `FeedbackGuidedContinue` (their exact
  system prompts).
- `seats_feedback_decon.py`, `seats_feedback_decon_separate.py` — the shared-history
  and separate/turn-fresh main-method engines (`DeconValidator`, the validator prompt,
  the recursive-leaf logic), vendored for audit and reuse.

No attacker-owned asset is loaded from disk: every seed/wrapper/persona/probe/judge
bank is a Python constant in the vendored code, so it ports with the source. The
three benchmark goal CSVs are **not** vendored (they are a SecurityClaim/Task concern).

## Deliberate deviations (each: what / why / recoverability)

1. **Inverted target loop.** Upstream `AttackEvaluator.evaluate()` /
   `DeconEvaluator.evaluate()` own the victim
   (`for turn: target_client.chat(transcript)`). superred inverts control: the
   target loops and calls the optimizer via `ControllablePreCallEvent`, and the
   optimizer answers with `ControllableInjection`. So the evaluator loops become the
   `on_event` state machine in `optimizer.py`. *Why:* the framework contract. *Recover:*
   the vendored evaluators are still present in `_vendor/` for reference.

2. **Fresh-conversation = fresh run; the search is a generator (planner) + pump.**
   The upstream control flow (`_run_goal` / `evaluate` / `_attack_task_recursively` /
   the v2 `_run_goal` + `_fallback_*`) is re-expressed in `planner.py` as a **Python
   generator** where each `response = target_client.chat(msgs)` becomes
   `reply = yield Unit(msg)`. `optimizer.py` is the **pump**: it runs each yielded
   `Unit` as one superred conversation. A `Unit.fresh` message starts a NEW
   conversation, which is a fresh superred **run** (the chatbot target resets between
   runs); a non-fresh message continues the current run. This maps the paper's
   fresh-per-sub-task / turn-fresh / recursive-leaf / goal-as-root protocols onto runs
   faithfully (a refusal on one sub-task does not poison the next). *Why:* it preserves
   the upstream algorithm structurally (copy-code faithfulness) while every victim
   interaction still flows through the framework's events/scope/trajectory. *Recover:*
   the vendored engine is present in `_vendor/` for reference.

3. **RDRT-lineage assembly run makes the attack framework-judgeable.** superred's `Task`
   judges each **run's** transcript, but a decomposition attack elicits the goal's
   pieces across separate innocuous conversations, so no single sub-task run contains
   the assembled harm. After the sub-tasks are elicited the planner emits one
   **assembly** `Unit` (prompt copied from `rdrt/deconstruct_multi_step_v6_multi_turn.py:856-902`,
   which the SEATS engine descends from) that asks the victim to synthesize the verified
   answers into the goal; the SecurityClaim judges that run, so
   `TaskResult.success` = the framework verdict on the assembly run (or on any run the
   judge already scores as goal-achieving). The engine variant omits this call (it relies
   on its internal all-subtasks judge); adding it is the faithful way to reconcile with
   superred's per-run verdict. The planner still runs the upstream **internal dual-judge**
   (sub-task + goal-level `llm_classify_reply`, consensus → reward 1.0 + early-stop) to
   drive the search; the framework verdict is layered on top as the authoritative
   reported success. *Why:* the superred principle that the SecurityClaim owns success,
   without weakening the attack to the shared-history variant. *Recover:* the internal
   judge modes (`JUDGE_MODE` / `judge_mode`) are exposed and used verbatim; the assembly
   is a small, isolated addition.

4. **Budget = controller cost/time cap.** Upstream's `max_target_queries_per_goal` is
   still exposed, but the real bound is the controller's `task_cost_cap_usd` /
   `task_time_cap_s`. `BudgetExhaustedError` propagates out of `on_event`; the internal
   judge/expansion degrade on transient failures. *Why:* superred fixes the attacker's
   compute per experiment. *Recover:* the soft knob remains.

5. **Sync -> async LLM transport, temperature dropped.** The vendored code calls a
   synchronous `client.chat(...)`; superred's `self.llm.complete` is async, and its
   model/credentials are locked. `VendorLLMBridge` bridges each vendored call onto the
   event loop from a worker thread (`asyncio.to_thread` + `run_coroutine_threadsafe`),
   so the vendored files run **byte-identical**. Sampling temperature is never sent to
   the provider (superred policy; see `tests/test_no_temperature.py`), so upstream's
   temperature knobs (seed/path diversity by rising temperature) are **inert** —
   diversity comes from independent resampling and the RNG instead. *Why:* framework
   LLM contract + the no-temperature invariant. *Recover:* the upstream temperature
   values still ride inside the vendored calls and are dropped only at the boundary.

6. **No urllib client / no vllm.** The attack core is stdlib-only; the vendored urllib
   `LLMClient` is dead code (never instantiated). The upstream `vllm` requirement is a
   local-model-serving concern, not an attack dependency; this package depends only on
   `superred`.

7. **Component toggles added.** Upstream ships the ablation variants as separate
   scripts/subclasses; this port exposes them as constructor toggles
   (`use_decomposition` / `use_feedback` / `use_tree_search` / `use_archive`), all on
   by default (= the headline method). *Why:* the user asked for one optimizer with
   switchable components. *Recover:* all-on reproduces the main method; each-off
   reproduces the corresponding ablation path.

8. **Out of scope.** `baselines.py` (PAIR/TAP/AutoRedTeamer/raw baselines) and
   `autoredteamer.py` are re-implementations of *other* attacks that already exist as
   their own superred modules (`pair`, `tap`, ...); they are not vendored.

9. **v2 frontier folded in; two utils fixes subsumed by the bridge.** The vendored
   `seats_feedback_decon_separate.py` and `utils.py` are the **v2** copies (byte-identical
   supersets of v1); `goal_as_root` and `fallback_enabled` are wired through the planner.
   The v2 `utils.py` fixes (skip `temperature` for Claude-4.7, coalesce `content=null`→`""`)
   are already subsumed by `VendorLLMBridge` (it never sends temperature and coalesces
   `content or ""`), so they are present for faithfulness but functionally inert here.

10. **Judge / validator endpoint decoupling.** Upstream can point the judge and the
    decomposition validator at a stronger model than the attacker (`--judge_*` /
    `--validator_*`). `judge_llm_config` / `validator_llm_config` expose this: when set,
    the optimizer builds separate `LLMClient`s. Like the target's own inference these are
    **out of** the controller's attacker cost cap. A `None` validator config leaves the
    validator gate off (accept the first decomposition), the default.

## Feature parity (all wired; copied from the vendored engine)

The following upstream behaviors are implemented in `planner.py`, copying the vendored
logic (UCB math, reward formulas, recursion, fallback dispatch) with the target call
replaced by a `yield`, and are exercised by `tests/test_planner.py`:

- **UCB wrapper selection** (`wrapper_selection` ∈ {ucb, priority, random}, `ucb_c`,
  `ucb_min_uses`, `wrapper_priority`, `SEATS_WRAPPER_SELECTION`/`_PRIORITY` env).
- **Decomposition validator gate** (`DeconValidator`, `validator_threshold`,
  `validator_max_retries`) — active when a validator endpoint is configured.
- **Dual-judge consensus** (sub-task + goal-level, AND-gate → reward 1.0 + early-stop).
- **turn_independent**, **recursive_leaf_attack** (with re-decompose-on-refusal),
  **goal_as_root** (v2 ternary tree), and the **A/B/C fallback** (v2).
- The strict-all-subtasks reward (`require_all_subtasks`) and both reward formulas.

**One deliberately dropped upstream extra:** the v2 Phase-3 *budget-exhausting retry
loop* (`_retry_main_tree_fresh_decomp`, which re-decomposes at rising temperatures after
A/B/C also fail) is not reproduced — under the no-temperature policy each retry round
would be an identical draw, so it adds no signal; the controller's cost/time cap is the
real budget bound. Documented here rather than silently omitted.

## Config knob deliberately omitted

- **`seed_temperature`** — superred never sends a sampling temperature to the provider
  (`tests/test_no_temperature.py`), so the upstream temperature knobs are inert here.
  Exposing this field would only be a name that trips the guard for no effect, so it is
  omitted (all other upstream knobs are present; see `tests/test_config_parity.py`).
