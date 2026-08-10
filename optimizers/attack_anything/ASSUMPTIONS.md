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

2. **One attack attempt = one run (assembled-transcript judging).** superred's `Task`
   judges each **run's** transcript for goal achievement, and the chatbot target owns
   one accumulating conversation per run. So a decomposition attempt walks all its
   sub-tasks in sequence **within one run/conversation** (the shared-history protocol,
   `seats_feedback_decon.py`), letting the framework judge the assembled result. The
   upstream "separate / turn-fresh" protocols (which reset the conversation per
   sub-task/turn) cannot be reproduced against a target that owns its conversation;
   `turn_independent` / `recursive_leaf_attack` are accepted as config but their
   per-conversation-reset semantics are **not** fully honored under this mapping.
   *Why:* the target owns conversation state and the Task judges per run. *Recover:*
   documented; a future target that resets per sub-task could restore it.

3. **Sub-task judge internal; framework verdict authoritative.** Upstream uses a dual
   judge (sub-task-level AND goal-level consensus). superred's
   `RunEndEvent.evaluation.success` is the authoritative goal verdict, so only the
   vendored **sub-task** judge is kept, and it steers the within-run turn loop
   (which probe/feedback next, early-exit on affirmative). A framework success is
   reward 1.0 (the upstream consensus-goal-affirmative override). *Why:* the superred
   principle that the SecurityClaim owns success. *Recover:* the internal judge modes
   (`JUDGE_MODE` / `judge_mode`) are exposed and used verbatim.

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

## Staged (accepted as config, behavior not yet fully implemented)

These upstream refinements are wired as flags but their advanced behavior is a
follow-up; the default all-on method does not depend on them:

- **UCB wrapper selection** (`wrapper_selection="ucb"`): the port uses the randomly
  assigned per-sub-task wrappers (`assign_wrappers`) that the vendored `DeconGenerate`
  produces; the separate-engine's UCB wrapper pool is not yet driving selection.
- **Decomposition validator** (`validator_threshold` / `validator_max_retries`): the
  `DeconValidator` prompt is vendored but the regenerate-until-faithful gate is not yet
  wired into seeding.
- **Recursive-leaf re-decomposition** (`recursive_leaf_attack`) and per-turn-fresh
  (`turn_independent`) semantics — see deviation 2.
- **v2 frontier extension** (`goal_as_root`, `fallback_enabled`): the goal-as-root
  recursive-ternary tree and A/B/C fallback are not yet ported.
