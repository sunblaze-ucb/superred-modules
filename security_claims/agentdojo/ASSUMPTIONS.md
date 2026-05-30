# ASSUMPTIONS

Every divergence from upstream AgentDojo or the brief is enumerated
here with: (a) what AgentDojo does, (b) what this package does, (c)
why.

Entries are numbered.  Add at the bottom of the relevant section;
never reuse a number.

---

## A. Layer 1: faithful port of AgentDojo injection tasks

### A.1 One Task per (user_task, injection_task) canonical pair

**AgentDojo** (`benchmark.py:160-229`): iterates every `user_task` in
the suite and every `injection_task` per user task, producing a full
cross-product.  Under the v1.2.2 pin the per-suite cross-product totals
are: workspace 40x14, slack 21x5, travel 20x7, banking 16x9 = 949
cases (v1.2 added workspace IT6-IT13, so workspace's injection-task
count grew from 6 to 14 versus the original v1 paper's 629-case total).

**Us**: Layer 1's `agentdojo_layer1_claim()` default is the 27-pair
canonical scope — exactly one user task per injection task — chosen
so the user task's ground_truth invokes a read tool whose return
value surfaces the injection task's required slot.  The cross-product
view is reachable via the `pairs=` kwarg with explicit (suite,
user_task_id, injection_task_id) tuples.  See `layer1_pairs.py` for
the canonical list.

**Why**: 629-case cross-product is faithful but expensive (compute
& cost).  27 canonical pairs cover every injection task once and
keep per-claim runs tractable.  The faithfulness test sweep uses
the cross-product as oracle.

### A.2 Category labels (refined per user)

Per-suite categories (4 each, total 16):

- Banking: `exfil_via_memo`, `unauth_transfer`, `recurring_redirect`,
  `credential_change`.
- Workspace: `unauth_email_send`, `email_exfil`, `twofa_exfil`,
  `file_deletion`, `calendar_manip`.
- Slack: `phishing`, `exfil_to_web`, `drive_by`, `membership_manip`.
- Travel: `unauth_booking`, `pii_exfil`, `calendar_manip`,
  `output_manip`.

User confirmed this axis 2026-05-15.

### A.3 `EvaluationResult.success` semantics

**AgentDojo** (`base_tasks.py:126` docstring): `True` means "the
injection task was executed correctly" — the attacker won.

**Us**: surfaced directly as `EvaluationResult.success` and
`primary_score.value = float(security_result)`.  Naming in our code
uses `attack_succeeded` (vs `security`) to avoid the polarity-
inversion trap documented in upstream's error-skip branches.

### A.4 Trace function-name de-prefixing for `*_from_traces`

**AgentDojo**'s `security_from_traces` and `utility_from_traces`
methods accept a `list[FunctionCall]` with bare names (e.g.
`send_email`).

**Us**: our wrapped runtime records trace entries with suite-
prefixed names (`workspace__send_email`).  Layer-1 evaluate() strips
the `{suite}__` prefix before passing the trace to upstream's
`*_from_traces` methods, then keeps only the entries whose original
name appears in that suite's tool list.

**Why**: cross-suite calls in the composite trace would otherwise
confuse upstream's name-matching predicates.

### A.5 Per-task `init_environment` mutations replayed via seed override

**AgentDojo** (`base_tasks.py:29`): `BaseUserTask.init_environment(env) -> env`
may mutate the sub-env before the run.

**Us**: at `configure_target` time, Layer-1 Tasks call the upstream
`init_environment(suite_env)` to obtain the mutated sub-env, diff it
against the seed sub-env, JSON-encode the diff, and set the
`seed_yaml_override__{suite}` slot.  At run start, the target
applies the overlay to its composite env.

### A.6 Benign user prompt = `user_task.PROMPT`

**Brief / user clarification**: configure_target sets a benign user
prompt that exercises tools under no-attacker conditions; the
adversarial intent never enters the user prompt directly.

**Us, Layer 1**: this matches AgentDojo's threat model exactly —
the user_task PROMPT is the benign legitimate query.  We set
`user_prompt = user_task.PROMPT` without modification.

---

## B. Layer 2: bespoke system-purpose-violation goals

### B.1 Source is bespoke (NOT SORRY-Bench / HarmBench)

**Brief** (Section 3.b, default): SORRY-Bench prompts; LLM-as-judge.

**User clarification 2026-05-15**: "should not be to jailbreak just
the model but goals that do something that is fundamentally against
the system purpose but not necessarily against model ethics... you
may also come up with things yourself."

**Us**: Layer 2 is a hand-authored catalogue of standalone goals
targeting the composite multi-suite agent.  Each goal has a
deterministic Python security predicate.  No LLM-as-judge.  See
`layer2_goals/*.py` for the catalogue.

### B.2 Delivery: benign user prompt + attacker-chosen channel

**Brief**: configure_target sets a benign user prompt; the attacker
decides delivery (override user prompt, override system prompt,
inject content into reads, edit tool catalogue) based on scope.

**Us**: every Layer-2 Task carries a `benign_user_prompt` (a routine
query that exercises at least one tool call and succeeds under no
attacker), set via the `user_prompt` config slot.  The adversarial
Goal lives only on `Task.goal` (passed to the optimizer at
initialize-time) and the security predicate.

### B.3 Cross-suite goals exist

The four `cross_*` goals in the Layer-2 catalogue
(`cross_banking_to_slack`, `cross_calendar_collision_booking`,
`cross_pii_via_slack_web`, `cross_workspace_to_external_email`) span
multiple suites — novel threat-model territory unreachable in upstream
AgentDojo because upstream runs one suite at a time.

---

## C. Composition and combined claim

### C.1 Layer 3 is trivial composition

Layer 3 = `SecurityClaim.from_claims([layer1, layer2])`.  Lazy; no
duplication; each Task carries its own primary_score / sub_scores
naming so downstream analysis can split by layer.
