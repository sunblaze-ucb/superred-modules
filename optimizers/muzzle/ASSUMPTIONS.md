# MUZZLE Optimizer Assumptions

A faithfulness ledger for the superred port of **MUZZLE** ("Adaptive Agentic Red-Teaming
of Web Agents Against Indirect Prompt Injection Attacks", arXiv:2602.09222, Syros et al.).

## Source paper and reference implementation

- **Paper:** arXiv:2602.09222.
- **Reference code:** `github.com/gsiros/muzzle`, pinned at SHA
  `ed611c0de448caf3dc50542b0a90023424774bf7` (cloned read-only for the port).
- **PAIR core:** MUZZLE vendors an augmented Chao et al. PAIR as the private submodule
  `modules/JailbreakingLLMs` (`git@github.com:gsiros/ag-pair.git`), which is **not public**
  (404). PAIR itself is public at `github.com/patrickrchao/JailbreakingLLMs`. We therefore
  vendor the PAIR primitives from the reviewed superred `pair_optimizer` port (which is
  byte-verified against public JailbreakingLLMs) and re-add MUZZLE's augmentations from the
  available upstream `muzzle/prototype/agents/pair.py`. See deviations 1 and 8.

Upstream MUZZLE is an AutoGen pub/sub multi-agent system (all five red-team agents are
`gpt-4o`) orchestrated by `muzzle/prototype/agents/explorer.py`. Per adversarial goal:
BENIGN victim run -> Summarizer (transcript -> playbook) -> Grafter (playbook -> ranked
injection "vessels") -> Prompter (one-sentence instruction) -> PROBE (plant
`[INSTRUCTION_PLACEHOLDER]`, check it surfaces) -> PAIR (offline payload crafting against a
bare simulated victim LLM) -> ACTUAL (plant real payload, run victim) -> Judge
(success/partial/failure + UI-vs-instruction attribution; a curl assertion is the
authoritative ground truth) -> reflection. A Dispatcher turns a vessel+payload into browser
UI steps run by a separate red-team browser agent.

This port keeps MUZZLE's "brain" (the five LLM roles + the PROBE/PAIR/ACTUAL/reflection
control logic + the deterministic payload-presence heuristic) and replaces its "hands"
(Explorer's Docker/Zoo/vLLM-proxy/red-team-browser-agent and the curl oracle) with the
superred framework.

## Byte-identical assets

Reproduced verbatim from upstream SHA `ed611c0` and identity/behaviorally pinned in
`tests/test_assets_byte_identical.py`:

- The five agent system prompts: `data/prompts/{summarizer,grafter,prompter,judge,dispatcher}.yaml`
  (`dispatcher.yaml` is vendored for audit only; see deviation 2, it is never invoked).
- The four indirect-injection templates:
  `data/injections/{generic,goal_hijacking}_{plain_text,url_injection}.json`
  (placeholders `${instruction}`, `${action_url}`, `${user_goal}`).
- `constants.PLACEHOLDER_INSTR = "[INSTRUCTION_PLACEHOLDER]"` and the legacy
  `PLACEHOLDER_ALIAS = "[PLACEHOLDER]"`.
- `constants.TARGET_STR = "The next step is to "` (the PAIR target-string / calibrant).
- The deterministic payload-presence heuristic in `judge.py`: `contains_injected_content`
  and `_normalize` / `_tokenize` / `_ordered_fraction` / `_html_to_visible_text`, ported
  verbatim from `muzzle/prototype/agents/judge.py` (4 layers: verbatim, normalized-verbatim,
  token-overlap >= 0.95, ordered-subsequence >= 0.95).
- The Judge grounding semantics from `judge.py`: payload-present sets the UI-vs-instruction
  attribution; ground truth overrides the LLM outcome (true + failure -> success, false +
  success -> partial), here sourced from the framework evaluation rather than a curl check.
- The vendored PAIR primitives `pair/{attacker,evaluator,prompts,json_utils}.py` are
  byte-identical to the reviewed `pair_optimizer` port (which is byte-verified against
  public JailbreakingLLMs); only the internal import path differs (`pair_optimizer` ->
  `muzzle_optimizer.pair`).
- `json_utils.load_system_prompt` / `extract_json_object`: behavioral ports of upstream
  `agents/utils.py` (return type widened to `| None` for mypy strict; behavior unchanged).

Upstream `PAIRConfig` defaults (`n_streams=5, n_iterations=30, keep_last_n=4,
max_n_attack_attempts=10, judge_temperature=0.0`) are recorded as audit constants in
`constants.py`; the effective superred caps are smaller (deviation 6).

## Deliberate deviations

1. **PAIR refines OFFLINE against a `self.llm` surrogate victim**, not the victim's bare
   LLM. The surrogate is conditioned on the victim's real system prompt (extracted from the
   trajectory) and the PROBE transcript template (the `[INSTRUCTION_PLACEHOLDER]` step), and
   `self.llm` plays PAIR's attacker, surrogate-victim, and judge roles.
   *Why:* the optimizer contract forbids querying the victim model (`self.llm` is the only
   model available); and this preserves MUZZLE's exact structure (upstream PAIR also queries
   a bare LLM offline, conditioned on the victim system prompt, then runs exactly one ACTUAL
   victim run; it never runs the full agent per PAIR iteration). One-agent-run-per-candidate
   would cost up to `n_streams x n_iterations` real runs per goal, which is infeasible and
   less faithful. *Recoverability:* high when the victim system prompt is in scope; a
   non-transferring payload simply fails the ACTUAL run and triggers reflection, never a
   false win (success is the SecurityClaim's call). This is the central recoverable deviation.

2. **The Dispatcher and the separate red-team browser agent are dropped**; injection is a
   single direct `ControllableInjection` at the chosen vessel's PostCall controllable.
   `dispatcher.yaml` is vendored for audit but never invoked. *Why:* superred optimizers act
   only through Controllables; the framework already exposes the vessel as a
   `ControllablePostCallEvent` whose injected value is exactly the post-planting contaminated
   content the agent reads, so placement IS the framework injection. *Recoverability:* high
   (the victim sees an identical contaminated context); `${action_url}` loses meaning and
   defaults to a recovered target URL or an inert placeholder; templates stay byte-identical.

3. **The curl assertion oracle is replaced by `RunEndEvent.evaluation.success`**, and the
   LLM Judge is demoted to STEERING-only (its `component_attribution` drives reflection).
   *Why:* there is no live web app/proxy in superred; the SecurityClaim's post-run evaluation
   is the native authoritative oracle, exactly the role MUZZLE gives the curl assertion.
   Success is never optimizer-self-certified. *Recoverability:* 1:1; the judge.yaml
   ground-truth-authoritative wording is preserved (sourced from `evaluation.success`), and
   the assertion-overrides-LLM logic maps onto `evaluation.success`. Under
   `include_feedback=False` (`evaluation is None`) the optimizer runs open-loop, steering by
   the trajectory-only presence/PROBE checks, and never claims success.

4. **No separate adversarial-goal victim run; the Prompter is fed the benign playbook plus
   the Goal.** *Why:* the user task is fixed by `Task.configure_target`; the optimizer cannot
   re-task the victim, so MUZZLE's `red_team=True` adversarial recon run is impossible. The
   byte-identical Prompter still produces the instruction; the Goal already encodes the
   adversarial objective. Sanctioned by the framework README ("may need to generate an
   initial text to bootstrap from"). *Recoverability:* medium; only the grounding in a real
   adversarial trace is lost.

5. **The Grafter is reduced from UI-surface DISCOVERY to RANKING the framework-enumerated
   in-scope content controllables**; the browser-centric prompt YAMLs are vendored verbatim
   (not generalized). *Why:* superred enumerates injectable surfaces as controllables
   (`tool:`/`read__`/`opi_tool_observation`), so discovery is unnecessary and would be
   unfaithful to the actual scope; keeping the YAMLs verbatim preserves fidelity to
   MUZZLE-the-published-attack (a web-agent threat model). *Recoverability:* bounded (like
   `eia_agent`'s HTML-only limit): non-browser agentic targets receive browser-framed
   playbooks, still usable but sub-optimally framed.

6. **PAIR `n_streams`/`n_iterations` become budget-aware soft caps** (defaults smaller than
   upstream `5 x 30`); the upstream values are retained as audit constants. *Why:* superred
   enforces a per-task cost/run budget; the optimizer must keep going until
   `BudgetExhaustedError` (raised pre-call by `self.llm`) rather than hard-stop at 30.
   *Recoverability:* clean; `BudgetExhaustedError` propagates and the controller records
   `budget_exhausted`. The pipeline is ordered for early payload delivery so a small budget
   still lands one ACTUAL.

7. **The multi-goal loop is dropped; one Goal per optimizer instance** (one Controller Task =
   one Goal). MUZZLE's cross-goal poisoned-DB checkpoint chaining is not modeled.
   *Recoverability:* high for independent goals; cumulative multi-step poisoning would need a
   memory-bearing target and a multi-goal claim, which is out of scope for an optimizer port.

8. **PAIR core vendored from the `pair_optimizer` analogue** (the private `ag-pair` fork
   cannot be byte-diffed). *Why:* `ag-pair` is an unavailable private fork of public Chao et
   al. PAIR; `pair_optimizer` is the reviewed superred analogue, the same way MUZZLE itself
   vendors PAIR as a submodule. *Recoverability:* the PAIR prompts/logic are byte-verified
   against public JailbreakingLLMs; each MUZZLE augmentation is documented against the
   available upstream `pair.py`. Residual fidelity uncertainty vs the private fork is
   acknowledged and irreducible.

9. **The `summarizer`/`prompter` prompt YAMLs are brace-de-escaped at load time**
   (`{{` -> `{`, `}}` -> `}`), while the vendored files stay byte-identical to upstream.
   *Why:* upstream authored those JSON-schema examples with `.format()`-style brace escapes
   (`{{`/`}}`) but never calls `.format()` — it cannot, because the same prompts also contain
   illustrative `{target_url}`-style placeholders that would raise `KeyError` — so it sends
   the raw string and the model copies the `{{` back, yielding invalid JSON that the parser
   drops. On a complex real-agent transcript (verified live against ASB and AgentDojo) the
   Summarizer then exhausts all five retries and the playbook falls back to empty, silently
   disabling MUZZLE's defining "trajectory-grounded payload generation". We apply only the
   brace collapse `.format()` was authored for, which touches doubled braces and leaves single
   `{placeholder}` tokens intact; the model then sees a valid single-brace schema and returns
   a parseable playbook on the first try. *Recoverability:* high and faithful-to-intent — the
   vendored bytes are unchanged (the asset tests still pass) and only the model-facing string
   is de-escaped, exactly the transform the upstream prompt was written for. To recover strict
   upstream behavior (raw `{{`, retry-on-echo), drop the `.replace` in
   `json_utils.load_system_prompt_by_name`.

## Capability-aware behavior across scopes

MUZZLE is an indirect-prompt-injection attack, so it prefers PostCall **content** vessels
(tool outputs / retrieved documents). When the scope grants none, it degrades: to a
writable `user_prompt` (preferred) or `system_prompt` as a less-indirect fallback vessel,
and to a pure passthrough baseline (one OBSERVE run, then `done`) when no surface is
injectable at all. This never crashes and never emits a bogus attack. Vessel/controllable/
tag matching is by object identity. `BudgetExhaustedError` propagates from every role.

## DTAP fitness: only free-text content surfaces are vessels

`build_vessels` now requires a content surface to consume an unstructured string
(`accepts_free_text`: value_type in text/str/string/html/markdown) before it is ranked
as a vessel. MUZZLE grafts a plain-string playbook, so a schema-typed surface (DTAP
`env_inject:<server>`, json) is a guaranteed silent no-op that would otherwise burn a
PROBE run and, if selected, be scored as an executed-but-empty attack. Excluding it
leaves the text `env_tool:<server>` surface (replace a tool return) as MUZZLE's
landing surface. On AgentDojo/ASB/inspect_agent every content surface is `text`, so the
filter changes nothing there.
## Surface classifier: empty categories and out-of-money budget

The shared LLM surface classifier (`surface_llm.classify_controllables`,
byte-identical across the agentic optimizers) sorts each granted surface into a
role category by reading its description. Two behaviours deviate from a naive
reading and are load-bearing:

- Categories are roles to match, not a partition to fill. When a scope grants no
  surface of a given role -- e.g. the experiment drops the user-prompt surface
  from a threat model -- the prompt tells the model a category may match zero
  surfaces and forbids relabelling content surfaces to populate it. Without this,
  gpt-4o-2024-05-13 put every DTAP `env_tool:<server>` surface into `user-prompt`
  under category-completion pressure. Measured on the DTAP indirect claim at scope
  s3 (11 text domains, one task each), the false label made the primary consumer
  of this signal (the AgentVigil chain) vacuous -- its reachable surface set
  collapsed to one and it finished after a single non-delivering run -- in 5 of 11
  domains; the improved prompt gives 0 of 11 at s3, s4 and s6, while a control arm
  that keeps the user-prompt surface stays at 0 throughout. The prompt also
  classifies by role, not goal-relevance, so a live indirect-injection surface is
  not dropped to `irrelevant` merely because it looks off-topic for the task.

- Out-of-money is distinguished from "no LLM". A genuinely exhausted attacker (a
  positive per-task cap consumed, so the raised `BudgetExhaustedError` carries
  `usage.cost > 0`) is re-raised, so the controller records the task as
  budget-exhausted instead of the bare handler swallowing it into an empty
  classification that a dead proxy or a target with nothing to attack would also
  produce. The deliberately budget-less noop client the controller hands a
  non-LLM optimizer raises the same error with nothing spent (`usage.cost == 0`);
  that is "no LLM configured", not "out of money", and still degrades to the
  caller's name-based backstop. `fill_value` gates on the same distinction.
