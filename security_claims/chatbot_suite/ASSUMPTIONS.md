# ASSUMPTIONS — chatbot_suite

This module is not a port of a single benchmark; it is a **combinator** over
three existing superred chatbot claims. It introduces no new prompts or
judges. The decisions below are the only ones it makes on top of the source
benchmarks.

## Sources (and what is excluded)

The three real chatbot harmful-instruction benchmarks in `superred-modules`:

- **HarmBench** (`secclaim_harmbench`) — 320 test behaviours across 3
  functional × 7 semantic categories; Llama-2-cls-prompt judge.
- **SORRY-Bench** (`security_claim_sorry_bench`) — 44 categories × 10 = 440
  prompts; `#thescore` LLM judge.
- **StrongREJECT** (`strongreject_claim`) — 313 prompts across 6 categories;
  rubric judge (continuous 0..1 score).

Deliberately **excluded**: `agentdojo` and `agentharm` are agentic claims
(they target tool-using agents, not a single-LLM chatbot) and are not
combinable here. `sorry_bench` (no trailing `security_claim_`) is a dead
leftover directory, not a real claim.

## Deduplication

- **Key**: `normalize_goal(task.goal.description)` =
  whitespace-collapsed, case-folded goal text.
- **Catches**: exact and trivial-variant duplicates (case, whitespace).
- **Does NOT catch**: semantic near-duplicates (paraphrases). Detecting those
  would need embeddings or an LLM, adding cost and nondeterminism, which
  defeats the point (avoid token waste, stay deterministic). In practice the
  three datasets are independently curated, so cross-benchmark exact overlap
  is ~0 at `per_category=1`; the dedup is a correctness guarantee, not a large
  reducer.
- **First-seen wins**: sources are processed in list order, so the first
  source to contribute a normalized prompt keeps it. `chatbot_suite_claim`
  orders sources **SORRY-Bench → HarmBench → StrongREJECT**.

## Per-category stratified subset

- `combine_claims(max_per_category=N)` keeps at most `N` tasks per category
  **key**, where the key is `"<attr>=<value>"` from the first populated of
  `semantic_category` (HarmBench), `category` (StrongREJECT), `category_name`
  (SORRY-Bench). Prefixing by attribute name namespaces categories so they
  never collide across benchmarks.
- This mirrors SORRY-Bench's existing `prompts_per_category` convention,
  generalised to the benchmarks that lack such a knob (HarmBench, StrongREJECT
  expose no per-category subsetting in their public factories).
- `chatbot_suite_claim` additionally passes `prompts_per_category=per_category`
  to SORRY-Bench so it doesn't materialise all 440 tasks only to discard them.
- At `per_category=1`: **57 tasks** (44 + 7 + 6). At `per_category=None`: the
  full benchmarks (large; intended for single-cell deep runs, not matrices).

## Judges and cost

- Each task keeps its **native** judge and `configure_target`; the combinator
  never re-judges. This is sound because the framework runs each task's own
  `evaluate` and never cross-references judges.
- **SORRY-Bench judge**: canonical `gpt-4-1106-preview` is not routable on the
  experiment proxy; default substitute `openai/gpt-4-turbo-2024-04-09` (GA
  snapshot of the same family). Matches the RQ1.1/RQ1.2 substitution.
- **HarmBench judge**: default `openai/gpt-4o-mini` (the module reuses the
  verbatim Llama-2-cls prompt through a chat endpoint).
- **StrongREJECT judge**: pinned to `openai/gpt-4o-mini` inside its own module
  (not configurable here).
- All judges run on their own clients (raw `litellm` calls or a separate
  `LLMClient`), so **judge cost is out of band** and never counts against an
  optimizer's per-task budget.

## StrongREJECT success threshold

StrongREJECT produces a continuous 0..1 score. Its own factory default
`success_threshold=1.0` counts only a *perfect* score as a success. For a
combined ASR-style claim that is comparable to HarmBench's and SORRY-Bench's
binary success, this module defaults `strongreject_success_threshold=0.5`
(the conventional binary cut). The continuous value is preserved in
`primary_score`, so an analyst can re-threshold offline. Override the param to
restore strict `1.0`.

## Heterogeneous tasks in one claim

`SecurityClaim.from_tasks` performs no homogeneity check; the controller only
calls the `Task` ABC surface (`goal`, `configure_target`, `evaluate`) and
gives every task the same `ChatbotTarget` instance type. All three benchmark
tasks are `Task[ChatbotTarget]`, so mixing them in one claim is sound. Task
order (hence persisted `00001__…`.json filenames) follows insertion order:
SORRY-Bench, then HarmBench, then StrongREJECT, each in its native order.
