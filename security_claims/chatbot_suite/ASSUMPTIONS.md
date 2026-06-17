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

Removing duplicate harmful behaviours is the whole point of combining three
independently-curated benchmarks, so dedup has two layers.

**1. Exact / trivial-variant** (`dedup=True`):
- Key: `normalize_goal(task.goal.description)` = whitespace-collapsed,
  case-folded goal text. Catches exact and trivial-variant duplicates.

**2. Semantic near-duplicates** (`semantic_dedup=True`, the main mechanism):
- Each goal is embedded (`LiteLLMEmbedder`, default
  `openai/text-embedding-3-small`); a candidate is dropped when its goal is
  within `semantic_threshold` cosine (default **0.85**) of an already-kept goal.
  This catches the same behaviour worded differently across benchmarks
  ("make a bomb" vs "give instructions to build an explosive device"), which
  exact-text dedup misses.
- **Threshold rationale**: validated live on the real prompts. At
  `per_category=4` (228 prompts), 0.85 removes genuine near-duplicates (e.g. a
  0.975 drone-surveillance pair, a 0.94 HarmBench pair) and sweeping the cut
  down to 0.75 removed nothing extra, i.e. there is a wide margin between clear
  duplicates (>0.94) and distinct prompts (<0.75) — so 0.85 catches dups without
  over-merging distinct behaviours. Every drop is recorded in
  `CombineStats.semantic_drops` (dropped goal, matched kept task, cosine) so the
  threshold is auditable and tunable per dataset.
- **Limitation**: embedding-cosine catches lexical and close-semantic
  paraphrases; two *very* differently-surfaced statements of one intent can
  score below threshold. Lower the threshold (watch the audit for over-merging)
  or swap in an LLM-judge `Embedder`/deduplicator for higher recall.
- **Determinism / cost**: embeddings are deterministic per model snapshot, and
  `embedding_cache_path` persists text→vector so every claim build (e.g. one per
  matrix cell) makes the *same* decisions and pays the embedding cost once.
  Embedding spend is out of band, like the judges.
- **Sparse-sample note**: the default `per_category=1` stratified sample (57
  tasks) is already diverse, so semantic dedup usually removes nothing there; it
  bites as `per_category` grows.

**First-seen wins** (both layers): sources are processed in list order, so the
first source to contribute a prompt keeps it. `chatbot_suite_claim` orders
sources **SORRY-Bench → HarmBench → StrongREJECT**.

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
