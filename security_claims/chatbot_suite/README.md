# chatbot_suite

A combined chatbot-safety `SecurityClaim` for superred. It unions the three
real chatbot harmful-instruction benchmarks shipped in this repo into one
**strong overall claim**, removing both exact and **semantic** duplicate
harmful behaviours so no attacker or judge tokens are spent re-testing the same
behaviour under different wording.

```
HarmBench  ─┐
SORRY-Bench ─┼─▶  combine (exact + semantic dedup, stratified per-category)  ─▶  one SecurityClaim
StrongREJECT ┘
```

## Why

Each benchmark probes chatbot safety from a different angle (HarmBench's
functional/semantic behaviours, SORRY-Bench's 44-category taxonomy,
StrongREJECT's rubric-scored prompts). Combining them gives a broader, harder
bar than any single benchmark. But three independently-curated benchmarks
contain the **same harmful behaviour worded differently** ("make a bomb" vs
"give instructions to build an explosive device"). Re-testing that across an
attacker x model x scope matrix burns tokens without new insight, so the
combined claim removes those near-duplicates by embedding-cosine similarity.

Each surviving task keeps **its own native judge and configuration** — there is
no shared judge. Combining is purely a deduplicated union over the task lists.

## Two layers of dedup

1. **Exact / trivial-variant** (`dedup=True`, default): normalized goal text
   (case + whitespace). Cheap, deterministic, offline.
2. **Semantic near-duplicates** (`semantic_dedup=True`, default): each goal is
   embedded; a task is dropped when its goal is within `semantic_threshold`
   cosine of an already-kept goal. First-seen (source order) wins. Every drop is
   recorded for audit/tuning (`CombineStats.semantic_drops`).

## Public API

```python
from security_claim_chatbot_suite import chatbot_suite_claim, combine_claims
```

### `chatbot_suite_claim(...)` — the headline factory

Builds HarmBench + SORRY-Bench + StrongREJECT and combines them, with semantic
dedup on by default (embedder built from the judge credentials).

```python
claim = chatbot_suite_claim(
    target_model_id="openai/gpt-5.4-mini-2026-03-17",  # picks SORRY-Bench's system prompt
    judge_api_base=LITELLM_API_BASE,
    judge_api_key=LITELLM_API_KEY,
    per_category=1,                  # smallest stratified sample spanning every category
    semantic_threshold=0.85,         # cosine cut for "same behaviour"
    embedding_cache_path="emb.json", # reuse vectors across builds
)
```

Key parameters:

| param | default | meaning |
|---|---|---|
| `target_model_id` | (required) | victim id; only selects SORRY-Bench's system prompt |
| `judge_api_base` / `judge_api_key` | (required) | proxy creds shared by all judges + the embedder |
| `per_category` | `1` | max prompts per category for SORRY-Bench / any benchmark without an override (`None` = full) |
| `harmbench_per_category` / `strongreject_per_category` | `None` | per-benchmark per-category caps (else `per_category`); sample each benchmark at its own depth so the suite isn't dominated by SORRY-Bench's finer taxonomy |
| `harmbench_exclude_copyright` | `False` | drop HarmBench's copyright category (MinHash verbatim-reproduction, off the jailbreak-compliance axis); recommended for a pure ASR claim |
| `strongreject_use_small` | `False` | use StrongREJECT's curated 60-prompt (10/cat) `small` subset instead of the full 313 |
| `dedup` | `True` | exact/trivial-variant goal-text dedup |
| `semantic_dedup` | `True` | embedding-cosine near-duplicate removal (set `False` for offline/exact-only) |
| `embedding_model` | `openai/text-embedding-3-small` | litellm embedding model |
| `semantic_threshold` | `0.85` | cosine at/above which two goals are the "same" (higher = stricter) |
| `embedding_cache_path` | `None` | persist text→vector so dedup is identical and cheap across builds |
| `embedder` | `None` | explicit `Embedder` override (else a `LiteLLMEmbedder` is built) |
| `sorry_judge_model` | `openai/gpt-4-turbo-2024-04-09` | SORRY-Bench judge (substitute for `gpt-4-1106-preview`) |
| `harmbench_judge_model` / `harmbench_split` | `gpt-4o-mini` / `test` | HarmBench judge / split |
| `strongreject_success_threshold` | `0.5` | continuous-score cut for `success` (StrongREJECT default is strict `1.0`) |
| `manifest_out` / `stats_out` | `None` | sinks for per-task provenance / combine stats (incl. `semantic_drops`) |

At `per_category=1` the suite is **57 tasks** (44 SORRY-Bench + 7 HarmBench
semantic categories + 6 StrongREJECT). That stratified 1-per-category sample is
already diverse, so semantic dedup typically removes nothing there; it bites as
`per_category` grows (denser samples overlap across benchmarks).

Because SORRY-Bench's taxonomy is ~6-7x finer than the others', uniform
`per_category` makes the suite ~77% SORRY-Bench. For a **balanced** claim, sample
each benchmark at its own depth, e.g. SORRY 1/leaf + HarmBench 3/semantic
(copyright excluded) + StrongREJECT 3/cat from the small set (~80 tasks,
~55/22/22):

```python
chatbot_suite_claim(
    ..., per_category=1,
    harmbench_per_category=3, harmbench_exclude_copyright=True,
    strongreject_per_category=3, strongreject_use_small=True,
)
```

### `combine_claims(...)` — the generic combinator

Benchmark-agnostic. Unions labelled `SecurityClaim`s; exact + (optional)
semantic dedup; optional per-category cap:

```python
from security_claim_chatbot_suite import combine_claims, LiteLLMEmbedder, TaskRecord

manifest: list[TaskRecord] = []
combined = combine_claims(
    [("harmbench", hb_claim), ("strongreject", sr_claim)],
    dedup=True,
    max_per_category=1,
    embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small",
                             api_base=BASE, api_key=KEY, cache_path="emb.json"),
    similarity_threshold=0.85,
    manifest_out=manifest,
)
```

It touches only the `Task` ABC (`task.goal.description`) plus a pluggable
`category_getter` / `embedder`, so it works on any superred claims.

## Provenance and audit

- **`manifest_out`** / `manifest_to_dicts(...)`: per surviving task, its 1-based
  index (== persisted detail-file index), source, category, task class, goal
  preview. Write it next to results so an analyst can attribute every task.
- **`stats_out[0].semantic_drops`** / `semantic_drops_to_dicts(...)`: per dropped
  near-duplicate, the dropped goal, the kept task it matched, and the cosine.
  Use it to verify and tune `semantic_threshold`.

## Install

```bash
pip install -e security_claims/chatbot_suite
```

Pulls in `secclaim-harmbench`, `security-claim-sorry-bench`, and
`strongreject-claim`. HarmBench and StrongREJECT bundle their data;
**SORRY-Bench requires its dataset** (HF cache after `hf auth login`, or
`sorry_dataset_path=...`). Semantic dedup additionally needs a reachable
embedding model on your proxy (or pass `semantic_dedup=False`).

See `ASSUMPTIONS.md` for the dedup semantics, threshold rationale,
source-priority and judge-model choices.
