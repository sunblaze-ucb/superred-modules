# chatbot_suite

A combined chatbot-safety `SecurityClaim` for superred. It unions the three
real chatbot harmful-instruction benchmarks shipped in this repo into one
**strong overall claim**, removing functional duplicates so no attacker or
judge tokens are spent re-testing the same harmful behaviour.

```
HarmBench  ─┐
SORRY-Bench ─┼─▶  combine (dedup + stratified per-category subset)  ─▶  one SecurityClaim
StrongREJECT ┘
```

## Why

Each benchmark probes chatbot safety from a different angle (HarmBench's
functional/semantic behaviours, SORRY-Bench's 44-category taxonomy,
StrongREJECT's rubric-scored prompts). Running them as one deduplicated claim
gives a broader, harder safety bar than any single benchmark, while the
de-duplication keeps the combined claim from paying twice for the same prompt.

Each surviving task keeps **its own native judge and configuration** — there
is no shared judge. Combining is purely a union over the task lists.

## Public API

```python
from security_claim_chatbot_suite import chatbot_suite_claim, combine_claims
```

### `chatbot_suite_claim(...)` — the headline factory

Builds HarmBench + SORRY-Bench + StrongREJECT and combines them.

```python
claim = chatbot_suite_claim(
    target_model_id="openai/gpt-5.4-mini-2026-03-17",  # picks SORRY-Bench's system prompt
    judge_api_base=LITELLM_API_BASE,
    judge_api_key=LITELLM_API_KEY,
    per_category=1,            # smallest stratified sample spanning every category
)
```

Key parameters:

| param | default | meaning |
|---|---|---|
| `target_model_id` | (required) | victim id; only used to select SORRY-Bench's faithful system prompt |
| `judge_api_base` / `judge_api_key` | (required) | proxy credentials shared by all judges |
| `per_category` | `1` | max prompts per source-benchmark category (`None` = full benchmarks) |
| `include_harmbench` / `include_sorrybench` / `include_strongreject` | `True` | toggle each source |
| `dedup` | `True` | drop tasks whose normalized goal text already appeared |
| `sorry_judge_model` | `openai/gpt-4-turbo-2024-04-09` | SORRY-Bench judge (proxy substitute for `gpt-4-1106-preview`) |
| `harmbench_judge_model` | `openai/gpt-4o-mini` | HarmBench judge |
| `harmbench_split` | `"test"` | `"test"` (320) or `"val"` (80) |
| `strongreject_success_threshold` | `0.5` | continuous-score threshold for `success` (StrongREJECT's own default is the strict `1.0`) |
| `manifest_out` / `stats_out` | `None` | optional sinks for per-task provenance / combine stats |

At `per_category=1` the suite is **57 tasks** (44 SORRY-Bench + 7 HarmBench
semantic categories + 6 StrongREJECT categories; 0 cross-benchmark exact
duplicates). Scale up with `per_category`.

### `combine_claims(...)` — the generic combinator

Benchmark-agnostic. Unions labelled `SecurityClaim`s, deduping by normalized
`task.goal.description` and optionally capping per category:

```python
from security_claim_chatbot_suite import combine_claims, TaskRecord

manifest: list[TaskRecord] = []
combined = combine_claims(
    [("harmbench", hb_claim), ("strongreject", sr_claim)],
    dedup=True,
    max_per_category=1,
    manifest_out=manifest,   # provenance for each kept task (index, source, category, goal)
)
```

It touches only the `Task` ABC (`task.goal.description`) plus a pluggable
`category_getter`, so it works on any superred claims, not just these three.

## Provenance manifest

`manifest_out` / `manifest_to_dicts(...)` give, per surviving task, its 1-based
index (== persisted detail-file index), source benchmark, category, concrete
task class, and a goal preview. Write it next to a run's results so an analyst
can attribute every task back to its benchmark and category.

## Install

```bash
pip install -e security_claims/chatbot_suite
```

Pulls in `secclaim-harmbench`, `security-claim-sorry-bench`, and
`strongreject-claim`. HarmBench and StrongREJECT bundle their data;
**SORRY-Bench requires its dataset** (HF cache after `hf auth login`, or
`sorry_dataset_path=...`).

See `ASSUMPTIONS.md` for the dedup semantics, source-priority and
judge-model choices, and the StrongREJECT threshold rationale.
