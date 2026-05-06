# AutoDAN-Turbo Optimizer — design notes

This document records the design decisions, paper alignment, and
deliberate departures for the AutoDAN-Turbo lifelong-strategy
jailbreak optimizer.

## Reference

- Paper: Xiaogeng Liu et al., "AutoDAN-Turbo: A Lifelong Agent for
  Strategy Self-Exploration to Jailbreak LLMs," arXiv:2410.05295
  (ICLR 2025 spotlight).
- Code: `SaFoLab-WISC/AutoDAN-Turbo`
  (https://github.com/SafoLab-WISC/AutoDAN-Turbo). Specifically:
  `framework/{attacker,scorer,summarizer,retrival,library}.py` and
  `pipeline.py::AutoDANTurbo.lifelong_redteaming`.

## Algorithm summary

Per epoch on a fixed malicious request `M`:

1. **Attack generation** — pick attacker mode based on what was
   observed last epoch:
   - first epoch (or no prior response visible) → cold
     `warm_up_attack`
   - retrieval finds high/medium-score strategies (avg ≥ 2.0) →
     `use_strategy(strategies)` with the best examples
   - retrieval finds only low-score strategies (avg < 2.0) →
     `find_new_strategy(strategies)` with strategies to avoid
2. **Refusal filter** — if the attacker output contains
   `I cannot` / `I am unable` / `I can't`, fall back to the raw
   request `M` as the jailbreak prompt.
3. **Probe** — `target.respond(prompt)` → response `R`.
4. **Score** — scorer LLM judges `(M, R)` on a 1.0–10.0 scale; 8.5 is
   the success threshold.
5. **Strategy mining** — if `score > prev_score`, summarizer LLM
   compares `(prev_prompt, current_prompt)`, extracts a
   `{Strategy, Definition}` object, and adds it to the library with
   `Example=current_prompt`, `Score=score-prev_score`.
6. **Update** — `prev_prompt`, `prev_response`, `prev_score` ←
   current.
7. **Stop** — `score ≥ break_score (8.5)`.

## Mapping to superred

| Paper construct | Superred surface |
| --- | --- |
| One epoch | One superred run |
| Attacker prompt | `ControllablePreCallEvent` for `user_message` (or named override) |
| Target response | `ObservableEvent` on trajectory (preferred), PostCall `answer` fallback |
| Internal scorer | LLM call via the optimizer's `LLMClient` |
| Summarizer | LLM call via `LLMClient`, only when `score > prev_score` |
| Strategy library | In-memory dict, persists across runs of one optimizer instance |
| External judge (when present) | `RunEndEvent.evaluation.success` — additional early-stop |

The library is purely in-memory; persistence (the upstream
`.pkl`/`.json` round-trip) is a workflow concern outside the
optimizer's contract.

## Adversarial information-access settings

The optimizer operates in all four settings without a setting knob —
behaviour is determined by what the framework makes visible.

1. **User query only** (no responses, no eval). No scorer call (no
   response to score), no library updates (no score deltas), no
   retrieval (no prior response to drive mode selection).
   Cold `warm_up` every run; runs to `max_attempts`.
2. **User query + feedback** (no responses, eval visible). Same as 1
   for the inner loop; `RunEndEvent.evaluation.success` triggers
   early-stop.
3. **User query + responses** (responses visible, no eval). The
   internal scorer runs every epoch and is the success signal: stop
   when `score ≥ break_score`. The full lifelong loop engages —
   library updates on score gains, retrieval picks the next attacker
   mode.
4. **User query + responses + feedback**. Full lifelong loop *plus*
   external success early-stop.

## Faithful upstream ports

- **Attacker system prompts** — `warm_up`, `use_strategy`,
  `find_new_strategy` carry the upstream wording (modulo a single
  `[START OF JAILBREAK PROMPT]` / `[END OF JAILBREAK PROMPT]`
  request appended explicitly so the parser is stable across model
  styles; upstream does the same via a prefix-condition mechanic
  specific to its HuggingFace driver).
- **Scorer system prompt** — verbatim, including the 1.0–10.0 scale,
  the 0.5 increments, and the 8.5 success threshold.
- **Summarizer system prompt** — verbatim, including the strategy
  pool listing, the JSON output spec, and the canonical 41-term
  vocabulary the summarizer is encouraged to reuse.
- **Refusal-marker list** — `("I cannot", "I am unable", "I can't")`,
  exactly upstream's `pipeline.py::refusal_list`.
- **Library merge semantics** — adding a strategy with an existing
  name appends `Example` and `Score`, preserving the original
  `Definition` (matches upstream `Library.merge`).
- **Three-tier retrieval logic** — `>=5 → high, [2,5) → medium,
  <2 → low/avoid`, identical thresholds to upstream
  `Retrieval.pop`'s post-faiss filter step (`HIGH_SCORE_THRESHOLD=5`,
  `LOW_SCORE_THRESHOLD=2`).
- **Cold-start fallback in the lifelong path** — when retrieval
  returns no strategies (empty library), the attacker uses
  `warm_up`. Matches upstream `pipeline.py::lifelong_redteaming`.
- **Default temperatures** — attacker 1.0, scorer 0.7, summarizer
  0.6 (paper / run-script values).
- **Default `break_score = 8.5`** — paper / run-script.

## Deliberate departures

### 1. No embedding-based retrieval (v1 simplification)

Upstream's `Retrieval` (`framework/retrival.py`) embeds the prior
target response with OpenAI text-embeddings, builds a faiss
`IndexFlatL2`, and ranks strategies by L2 similarity to the response
embedding before applying the score-tier filter. We drop the
embedding step entirely in v1:

- Superred's `LLMClient` is a text-completion client. There is no
  embedding API on it, and adding hard dependencies on `faiss` and
  `numpy` (plus an embedding model + API key) would inflate a slim
  optimizer module by an order of magnitude.
- The score-tier filter (`>=5 → high, [2,5) → medium, <2 → avoid`) is
  the actual decision boundary for which attacker mode runs next; the
  embedding step only changes *which* of several entries within the
  same tier is shown to the attacker.

In v1 we rank strategies by **average score** across all examples
(the same field the embedding step ultimately relies on after
similarity ranking) and return the highest-scoring ones, capped at
`top_k_strategies`. This preserves the tri-modal `use_strategy` /
`find_new_strategy` / cold-start decision exactly. Embedding-based
ranking can be added later as an optional plug-in (e.g.
`StrategyLibrary` subclass) without touching the optimizer state
machine.

### 2. Lifelong-only path (no separate warm-up stage)

Upstream's pipeline runs a `warm_up → build_from_warm_up_log →
lifelong_redteaming` sequence. The warm-up phase collects a flat
attack log per request, then post-hoc summarises strategies by
comparing the lowest- and highest-scoring entries per request. This
is a workflow detail for batch experiments (collect a log, summarise
at the end).

Superred is online: each run sees one Goal, the library starts empty,
and the lifelong loop subsumes warm-up cleanly — the very first run
does `warm_up` (matches upstream's epoch-0 path inside
`lifelong_redteaming`), and library entries accumulate as score
improvements are observed run-by-run.

### 3. Single-line score parser instead of two-stage scorer LLM

Upstream's scorer makes *two* LLM calls per epoch: a "scoring" call
that returns a free-form analysis containing a number, then a
"wrapper" call that asks a smaller LLM to extract just the number.
We replace the wrapper call with a regex parser:

- Halves scorer cost (every epoch makes a scorer call).
- The wrapper's prompt is one line (`extract the score and output
  only the number`); a regex is at least as reliable on well-formed
  scorer output and falls back gracefully on noise.
- We additionally instruct the scorer system prompt to end with
  `Score: <number>` to give the regex a stable anchor.

### 4. Single-line `{Strategy, Definition}` parser instead of two-stage summarizer

Same trade-off: upstream calls the summarizer twice — once for
analysis, once for JSON extraction. We extract the JSON via regex
(prefer the literal `{"Strategy": ..., "Definition": ...}` shape,
fall back to fenced code blocks, then any `{...}` substring). On
parse failure we return `None` and the optimizer skips the library
update for that run, matching upstream's "retry-then-skip" semantics.

### 5. No multi-request iteration

Upstream iterates over a list of `data['lifelong']` requests. Superred
optimisers operate on one `Goal`. The optimiser keeps `prev_prompt /
prev_response / prev_score` per Goal — exactly the per-request state
upstream maintains inside the lifelong loop.

### 6. Default `max_attempts = 10` (vs paper's 150 for batch experiments)

The paper budgets 150 epochs per request for a batch experiment with
a strategy library being progressively built across 50 requests.
Superred runs are typically interactive single-Goal sessions; 10
matches the merged-optimizer norm (Crescendo, GOAT, Bijection all in
that range). Callers can set `max_attempts=150` for batch-style
experiments.

## Constructor surface

| Argument | Default | Source |
| --- | --- | --- |
| `max_attempts` | `10` | merged-optimizer norm (paper batches use 150) |
| `break_score` | `8.5` | paper / run script |
| `attacker_temperature` | `1.0` | paper |
| `scorer_temperature` | `0.7` | paper |
| `summarizer_temperature` | `0.6` | paper |
| `top_k_strategies` | `1` | retrieval cap |
| `response_observable_names` | `{"response","model_response","assistant_response"}` | matches Crescendo / GEPA / GOAT / Bijection |
| `target_controllable_name` | `None` | optional named-controllable override |
| `max_no_signal_runs` | `0` (disabled) | blind-loop guard |

## Module layout

```
src/autodan_turbo_optimizer/
  __init__.py          # exports AutoDANTurboOptimizer
  attacker.py          # Attacker.warm_up / use_strategy / find_new_strategy
  scorer.py            # Scorer.score(request, response) -> 1.0–10.0
  summarizer.py        # Summarizer.summarize(...) -> StrategyDescriptor | None
  library.py           # StrategyLibrary.add / retrieve / all
  optimizer.py         # AutoDANTurboOptimizer event-state machine
tests/
  test_attacker.py     # 12 tests — three modes + tag extraction
  test_scorer.py       # 12 tests — parser + LLM-driver
  test_summarizer.py   # 11 tests — parser + LLM-driver
  test_library.py      # 14 tests — add + 3-tier retrieval
  test_optimizer.py    # 31 tests — state machine + 4 settings + e2e
```

## Test coverage

80 tests total. Coverage:

- **Library**: add / merge / retrieve at each of the three score
  tiers; empty-library, k-cap, score-strip behaviour.
- **Attacker**: warm-up / use-strategy (single + multi) /
  find-new-strategy system-prompt rendering; START/END tag
  extraction; empty-strategy fall-through.
- **Scorer**: keyword-anchored / fall-through / clamped score parsing;
  request and response embedding; temperature passthrough.
- **Summarizer**: inline-object / fenced-block / escaped-string
  parsing; missing-field returns `None`; temperature passthrough.
- **Optimizer**: construction validation; first run uses warm-up,
  subsequent runs use retrieval; per-run state reset; user-vs-system
  PreCall lock; `target_controllable_name` override; trajectory >
  PostCall response source; scorer-failure resilience; refusal
  fallback to raw goal; library-update gating on score improvement;
  summarizer-`None` skip; break-score early-stop; external eval
  early-stop; failure eval continues; all four adversarial settings;
  `max_no_signal_runs`; ChatbotTarget-style end-to-end.
