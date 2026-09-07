# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| Code | [`garak/probes/snowball.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/snowball.py), [`garak/detectors/snowball.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/detectors/snowball.py) |
| Data | `garak/data/{graph_connectivity,primality_testing,senator_search}.json` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 (garak); question sets originally MIT (Zhang et al., Nanami18/Snowballed_Hallucination) |

## Copied byte-for-byte

- The three JSON question sets under `data/`.
- The `_YES` / `_NO` detector regexes in `judge.py`, from `DontStartYes` /
  `DontStartNo`.

`scripts/sync_upstream.py --check` re-downloads the data and re-extracts the
regexes from upstream, and fails on any drift.

## Deviations

### 1. The deterministic detector is the judge

Snowball's grading is a regex, not a model — so this claim ships that regex as
its judge and needs no `judge_model` (unlike the HarmBench / JailbreakBench
claims). A match means the model produced the hallucinated confident answer,
which is the attack success. Polarity is fixed per subset, exactly as garak
pairs each probe with `DontStartYes` or `DontStartNo`.

### 2. Question is the goal; single-turn

Each question becomes a `Task` goal; `configure_target` clears the system prompt
(garak adds none). One question, one response, judged once.

### 3. `limit` mirrors garak's capped variants

garak ships a `*Full` probe (all 500) and a default-active capped probe
(`self.prompts[-100:]`). `limit=100` (default) reproduces the capped variant
from the *end* of the file; `limit=None` reproduces `*Full`. `Primes`/`Senators`
are `active=False` upstream (opt-in); here all three subsets are included by
default and selectable via `subsets=`.

### 4. Per-response detector, not a corpus aggregate

garak's detector returns one score per output and aggregates across the probe.
Here each question is its own `Task`, so the framework aggregates ASR across
tasks — the same numerator/denominator, surfaced through superred's own metrics.

## Upstream coverage

`garak/probes/snowball.py` defines six probes — three question sets, each in a
`*Full` and a capped variant. **All are covered:**

| Upstream | Ported as |
| --- | --- |
| `GraphConnectivityFull` / `GraphConnectivity` | `subsets=["graph_connectivity"]`, `limit=None` / `limit=100` |
| `PrimesFull` / `Primes` | `subsets=["primes"]` |
| `SenatorsFull` / `Senators` | `subsets=["senators"]` |

Both detectors (`DontStartYes`, `DontStartNo`) are ported, paired with the same
subsets upstream pairs them with. Nothing in this probe family is omitted.

## Known limitations

- The `_YES` regex matches the prefix `there's ` regardless of what follows,
  so "There's no route" reads as an affirmation (a false positive). This is
  garak's exact regex, preserved rather than "fixed"; changing it would diverge
  from the benchmark's published grading.
- The detector keys on the *first* words of the response (an affirmation or
  negation). A model that buries a wrong answer after hedging, or answers in an
  unusual phrasing, can be missed — this is upstream's known regex limitation,
  preserved for faithfulness rather than "improved".
- `primes` and `senators` are `active=False` in garak (the maintainers treat
  graph_connectivity as the headline). They are included here but selectable, so
  a run can match garak's default by choosing `subsets=["graph_connectivity"]`.
- Success measures hallucination, not harm: this claim answers "can the target
  be made to state a confident falsehood?", a different property from the
  safety-refusal claims.
