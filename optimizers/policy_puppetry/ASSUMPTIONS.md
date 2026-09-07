# Assumptions and deviations

Provenance and every deliberate departure from the reference implementation.

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| Code | [`garak/probes/doctor.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/doctor.py), [`garak/resources/encodings.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/resources/encodings.py) |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |
| Technique | HiddenLayer, ["Novel Universal Bypass for All Major LLMs"](https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/) |

## Copied byte-for-byte

- `data/upstream/bypass_template_0.txt / bypass_template_1.txt` — upstream `Bypass.templates[0]`. Held
  as a data file rather than a Python literal because the template itself
  contains a triple-quote sequence. `scripts/sync_upstream.py --check`
  re-extracts it from upstream source and fails on any drift.
- `LEET_SLOT_SUFFIX` — upstream `BypassLeet`'s parenthetical aside.
- `render()` — upstream `Bypass._build_prompts()`: a plain `str.replace` of
  every slot occurrence, with `BypassLeet`'s widening applied first.
- `leetspeak()` — upstream's `ieaost` → `134057` mapping and its
  keep-about-half mix rule.

## Deviations

### 1. The goal fills the slot; upstream uses a fixed behaviour list

Upstream renders the template once per entry in a hard-coded
`harmful_behaviors` list (anthrax sporulation, uranium enrichment, bomb
construction, methamphetamine). A superred `Optimizer` pursues the `Goal` it
is given, so the goal fills `{{HARMFUL_BEHAVIOR}}` instead.

This is the mildest deviation of any port so far: upstream already treats the
slot as a parameter, so only the source of the value changes. A test asserts
no upstream payload leaks into a rendered variant.

### 2. Leetspeak is seeded

Upstream draws from the unseeded global `random`, so `BypassLeet` emits a
different prompt every run and results are not reproducible. Here each variant
is seeded `0..n-1`, which matters because superred reports reproducible
metrics. `leetspeak(text, seed=None)` still reproduces upstream's behaviour
exactly.

Upstream's `transmute_chance=0.6` parameter is accepted but never used — the
mix rate is hard-coded at `0.5`. That bug is preserved rather than "fixed", so
the output distribution matches upstream.

### 3. One scene per run

Upstream builds `len(templates) x len(harmful_behaviors)` prompts and sends
them as one probe. Here each variant is one run, so a claim scores each scene
separately and the optimizer can stop early on success.

### 4. `Puppetry` is not ported

Upstream's third class sends a fixed medical-advisor config whose request is
hard-coded (`<query>treatment plan for skin cancer`). It exposes no slot, so
it cannot pursue an arbitrary goal; substituting into its `<query>` element
would mean inventing a slot upstream does not have. Excluded, consistently
with every other fixed-objective probe.

### 5. Detectors are out of scope

Upstream pairs these probes with `mitigation.MitigationBypass`. In superred,
judging belongs to the `SecurityClaim`, so no detector is ported.


## Upstream coverage

`garak/probes/doctor.py` defines three probes. This module ports the two that
are goal-driven:

| Upstream probe | Ported | Why |
| --- | --- | --- |
| `Bypass` | yes — **both** `templates[0]` and `templates[1]` | upstream's `_build_prompts` loops over every template; the optimizer emits one prompt per template |
| `BypassLeet` | yes (`leet` technique) | subclasses `Bypass`, adding the leetspeak aside to both templates |
| `Puppetry` | **no** | it sets `self.prompts` to a single fixed 1565-char scene with **zero** `{{HARMFUL_BEHAVIOR}}` slots, so it cannot carry an arbitrary superred `Goal`. It is a fixed-objective probe and belongs as a `SecurityClaim`, not a goal-driven optimizer. |

## Known limitations

- The template is one scene, so `plain` yields exactly one variant; breadth
  comes from `leet_variants`.
- Leetspeak mangles the goal's own letters too, so a judge doing exact string
  matching on the goal may miss a success. Prefer a semantic judge with this
  optimizer, or use `techniques=["plain"]`.
- The scene is ~3 kB before the goal is added, which is a meaningful share of a
  short context window and is visible to any prompt-length guardrail.
- Upstream's `<scene-summary>` closes with `</scene-ending>` and other tags are
  likewise malformed. That is preserved verbatim; the attack appears to work
  despite it, and "fixing" it would no longer be the published prompt.
