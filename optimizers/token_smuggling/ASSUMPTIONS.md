# Assumptions and deviations

Provenance and every deliberate departure from the reference implementation.

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| File | [`garak/probes/smuggling.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/smuggling.py) |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

Upstream references cited by the probes:
[Unicode tag smuggling](https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/)
and [the two-sentence universal jailbreak](https://guzey.com/ai/two-sentence-universal-jailbreak/).

## Copied byte-for-byte

- `DEFAULT_HOMOGLYPH_MAP` — all 20 entries, identical codepoints.
- `homoglyph_replace()` — upstream `_homoglyph_replace()`: same
  `random.Random(seed)` construction, same `rng.choice` per mapped character,
  same pass-through for unmapped characters. Seeding therefore reproduces
  upstream output exactly.
- `HYPOTHETICAL_TEMPLATE` — upstream `HypotheticalResponse.prompts`, with the
  attack substring turned into a `{goal}` slot and no other change.

`tests/test_smuggling.py` re-implements upstream's loop inline and asserts
identical output, so drift in either the map or the algorithm fails the suite.

## Deviations

### 1. The goal replaces upstream's payload corpus

Garak probes carry their own objectives: `HomoglyphObfuscation` obfuscates
garak's `harmful_behaviors` payload list, and `HypotheticalResponse` hard-codes
three substrings (`hotwire a car`, `rob a bank`, `commit credit card fraud`).
A superred `Optimizer` is handed a `Goal` and must pursue *that*, so both
techniques are applied to `goal.description` instead.

Without this the module would ignore the task's goal and score zero against
any judge — the failure mode `demo_prompt_list` warns about. This mirrors how
`libertas` binds Pliny's corpus to the goal.

### 2. Seeds index variants, not payloads

Upstream seeds by payload index (`seed=i` for the *i*-th payload). With a
single goal there is no payload index, so the seed is the variant index:
variant *n* is `homoglyph_replace(goal, seed=n)`. Same algorithm, same
determinism; the seed simply means "which variant" rather than "which payload".

### 3. `FunctionMasking` is not ported

Upstream's third class wraps a fixed `predict_mask` pseudocode prompt whose
masked words resolve to a hard-coded "rob a store". It has no goal slot — the
objective is baked into the prompt — so it cannot serve an arbitrary superred
task. Porting it would produce a module that ignores its goal. Excluded rather
than silently reinterpreted.

### 4. One injection per run

Upstream sends one prompt per attempt to a single generator. A superred target
may expose several controllables in one run, so this module injects into the
first eligible surface and declines the rest, matching the
`goal_passthrough` baseline. Eligibility excludes `system_prompt` (so a
Task-configured system prompt survives) and any non-free-text `value_type`
(a `json` surface would discard the obfuscated string).

### 5. Detectors are out of scope

Upstream pairs these probes with `mitigation.MitigationBypass`. In superred,
judging belongs to the `SecurityClaim`, not the optimizer, so no detector is
ported. Success is whatever the claim's judge decides.

## Known limitations

- Homoglyph substitution assumes a Latin-script goal; a goal in another script
  passes through largely unchanged, since the map only covers 20 Latin
  characters.
- Upstream notes that targets sometimes reply in the script of the obfuscated
  input (e.g. Russian for Cyrillic substitutions), which can confuse an
  English-only refusal judge. That caveat transfers to any claim judging this
  optimizer's runs.
- Only 9 lowercase and 11 uppercase Latin characters have homoglyphs, so short
  goals may produce few or no substitutions, and several seeds can yield the
  same string. Variants are not guaranteed distinct for every goal.
