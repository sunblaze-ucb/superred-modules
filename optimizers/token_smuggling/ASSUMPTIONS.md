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

### 3. `FunctionMasking` is ported, with a verbatim fallback

Upstream's third class wraps a fixed `predict_mask` pseudocode prompt whose
masked words resolve to a hard-coded "rob a store", so it has no goal slot. This
module does not drop it: an LLM rewrites the masked vocabulary from the task goal
(see the **FunctionMasking** subsection under *Upstream coverage*), and upstream's
complete prompt is retained verbatim as `UPSTREAM_FUNCTION_MASKING_PROMPT` and
emitted as the fallback when no rewrite is available. So all three probes are
ported; an earlier revision excluded this one, and that is no longer the case.

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

### Injection-surface selection (superred adaptation)

Upstream aims a single prompt at a chat box; superred targets expose arbitrary
named surfaces. Following the house pattern used by `muzzle`, `chord_xthp`,
`eia_agent`, `gepa_agentic`, `poisonedrag` and `agentvigil_websentinel`, this
module ships a byte-identical copy of the shared `surface_llm.py` and calls
`classify_controllables` once at `initialize`: the attacker's own LLM reads
each surface's description and says which one is the user's prompt, so the
payload is not spent on a content surface that merely fired first.

The classifier returns `{}` on any failure (no budget, malformed reply), and a
surface the target raises at run time without listing it is unknown to the
classification; both fall back to the previous name/value-type backstop, so
behaviour is unchanged when no LLM is available.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules that ship the
classifier resolve this by not carrying the guard at all; this module keeps the
guard and skips only that one pinned shared file, so the guard still covers all
first-party code here.

## Upstream coverage

`garak/probes/smuggling.py` defines three probes; all three are ported:

| Upstream probe | Ported as | Note |
| --- | --- | --- |
| `HypotheticalResponse` | `hypothetical` technique | template vendored verbatim, goal in the slot |
| `HomoglyphObfuscation` | `homoglyph` technique | map and algorithm byte-faithful, seeded for reproducibility |
| `FunctionMasking` | `function_masking` technique | see below |

### FunctionMasking

Upstream ships **one fixed prompt** whose two masked words ("rob", "store") are
baked in, so it cannot carry an arbitrary goal. The scaffolding — the
`predict_mask` preamble, the `prompt = '...' + X + '...'` assembly line and the
closing instruction — is reproduced verbatim, and upstream's complete prompt is
kept as `UPSTREAM_FUNCTION_MASKING_PROMPT`.

Only the masked vocabulary generalises: the attacker's own LLM picks which
words of the goal to hide and writes a dictionary definition for each, so the
sensitive words never appear literally. Any failure (no budget, malformed
reply, a word that is not in the goal) yields no definitions, and the technique
then emits **upstream's verbatim prompt** — so the module never silently
degrades into something weaker than the original probe.

`active = False` upstream (it is opt-in); here it is one of three techniques
and can be selected with `techniques=["function_masking"]`.

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
