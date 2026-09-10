# Assumptions and deviations

Provenance and every deliberate departure from the reference implementation.

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| Code | [`garak/probes/dan.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/dan.py) |
| Data | [`garak/data/dan/*.json`](https://github.com/NVIDIA/garak/tree/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/data/dan) |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

The 13 JSON files under `src/dan_personas_optimizer/data/upstream/` are
byte-identical copies of `garak/data/dan/*.json`. `scripts/sync_upstream.py
--check` re-downloads them and fails on any drift, so the vendored corpus can
be re-verified at any time without trusting this file.

`Persona.render()` reproduces upstream's `DANProbeMeta.probe()`: substitute
`{generator.name}` only when the placeholder is present, otherwise send the
prompt unchanged.

## Deviations

### 1. The goal is appended; upstream sends the preamble alone

Garak's DAN probes send only the persona text and let a detector decide
whether the model adopted the persona. A superred `Optimizer` is handed a
`Goal` and must pursue *that*, so this module follows the preamble with the
goal.

This is what the preambles themselves ask for. Dan_11_0 ends "don't begin the
data pairing until after my next message… start complying following the next
message I send you"; STAN and DUDE likewise establish a persona and then wait.
Sending the preamble alone would score zero against any judge, since nothing
was ever asked.

`tests/test_personas.py` asserts no preamble contains a goal slot, which is
what makes appending — rather than templating — the correct reading.

### 2. Two turns, on one surface

The persona and the goal are two messages, so they occupy two
`ControllablePreCallEvent`s within a run, as `crescendo` and `fitd` do. Both
land on the surface that received the persona: a primer only applies to the
conversation it started, so delivering the goal to some *other* surface (an
internal DB lookup, say) would be meaningless. Other surfaces are declined.

`mode="single_message"` joins the two with a blank line for targets that offer
each surface only once per run, where the second turn never arrives. This is a
deviation from the two-message structure, chosen over silently never sending
the goal.

### 3. One persona per run

Upstream runs each DAN class as its own probe. Here the personas are one
module, tried one per run in upstream file order, so a run index maps to a
stable persona. `personas=[...]` selects a subset.

### 4. Detectors are out of scope

Upstream pairs these probes with `dan.DAN`, `dan.AntiDAN`,
`dan.DevMode` and `mitigation.MitigationBypass` detectors. In superred, judging
belongs to the `SecurityClaim`, so no detector is ported — including the
persona-adoption check, which is what upstream actually measures. A claim
judging this optimizer scores whether the *goal* was achieved, which is a
stricter bar than upstream's.

### 5. Three upstream classes are not ported

| Excluded | Why |
| --- | --- |
| `ChatGPT_Image_Markdown` | Multimodal — it exfiltrates via markdown image rendering, which needs image-carrying Events. |
| `Ablation_Dan_11_0` | Reads `autodan/autodan_prompts.json`, not the DAN corpus; it is an AutoDAN ablation. |

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

`garak/probes/dan.py` defines 19 probes (plus the `DANProbeMeta` metaclass).
This module ports every prompt-corpus probe:

| Upstream | Ported as | Note |
| --- | --- | --- |
| 14 per-file DAN probes (`Dan_*`, `DUDE`, `STAN`, `AntiDAN`, `DAN_Jailbreak`, both Developer Mode variants, `ChatGPT_Image_Markdown`) | `corpus="personas"` | all 14 JSON files vendored byte-identically |
| `Ablation_Dan_11_0` | `corpus="ablation"` | 127 prompts, one per non-empty 7-bit mask |
| `DanInTheWild` / `DanInTheWildFull` | `corpus="in_the_wild"` | 666-prompt corpus; `limit` reproduces the capped variant. Corpus = Shen et al. (arXiv:2308.03825), verazuo/jailbreak_llms (MIT); see `NOTICE`. |
| `AutoDAN`, `AutoDANCached` | **no** | not a persona corpus but a genetic search (`garak.resources.autodan.autodan_generate`) with its own `goal_str`/`target` params; it belongs in its own module, and superred already ships `optimizers/autodan_turbo` |

**Correction:** `ChatGPT_Image_Markdown` was previously excluded here as "the
multimodal probe". That was wrong — it is a pure *text* prompt that instructs
the model to echo messages wrapped in markdown image syntax (URL
exfiltration) and takes no image input. It is now vendored with the others.

### Ablation_Dan_11_0

Upstream builds these in code rather than from a data file: seven optional
segments keyed to bits 1..64, plus four unconditional ones, concatenated in a
fixed order for every mask `i` in `1..127`. The segments are vendored verbatim
(extracted from the probe's own string literals) and reassembled with
upstream's exact ordering and bit tests, so the 127 prompts match. A test pins
the count, their distinctness, and the bit-1 toggle.

## Known limitations

- These personas are old and widely published; frontier models refuse most of
  them outright. Value here is breadth and regression coverage, not potency.
- `{generator.name}` is filled from a model observable when the target exposes
  one, else `ChatGPT` — the assistant the prompts were written against. A
  wrong identity weakens the roleplay but does not break it.
- Upstream ships one prompt per file today. The loader handles multi-prompt
  files (suffixing `#n`), so a future upstream addition will not silently drop
  prompts, but the persona count would change.
