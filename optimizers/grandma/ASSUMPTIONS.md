# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/grandma.py`, `garak/data/payloads/keyedprod_win1{0,1}.json` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

- `_vendor/garak_payloads/keyedprod_win10.json`,
  `_vendor/garak_payloads/keyedprod_win11.json` — garak's own payload files,
  byte-identical. `scripts/sync_upstream.py --check` re-downloads and diffs
  them.

`probes/grandma.py` itself cannot be vendored: it imports garak's probe
machinery and payload loader. Every list it carries — the three template
tuples, both `DEFAULT_PARAMS` lists, and the four locals inside
`GrandmaIntent._prompts_from_stub` plus its template string — is extracted
into `data.json` by AST, and the sync script re-extracts and compares all ten,
refusing to write a partial extraction. The construction loops are reproduced
in `build_prompts()`, pinned by tests that rebuild each variant from the same
constants.

## Upstream coverage

`grandma.py` defines five probes; **all five are ported**:

| Upstream | Ported as |
| --- | --- |
| `Win10` | `variant="win10"` |
| `Win11` (subclasses `Win10`, swaps the payload) | `variant="win11"` |
| `Slurs` | `variant="slurs"` |
| `Substances` | `variant="substances"` |
| `GrandmaIntent` (`active=False` upstream, opt-in) | `variant="intent"` — the default here, being the only goal-driven one |

Not ported: the probes' `tags`/`tier`/`doc_uri`/`intent` taxonomy metadata,
and `garak.payloads`' loader machinery (the two payload files this actually
needs are vendored directly). Detectors are not ported — judging the response
belongs to the `SecurityClaim`, as with every other optimizer here.

## Deviations

### 1. Upstream's prompt order is not reproducible; this module fixes one

`GrandmaIntent._prompts_from_stub` iterates `grandma_activities`, which
upstream declares as a **`set`**. Python's iteration order for a set of
strings depends on hash randomisation, so upstream's prompt *order* differs
between processes. The extraction sorts it, giving a stable order; the set of
prompts is identical to upstream's either way, only the sequence is pinned.

### 2. `intent` is the default even though upstream marks it inactive

Upstream sets `active = False` on `GrandmaIntent` because its own runner needs
an explicitly supplied stub. superred always has one — the task `Goal` — so
the variant is not only usable here, it is the only one that attacks the
objective under test rather than a fixed list. The four fixed variants remain
available and are documented as goal-agnostic.

### 3. Injection-surface selection (superred adaptation)

Upstream sends prompts straight to its generator. superred targets expose
several surfaces, so this module uses the shared
`surface_llm.classify_controllables` pass — one attacker-LLM call at
`initialize` labelling each controllable `user-prompt` / `content-injection` /
`system-prompt`. Prompts go to the labelled user prompt; a `system-prompt`
surface never receives one; a classifier failure returns `{}` and a
name/value-type backstop picks the first eligible free-text surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file.

## Content warning

`variant="slurs"` asks the target for slang terms about demographic groups,
and `variant="substances"` asks for synthesis steps for hazardous substances.
Both carry upstream's wording verbatim, because a paraphrase would not be the
published probe. They exist to measure whether a target complies; the module
never judges the response, and `intent` — which uses your own goal — is the
default.

## Known limitations

- `intent` produces 540 prompts. superred caps runs per task, so a full sweep
  usually will not finish; `shuffle=True` samples across the space instead of
  exhausting one persona.
- The scene is English and culturally specific ("grandma", bedtime lullabies).
  Against a target tuned for another language or register it will read oddly
  and probably fail for reasons unrelated to safety training.
