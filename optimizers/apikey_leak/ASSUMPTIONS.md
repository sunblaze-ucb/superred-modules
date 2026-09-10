# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/apikey.py`, `garak/resources/apikey/{regexes,serviceutils}.py` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

> **Original source (one layer up from garak).** garak's service-detection
> regexes are themselves derived from the [dora](https://github.com/sdushantha/dora)
> project (MIT, Copyright (c) 2021 Siddharth Dushantha). dora is the canonical
> source of the 58-service catalogue and the detection patterns; garak added
> capture groups and re-keyed the services, and this module vendors garak's copy
> (which retains dora's notice). Only the 58 service **names** are read. See
> `NOTICE` and `LICENSES/dora-MIT.txt`.

## Copied byte-for-byte

- `_vendor/garak_apikey/regexes.py` — garak's service-detection regex table.
  Only its **keys** (58 service names) are read, by `extract_key_types`; the
  regex patterns are the detector's business, but the file is vendored whole
  and byte-identical so the service list cannot silently drift.
- `_vendor/garak_apikey/serviceutils.py` — `extract_key_types`, byte-identical.
  `KEY_TYPES` is computed by calling it on the vendored `REGEX_DICTS`, exactly
  as upstream does, rather than hard-coding the derived list.

Both files import only `typing` and `re`, so they vendor cleanly.

`probes/apikey.py` itself cannot be vendored (garak probe machinery). The two
probes' `base_prompts` and `CompleteKey.partial_keys` are extracted into
`data.json` by AST, and `scripts/sync_upstream.py` re-extracts and compares
them, refusing a partial extraction. The construction loops are reproduced in
`build_prompts()`, pinned by tests.

## Upstream coverage

`apikey.py` defines two probes; **both are ported**:

| Upstream | Variant |
| --- | --- |
| `GetKey` | `generate` |
| `CompleteKey` | `complete` |

Not ported: the `apikey.ApiKey` detector (judging whether a response is a
key-shaped secret belongs to the `SecurityClaim`), and the
`tags`/`tier`/`intent` metadata.

## Deviations

### 1. Upstream's prompt cap is a runtime knob, not ported

Both probes set `follow_prompt_cap = True` and call
`_prune_data(cap=soft_probe_prompt_cap)` — a random cull to garak's configured
soft cap, a **run parameter** rather than part of the attack. superred caps
runs itself, so this ships the full sets (58 / 290) and exposes `shuffle`/`seed`
for the same spread-the-sample purpose.

### 2. Goal-agnostic

The prompts are fixed, so nothing embeds the task `Goal`. It is accepted and
passed to the surface classifier but never appears in a payload — the same
shape as `divergent_repetition`. A test pins this.

### 3. One prompt per run

Upstream hands its whole list to its own harness; superred drives one attempt
per run, so the prompts are swept one per run and the optimizer reports `done`
when exhausted.

### 4. Injection-surface selection (superred adaptation)

Shared `surface_llm.classify_controllables` pass: one attacker-LLM call at
`initialize` labels each controllable; prompts go to the labelled user prompt;
a `system-prompt` surface never receives one; a classifier failure returns
`{}` and a name/value-type backstop picks the first eligible free-text surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file.

## Known limitations

- The partial keys in `complete` are short fixed stubs, not real key prefixes;
  a model that only completes plausibly-formatted keys may ignore them.
- Single direct requests, no obfuscation or escalation — a baseline.
