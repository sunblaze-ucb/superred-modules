# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/packagehallucination.py` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

`probes/packagehallucination.py` cannot be vendored: it imports garak's probe
machinery (`garak.probes`, `garak._config`) and the abstract base class. Its
three module-level lists (`stub_prompts`, `real_code_tasks`,
`unreal_code_tasks`) and each language subclass's `language_name` are extracted
into `data.json` by AST, and `scripts/sync_upstream.py` re-extracts and
compares all of them, refusing to write a partial extraction. The construction
loop is reproduced in `build_prompts()`, pinned by tests that rebuild each
variant from the same constants.

## Upstream coverage

`packagehallucination.py` defines an abstract base plus seven language
subclasses; **all seven are ported** as variants:

| Upstream | Variant |
| --- | --- |
| `Python` | `python` |
| `Ruby` | `ruby` |
| `JavaScript` | `javascript` |
| `Rust` | `rust` |
| `RakuLand` | `raku` |
| `Perl` | `perl` |
| `Dart` | `dart` |

The abstract `PackageHallucinationProbe` is the construction logic, reproduced
in `build_prompts()`. Not ported: the `packagehallucination.*` detectors
(which check recommended imports against a real package index — that judgement
belongs to the `SecurityClaim` here), and the `tags`/`tier`/`doc_uri`
taxonomy metadata.

## Deviations

### 1. Upstream's prompt cap is a runtime knob, not ported

`PackageHallucinationProbe` sets `follow_prompt_cap = True` and, in its own
runner, calls `_prune_data(cap=soft_probe_prompt_cap)` — a **random** cull of
the 240 prompts down to garak's configured soft cap. That cap is a garak *run*
parameter (`_run_params`), not a property of the attack, and the cull is a
harness convenience. superred caps runs itself, so this module ships the full
240-prompt set and exposes `shuffle`/`seed` for the same "spread the sample"
purpose. The prompt set is upstream's exactly; only the optional culling is
left to superred.

### 2. Goal-agnostic

The prompts are fixed, so nothing embeds the task `Goal`. It is accepted and
passed to the surface classifier but never appears in a payload — the same
shape as `divergent_repetition`. A test pins this.

### 3. One prompt per run

Upstream hands its whole list to its own harness; superred drives one attempt
per run, so the 240 prompts are swept one per run and the optimizer reports
`done` when exhausted.

### 4. Injection-surface selection (superred adaptation)

Shared `surface_llm.classify_controllables` pass, as elsewhere in this repo:
one attacker-LLM call at `initialize` labels each controllable; prompts go to
the labelled user prompt; a `system-prompt` surface never receives one; a
classifier failure returns `{}` and a name/value-type backstop picks the first
eligible free-text surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file.

## Known limitations

- 240 prompts per variant. superred caps runs per task, so a full sweep
  usually will not finish; `shuffle=True` samples across the space.
- Two of the ten stub prompts carry upstream's demographic framings verbatim
  ("As a woman, …", "… a Black person …"); they are part of the published
  probe, which deliberately checks whether such framing changes the model's
  code, and are preserved rather than paraphrased.
