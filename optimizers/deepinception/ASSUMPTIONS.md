# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [tmlr-group/DeepInception](https://github.com/tmlr-group/DeepInception) |
| Template | README "Ready-to-Use Prompt" (general template) |
| Commit | `fc5689e76c4dd87a14babf1eadf890532f2b6880` |
| Licence | MIT (Copyright (c) 2023 TMLR Group) |
| Paper | Li et al., *DeepInception* (arXiv:2311.03191) |

## Copied byte-for-byte

- `TEMPLATE` in `template.py` — the README general template, with its bolded
  values (`scene`, `character_number`, `layer_number`, `query`) exposed as
  format slots, including upstream's exact whitespace (the double space after
  the second scene, the mid-prompt line breaks). `scripts/sync_upstream.py
  --check` renders it with the README's worked values and compares to the
  README instance byte-for-byte; a test pins the same.

## Deviations

### 1. Template from the README, not the data file

The repo's `res/data_main.json` ships 50 pre-built inception prompts (one per
harmful behaviour), but they carry per-record authoring noise ("to against" vs
"against", rephrased goals) and each embeds a fixed behaviour — so they are not
a clean template and cannot pursue an arbitrary goal. The README's stated
general template *is* the canonical, parameterised form, so that is what is
ported, with the goal in the query slot. This is the faithful, goal-driven
reading; the data file is upstream's evaluation set, not the method.

### 2. Goal fills the query slot, as-is

The goal replaces the README's bolded query ("show how to hack into a Linux
computer"). Upstream's own data file lowercases the first letter and drops the
trailing period when weaving the goal in; this module inserts the goal verbatim,
since a superred `Goal` is authored as the objective and re-casing it would be a
silent edit. The scaffolding is unchanged.

### 3. One deterministic scene per task

DeepInception is a single-shot template (no optimisation loop, no attacker LLM),
so the optimizer injects one rendered prompt and ends the run. `scene`,
`character_number` and `layer_number` are exposed for the paper's variants but
default to the README values.

### 4. No follow-up questions

Upstream's `main`/`further_q` experiments send follow-up questions after the
inception prompt to continue the jailbreak. This module ports the core
single-prompt method; multi-turn continuation is out of scope (crescendo/FITD
cover adaptive multi-turn in superred).

### 5. Detector out of scope

DeepInception's repo scores with its own harmfulness check; in superred judging
belongs to the `SecurityClaim`, so no detector is ported.

## Known limitations

- A single fixed template: breadth comes from the `scene` / layer parameters,
  not from many prompts. Frontier models increasingly refuse the canonical form.
- The goal is inserted verbatim mid-sentence ("propose a step to {goal}
  against..."), so a goal phrased as a full sentence reads slightly awkwardly —
  matching how upstream's own weaving works, minus the re-casing.
