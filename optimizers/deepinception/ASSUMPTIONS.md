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

### 4. Follow-up questions are ported (opt-in)

Upstream's `main` / `further_q` runs send the inception prompt and then keep
asking within the established scene. `follow_ups=True` reproduces that: the
questions land on the same surface that received the scene, so the
conversation continues rather than restarting.

Only the goal-agnostic questions are vendored — those phrased against "the
above goal" / "their goal". The remainder of `res/data_further_q.json` is bound
to one topic (a news story, a bank, a Linux box, a firewall) and cannot follow
an arbitrary goal.

### 5. Detector out of scope

DeepInception's repo scores with its own harmfulness check; in superred judging
belongs to the `SecurityClaim`, so no detector is ported.

### Injection-surface selection (superred adaptation)

Upstream is a single prompt aimed at a chat box; superred targets expose
arbitrary named surfaces. Following the house pattern used by `muzzle`,
`chord_xthp`, `eia_agent`, `gepa_agentic`, `poisonedrag` and
`agentvigil_websentinel`, this module ships a byte-identical copy of the shared
`surface_llm.py` and calls `classify_controllables` once at `initialize`: the
attacker's own LLM reads each surface's description and says which one is the
user's prompt -- the channel a nested-scene jailbreak must arrive on.

The classifier returns `{}` on any failure (no budget, malformed reply), in
which case the previous name/value-type backstop decides instead, so behaviour
is unchanged when no LLM is available. A surface the target raises at run time
without ever listing it also falls back to the backstop rather than being
dropped.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules that ship the
classifier resolve this by not carrying the guard at all; this module keeps the
guard and skips only that one pinned shared file, so the guard still covers all
first-party code here.

## Upstream coverage

`main.py` exposes six experiments. Their prompt material is covered as:

| Upstream experiment | Ported as |
| --- | --- |
| `main` | the README template with the goal in the query slot |
| `multi_scene` | `scene=` — all four swept scene types are vendored (`scenes()`) |
| `abl_c` / `abl_layer` / `abl_fig6_4` | `character_number=` / `layer_number=`, which are exactly the parameters those ablations sweep |
| `further_q` | `follow_ups=True` (goal-agnostic questions only, see deviation 4) |

The per-experiment `res/data_*.json` files themselves are upstream's fixed
evaluation set (each record bakes in one behaviour), not a goal-driven
template, so the parameters and question set are ported rather than the rows.

## Known limitations

- A single fixed template: breadth comes from the `scene` / layer parameters,
  not from many prompts. Frontier models increasingly refuse the canonical form.
- The goal is inserted verbatim mid-sentence ("propose a step to {goal}
  against..."), so a goal phrased as a full sentence reads slightly awkwardly —
  matching how upstream's own weaving works, minus the re-casing.
