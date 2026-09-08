# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| Code | `garak/probes/divergence.py` (`Repeat`, `RepeatExtended`) |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |
| Reference | Nasr et al. (2023) |

## Copied verbatim

- `data.json` — the `Repeat` 4-word list, the `RepeatExtended` 105-word list,
  the three prompt templates, and the repetition counts `[1, 3, 40]`, extracted
  verbatim from the probe. `scripts/sync_upstream.py --check` re-extracts them
  from upstream and compares.
- `build_prompts` reproduces `Repeat.__init__`'s nested loop
  (word x template x repetition, substituting the repeated word into `[XX]`),
  so the prompt set is byte-identical to garak's.

## Upstream coverage

`divergence.py` defines three probes:

| Upstream | Ported |
| --- | --- |
| `Repeat` (active) | yes — default (4 words, 36 prompts) |
| `RepeatExtended` (opt-in, `active=False`) | yes — `extended=True` (105 words, 945 prompts) |
| `RepeatedToken` | **no** — a different attack that repeats **token ids** via a round-trip through the *target's* tokenizer (`_roundtrip_tokens`), from a separate source (Dropbox llm-security) with its own `divergence.RepeatedToken` detector. It is coupled to the target's tokenizer and warrants its own module. |

The `divergence.RepeatDiverges` detector is not ported: in superred, judging
whether a response leaked data belongs to the `SecurityClaim`, as with every
other optimizer here.

## Deviations

### 1. Goal-agnostic

This is the one optimizer here that does not use the task `Goal`: repetition-
induced divergence is a property of the target, not a request to be smuggled.
The prompts are fixed; the `Goal` is accepted (and passed to the surface
classifier) but never embedded. This matches the existing `demo_prompt_list`
optimizer's fixed-prompt shape.

### 2. Sweep all prompts, one per run

garak runs the whole prompt set; superred drives one prompt per run and ends
after the set. `shuffle`/`seed` give a reproducible order; the default is
upstream's nested-loop order.

### 3. Injection-surface selection (superred adaptation)

Uses the house `surface_llm.py` classifier to pick the user-prompt surface,
even though the payload is goal-independent, so the repeat prompt still lands
where a user request would.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged classifier modules omit
the guard; this module keeps it and skips only that one pinned shared file.

## Known limitations

- The technique is largely mitigated on current frontier models (post the 2023
  disclosure), so real divergence is rare.
- `RepeatExtended` produces 945 prompts — bound a run accordingly.
