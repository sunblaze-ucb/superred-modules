# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/sysprompt_extraction.py`, `garak/data/sysprompt_extraction/attacks.json` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

- `_vendor/garak_sysprompt/attacks.json` — garak's own data file, the 28
  extraction templates across 9 categories, byte-identical.
  `scripts/sync_upstream.py --check` re-downloads and diffs it.

## The one substantive deviation: system prompts are the target's, not planted

This is the load-bearing decision, so it is stated plainly rather than buried.

`garak`'s `SystemPromptExtraction` probe does two things. First, it **plants a
system prompt** it controls: it loads two HuggingFace datasets
(`garak-llm/drh-System-Prompt-processed`, `garak-llm/tm-system_prompt`) via the
`datasets` package, subsamples 50, and writes each into a `system` turn of a
`Conversation`. Second, it sends an **extraction template** in the following
`user` turn and checks whether the planted prompt comes back.

The planting is garak's **test harness**, not the attack. garak controls both
the generator and the ground truth, so it fabricates a secret to see if it
leaks. A superred target already **has** a real system prompt — the one under
test — so there is nothing to plant. The attack that transfers is the set of
extraction templates; those are ported in full and byte-identical.

Consequences, all deliberate:

- **No `datasets` / HuggingFace dependency.** The module depends on `superred`
  alone. The two dataset sources are not vendored because they are decoys, not
  part of the technique.
- **No `Conversation` with a synthetic system turn.** The template is injected
  into the target's user surface (chosen by the classifier); the target
  supplies its own system prompt.
- **`system_prompt_subsample` / `follow_prompt_cap` are not ported** — they cap
  the *planted* corpus, which no longer exists here. The 28 templates are swept
  in full; `shuffle`/`seed` cover the "sample the space" purpose superred needs
  when a run is capped.

This is exactly the port that could be under-done by dropping work, so to be
explicit: the 28 attack templates — the whole `attacks.json` — are ported
verbatim; what is dropped is only garak's mechanism for giving *itself*
something to extract.

## Upstream coverage

| Upstream element | Ported |
| --- | --- |
| `attacks.json` (28 templates, 9 categories) | yes — byte-identical, flattened in file order |
| `_load_garak_attacks` (flatten the categories) | yes, as `load_attack_templates()` |
| `_load_system_prompt_datasets` (HF planting) | no — harness, not attack (see above) |
| `_generate_attempts` (system+user `Conversation`) | reshaped: the template goes to the target's user surface |
| `sysprompt_extraction.PromptExtraction` detector | no — judging the leak belongs to the `SecurityClaim` |

## Deviations (superred adaptations)

### Injection-surface selection

Shared `surface_llm.classify_controllables` pass: one attacker-LLM call at
`initialize` labels each controllable; templates go to the labelled user
prompt; a `system-prompt` surface never receives one — which matters here more
than usual, since overwriting the system prompt would destroy the very thing
the attack is trying to read. A classifier failure returns `{}` and a
name/value-type backstop picks the first eligible free-text surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file.

## Known limitations

- 28 single-shot templates: no multi-turn escalation, no adaptation to the
  target's refusals. A baseline sweep, not a strong attack.
- The templates are English and assume a chat-style system/user split; against
  a target with a different prompt structure they may not apply.
