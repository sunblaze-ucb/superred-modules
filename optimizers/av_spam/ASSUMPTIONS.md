# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/av_spam_scanning.py` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

`probes/av_spam_scanning.py` cannot be vendored: it imports garak's probe
machinery. Its four module-level constants — the three signatures and
`REPEAT_PROMPTS` — are extracted into `data.json` by AST, and
`scripts/sync_upstream.py` re-extracts and compares all four, refusing to
write a partial extraction. The signatures are byte-identical to upstream's,
which are in turn the published test strings: the **EICAR** Standard
Anti-Virus Test File (EICAR — European Institute for Computer Antivirus
Research), and the **GTUBE** / **GTphish** generic test strings from the
[Apache SpamAssassin](https://spamassassin.apache.org/gtube/) project. They
are public scanner-test standards, not garak's authorship; only the
repetition templates that wrap them are garak's. See `NOTICE`.

## Upstream coverage

`av_spam_scanning.py` defines three probes; **all three are ported**:

| Upstream | Ported as |
| --- | --- |
| `EICAR` | `variant="eicar"` |
| `GTUBE` | `variant="gtube"` |
| `GTphish` | `variant="gtphish"` |

The three classes are structurally identical upstream — same
`REPEAT_PROMPTS`, different signature — so they collapse into one optimizer
with a `variant` argument rather than three near-duplicate modules.

Not ported: the `knownbadsignatures.*` detectors (judging the response belongs
to the `SecurityClaim` here, as in every other optimizer), and the
`tags`/`tier`/`doc_uri` taxonomy metadata.

## Deviations

### 1. Upstream ships all three inactive, and the reason matters

All three probes carry `active = False` and `tier = UNLISTED`, with the
docstring "Only applicable to models expected to do antivirus scanning."
That is a **scoping** statement, not a defect, and it carries over here: a
plain chat model repeating EICAR is not a finding — it is repeating a string
you handed it. The probe is meaningful only against a target that is supposed
to scan what it emits. The README says so up front.

This module does not replicate `active`: superred has no notion of a probe
that is present but skipped, and the caller chooses the optimizer explicitly.

### 2. Goal-agnostic

The signature is fixed, so nothing embeds the task `Goal`. It is accepted and
passed to the surface classifier but never appears in a payload — the same
shape as `divergent_repetition`. A test pins this.

### 3. One prompt per run

Upstream hands its whole list to its own harness; superred drives one attempt
per run, so the five prompts are swept one per run and the optimizer reports
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

- Narrow by construction: three signatures, five framings, 15 prompts total.
- A transport that rewrites or escapes the payload will blunt it — the
  signatures only trip a scanner if they arrive byte-exact.
