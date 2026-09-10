# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [garak](https://github.com/NVIDIA/garak) |
| Path | `garak/probes/ansiescape.py`, `garak/resources/ansi.py` |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |

## Copied byte-for-byte

- `_vendor/garak_ansi/ansi.py` — garak's `resources/ansi.py`, byte-identical.
  It has no imports of its own, so it vendors verbatim and the payload
  constants (`LIVE_PAYLOADS`, `ESCAPED_PAYLOADS`) are upstream's exactly.

`probes/ansiescape.py` itself cannot be vendored: it imports garak's probe
machinery (`garak.probes`, `garak._config`). Its four prompt-stub lists —
`ASKS`, `HIGH_LEVEL_TASKS`, `REPEAT_STUBS`, `UNESCAPE_STUBS` — are extracted
into `data.json` by AST, and `scripts/sync_upstream.py` re-extracts and
compares them. The construction loops are reproduced in `build_prompts()`,
pinned by tests that rebuild the expected list from the same constants.

## Upstream coverage

`ansiescape.py` defines three probes:

| Upstream | Ported |
| --- | --- |
| `AnsiEscaped` | yes — `variant="escaped"`, 51 prompts |
| `AnsiRaw` | yes — `variant="raw"`, 58 prompts |
| `AnsiRawTokenizerHF` | **no** — it walks a HuggingFace tokenizer's vocabulary looking for tokens that carry escape sequences (`supported_generators = {"huggingface.Model"}`). That needs the target's tokenizer, which a black-box chat target does not expose; it is a model-file inspection, not an attack driven through a prompt. |

The `ansiescape.Escaped` / `ansiescape.Raw` detectors are not ported: in
superred, judging whether a response contains a control sequence belongs to
the `SecurityClaim`, as with every other optimizer here.

## Deviations

### 1. Goal-agnostic

Upstream's prompts are fixed, so nothing embeds the task `Goal`. It is
accepted and passed to the surface classifier, but never appears in a payload
— the same shape as the existing `divergent_repetition` and
`demo_prompt_list` optimizers. A test pins this.

### 2. One prompt per run

Upstream hands its whole prompt list to its own harness. superred drives one
attempt per run, so the list is swept one prompt per run and the optimizer
reports `done` when it is exhausted. `shuffle` (seeded) is offered because
superred runs may be capped before the sweep finishes; upstream file order is
the default.

### 3. Injection-surface selection (superred adaptation)

Upstream sends prompts straight to its generator. superred targets expose
several surfaces, so this module uses the shared
`surface_llm.classify_controllables` pass — one attacker-LLM call at
`initialize` that labels each controllable `user-prompt` /
`content-injection` / `system-prompt`. Prompts go to the labelled user prompt;
a `system-prompt` surface never receives one; a classifier failure returns
`{}` and a name/value-type backstop picks the first eligible free-text
surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file.

## Known limitations

- Whether an escape sequence is *dangerous* depends entirely on what renders
  the output. A model emitting `\x1b[32m` matters in a terminal and not in a
  browser. The optimizer only elicits; the claim decides.
- `variant="raw"` puts real `ESC` bytes into the payload. A transport that
  strips or mangles control characters will blunt it — prefer `escaped` there.
