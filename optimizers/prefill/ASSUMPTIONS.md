# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard) |
| Path | `AIG-PromptSecurity/deepteam/attacks/single_turn/prefill/` |
| Commit | `dd6bd54655c9ff5fb7351f4299b56916f09ec6da` |
| Licence | Apache-2.0, with the NOTICE attribution requirement below |

AI-Infra-Guard's NOTICE imposes a **mandatory attribution requirement** under
Apache-2.0 §4(d): derivative works must state "Based on Tencent Zhuque Lab
AI-Infra-Guard" in their documentation and link the repository. That is
satisfied in `README.md`, `NOTICE`, `LICENSES/NOTICE.md` and the module
docstring, with upstream's NOTICE reproduced verbatim at
`LICENSES/AI-Infra-Guard-NOTICE.txt`. **It propagates to anything
redistributing this module.**

## Copied byte-for-byte

- `_vendor/aig_prefill/template.py` — `DEFAULT_PREFILL`, byte-identical to
  upstream. `scripts/sync_upstream.py --check` re-downloads and diffs it.

## Upstream coverage

`prefill.py` defines one attack and it is fully ported:

| Upstream | Ported |
| --- | --- |
| `enhance` — `f"{attack}\n\n{self.prefill}"` | yes, as `render()`; the sync script compares the f-string's literal parts against upstream's so the separator cannot drift |
| `prefill` constructor parameter | yes, same name, same default |
| `DEFAULT_PREFILL` | yes, vendored byte-identical |
| `weight` | **no** — DeepTeam's sampling weight, used by its own attack scheduler to choose among attacks. superred's `Controller` selects the optimizer, so the parameter has no analogue here. |
| `get_name` | **no** — DeepTeam registry plumbing, not attack behaviour |

## Deviations

### 1. Upstream primes the user turn, not the assistant turn

The published prefilling attack puts the compliant opening into the
*assistant* turn, so the model literally continues its own sentence.
AI-Infra-Guard's implementation appends it to the *user* message instead,
which is what this ports — a weaker but transport-independent form that works
against any target exposing a text surface. superred has no cross-target way
to write a partial assistant turn, so upstream's form is also the portable
one.

### 2. Injection-surface selection (superred adaptation)

Upstream returns a string and leaves delivery to its own harness. superred
targets expose several surfaces, so this module uses the shared
`surface_llm.classify_controllables` pass — one attacker-LLM call at
`initialize` that reads each controllable's own description and labels it
`user-prompt` / `content-injection` / `system-prompt`. The payload goes to the
labelled user prompt; a surface labelled `system-prompt` never receives it. If
the classifier fails or is unavailable it returns `{}` and a name/value-type
backstop picks the first eligible free-text surface, so the attack still
lands.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The merged modules shipping the
classifier resolve this by omitting the guard; this module keeps the guard and
skips only that one pinned shared file, so it still covers all first-party
code.

## Known limitations

- A single fixed prompt: nothing adapts, and a target that refuses the goal in
  plain text will refuse this too. It is a baseline, and cheap to run.
- The priming text is English; a non-English target may not continue from it.
