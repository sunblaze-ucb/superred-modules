# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [renqibing/CodeAttack](https://github.com/renqibing/CodeAttack) |
| Code | `src/codeattack/data_preparation.py` (`CodeAttack.wrap`), `prompt_templates/*.txt` |
| Commit | `6777ed03b11567a91908f23bb8ccedca6103772c` |
| Licence | MIT |
| Paper | Ren et al., arXiv:2403.07865 |

## Copied byte-for-byte

- `data/upstream/code_python_{list,stack,string}.txt` — the three Python
  templates, identical to upstream.
- `_wrap_input` — upstream's `CodeAttack.wrap` tokenisation: `python_list`
  splits on whitespace and appends in order; `python_stack` splits on
  whitespace/hyphens, reverses, and appends (and splits a single token into
  characters); `python_string` embeds the whole query.

`scripts/sync_upstream.py --check` verifies the templates are byte-identical and
that `render()` reproduces upstream's own shipped `data_python_{list,string}
_full.json` byte-for-byte (520/520 each).

## Deviations

### 1. No trailing newline on the wrapped input

Upstream's `wrap` appends a `\n` to the tokenised block, but its own shipped
`data_python_{list,string}_full.json` do **not** carry that newline — the repo's
code and materialised data disagree by one `\n`. This module matches the
**shipped data** (the prompts CodeAttack actually sends), which is byte-identical
to the current template filled without the trailing newline.

### 2. Stack uses the current template; its shipped data is stale

`data_python_stack_full.json` was materialised from an older stack template
(`my_queue` / `deque` with a different structure) and no longer matches the
repo's current `code_python_stack.txt` (`my_stack = deque()`). This module
vendors and fills the **current** template — faithful to the repo's present
state — so `python_stack` output intentionally differs from the stale shipped
stack data. `python_list` / `python_string` have no such drift and are
byte-verified against their shipped data.

### 3. All eight upstream templates are ported

`python_list`, `python_stack`, `python_string`, their `*_plus` counterparts
(which demand 100-word paragraphs instead of terse steps), plus `C_string` and
`go_string`. The non-Python scaffolds carry a hardcoded example rather than a
`{wrapped_input}` slot, so upstream fills them by replacing a fixed 0-indexed
line (`replacement_line` 10 and 17); that line-replacement is reproduced
byte-faithfully, including upstream's append-if-past-EOF branch.

### 5. One prompt per run, no detector

CodeAttack is single-shot (no optimisation loop, no attacker LLM), so the
optimizer injects one rendered prompt and ends the run. Judging belongs to the
`SecurityClaim`, so the upstream code-output detector is not ported.

### Injection-surface selection (superred adaptation)

Upstream aims a single prompt at a chat box; superred targets expose arbitrary
named surfaces. Following the house pattern used by `muzzle`, `chord_xthp`,
`eia_agent`, `gepa_agentic`, `poisonedrag` and `agentvigil_websentinel`, this
module ships a byte-identical copy of the shared `surface_llm.py` and calls
`classify_controllables` once at `initialize`: the attacker's own LLM reads
each surface's description and says which one is the user's prompt, so the
payload is not spent on a content surface that merely fired first.

The classifier returns `{}` on any failure (no budget, malformed reply), and a
surface the target raises at run time without listing it is unknown to the
classification; both fall back to the previous name/value-type backstop, so
behaviour is unchanged when no LLM is available.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules that ship the
classifier resolve this by not carrying the guard at all; this module keeps the
guard and skips only that one pinned shared file, so the guard still covers all
first-party code here.

## Known limitations

- A goal containing a double-quote would break the generated Python string
  literal; upstream has the same limitation (it interpolates the query directly),
  preserved here rather than silently escaping.
- Breadth comes from the three variants, not many prompts.
