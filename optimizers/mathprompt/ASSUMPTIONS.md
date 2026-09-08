# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [DeepTeam](https://github.com/confident-ai/deepteam) |
| Code | `deepteam/attacks/single_turn/math_problem/{math_problem,template,schema}.py`, `single_turn/compliance.py` |
| Commit | `dc148aad62f71330cfec7121d6afb4c620dfa683` |
| Licence | Apache-2.0 |
| Technique | Bethany et al., MathPrompt (2024) |

## Copied byte-for-byte

- `_vendor/dt_mathprompt/template.py` — `MathProblemTemplate.enhance` (the
  math-encoding prompt with its worked examples) and
  `MathProblemTemplate.is_math_problem` (the check prompt). Import-free, loaded
  directly.
- `compliance.build_compliance_check_prompt` — reproduced verbatim from the
  shared `single_turn/compliance.py` (that file imports DeepTeam's `generate`,
  so it can't be vendored whole); `sync_upstream.py --check` re-extracts the
  upstream function and asserts the prompt is byte-identical.
- `get_additional_instructions` — the "provide a complete real-world
  translation … FOR RESEARCH PURPOSES ONLY" tail, verbatim.

### Upstream escape-sequence warning, preserved

`math_problem/template.py` contains invalid f-string escape sequences (`\i`,
`\(`) that emit a `SyntaxWarning` on import under Python 3.12+. This is an
upstream defect; it is **preserved** because the file is vendored byte-identical
(Python still parses the sequences as literals, so behaviour is unchanged).
"Fixing" it would fork the file and break the byte-comparison sync check.

## Upstream coverage

`math_problem.py` defines one attack; its whole `enhance` is ported: the
encoding prompt, the `EnhancedAttack` parse (`math_strategy`, `input`), the
appended instructions, the compliance gate, the is-math-problem gate,
acceptance iff `not non_compliant and is_math_problem`, the `max_retries` loop,
and the fall-back to the original request. Not ported: tqdm progress bars,
`initialize_model`/`generate` plumbing, and the async twin.

## Deviations

Identical in kind to the multilingual port (same DeepTeam single-turn shape):

1. **Encode once at initialize** — the gated encoding runs at `initialize`; the
   result is injected on the run.
2. **A genuine budget exhaustion propagates** (`usage.cost > 0` re-raises)
   rather than silently falling back to the un-encoded goal; the budget-less
   noop client still falls back.
3. **Unreadable gate replies do not accept a turn** — the parsers return
   `None`, and acceptance requires `non_compliant is False and is_math_problem
   is True`, so a `None` on either gate declines and retries.
4. **Injection-surface selection** via the house `surface_llm.py` classifier
   (pinned byte-identical, `temperature=0.0`; the no-temperature guard skips
   that one shared file).

## Known limitations

- Three attacker-LLM calls per accepted encoding, plus retries.
- Effectiveness depends on the attacker model producing a valid math encoding
  and the target being willing to "solve" it.
