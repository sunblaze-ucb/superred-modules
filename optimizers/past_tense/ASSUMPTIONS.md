# Assumptions and deviations

**Based on Tencent Zhuque Lab AI-Infra-Guard
(https://github.com/Tencent/AI-Infra-Guard)** — required by that project's
NOTICE under Apache-2.0 Section 4(d).

## Upstream

| | |
| --- | --- |
| Project | [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard) |
| Code | `AIG-PromptSecurity/deepteam/attacks/single_turn/past_tense/{past_tense,template}.py` |
| Commit | `dd6bd54655c9ff5fb7351f4299b56916f09ec6da` |
| Licence | Apache-2.0 (with a mandatory attribution requirement) |
| Technique | Andriushchenko & Flammarion (2024) |

### Why this source and not the authors' repository

The method's origin, [tml-epfl/llm-past-tense](https://github.com/tml-epfl/llm-past-tense),
ships **no licence file** and is therefore all-rights-reserved. Tencent's
AI-Infra-Guard has an Apache-2.0 implementation of the same method against
DeepTeam's attack interface, and that is what this module ports. (garak also
ships the technique, but only as a fixed corpus of pre-rephrased prompts, which
cannot pursue an arbitrary superred `Goal`; the AI-Infra-Guard version
reformulates dynamically.)

## Copied byte-for-byte

- `_vendor/aig_past_tense/template.py` — `PAST_TENSE_PROMPT` and
  `FUTURE_TENSE_PROMPT`, the reformulation instructions with their three
  few-shot examples. The file has no imports and is loaded directly.

`scripts/sync_upstream.py --check` re-downloads it for a byte comparison.

## Upstream coverage

`past_tense.py` defines one attack; its whole `enhance` is ported:

| Upstream behaviour | Ported |
| --- | --- |
| `tense="past"` reformulation | yes |
| `tense="future"` reformulation | yes |
| `tense="present"` control (returns the request unchanged, no LLM call) | yes |
| `max_retries` loop over non-empty reformulations | yes |
| strip surrounding quotes (`.replace('"', '').strip()`) | yes |
| fall back to the original request when every retry is empty | yes |

Not ported: `_generate_text`/`initialize_model` model plumbing (the framework
injects `self.llm`), and the DeepEval `ReformulatedRequest` pydantic schema —
upstream's `enhance` returns the raw text, not the schema, so no structured
parse is needed.

## Deviations

### 1. Reformulate once at initialize

Upstream's `enhance` is called once to produce the attack string. superred
drives attempts through the event loop, so the reformulation happens at
`initialize` and the same tense-shifted prompt is injected on the run.

### 2. A genuine budget exhaustion propagates

Upstream retries on any failure and falls back to the original request.
This module keeps that, **except** for a `BudgetExhaustedError` with
`usage.cost > 0`, which is re-raised: silently sending the un-rephrased goal
after the attacker ran out of money would report a run the target never
actually defended. The budget-less noop client raises the same error with
nothing spent and still falls back.

### 3. Injection-surface selection (superred adaptation)

Follows the house pattern (`muzzle`, `chord_xthp`, `eia_agent`, `gepa_agentic`,
`poisonedrag`, `agentvigil_websentinel`): a byte-identical `surface_llm.py`,
with `classify_controllables` at `initialize` picking the user-prompt surface.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged classifier modules omit
the guard; this module keeps it and skips only that one pinned shared file.

## Known limitations

- One reformulation per task: there is no search over multiple tense variants.
- Effectiveness depends on the attacker model's willingness to reformulate; a
  model that itself refuses the reformulation falls back to the plain goal.
