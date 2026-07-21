# Bijection Optimizer — design notes

This document records the design decisions, paper alignment, and
deliberate departures for the Bijection learning jailbreak optimizer.

## Reference

- Paper: Brian R. Y. Huang, Maximilian Li, Leonard Tang, "Endless
  Jailbreaks with Bijection Learning," arXiv:2410.01294 (ICLR 2025).
- Code: `haizelabs/bijection-learning`
  (https://github.com/haizelabs/bijection-learning), specifically
  `that_good_infra/transformations.py` (bijection construction,
  encoding, decoding) and `run_bijection_attacks.py` (prompt
  construction).

## Algorithm summary

Bijection learning is a black-box, language-model-agnostic jailbreak
that randomizes a bijective character map (English ↔ "Language Alpha")
and teaches it to the target in-context. The harmful query is
encoded under the same map; capable models follow the encoding into
the encoded answer space, sidestepping refusal training.

Per attempt:

1. Sample a fresh random bijection at fixed complexity ("dispersion"
   = `26 - fixed_size`).
2. Render a packed prompt: a teaching intro that names the language
   and exposes the alphabet table, plus `num_teaching_shots` English
   ↔ encoded teaching pairs, plus a multi-turn indicator, plus the
   bijection-encoded harmful query.
3. Submit it to the target. The paper uses best-of-N sampling — keep
   sampling fresh bijections until one succeeds or budget runs out.

Optionally, the response can be decoded back to English and judged.

## Mapping to superred

| Paper construct | Superred surface |
| --- | --- |
| User query (encoded harmful intent) | `ControllablePreCallEvent` for `user_message` (or any single text controllable) |
| Teaching intro + alphabet table | (a) packed into `user_message`, or (b) split into `system_prompt` PreCall when that controllable is in the optimizer's scope (FlipAttack-style two-channel split) |
| Target response (encoded) | `ObservableEvent` on the trajectory; PostCall `answer` as fallback |
| Best-of-N stopping rule | `RunEndEvent.evaluation.success` (controller-supplied judge) |
| Best-of-N budget `N` | `max_attempts` |

One superred run = one bijection trial (single-turn injection). Across
runs the optimizer samples a fresh random bijection at the configured
complexity. No reflection loop, no LLM-driven attacker — bijection
construction is deterministic given the random seed.

## Adversarial information-access settings

The optimizer operates in all four settings without a setting knob —
the framework's scope filter and `include_feedback` flag select which
information surfaces are visible. The optimizer's behaviour on the
attack-construction path is **response-blind by design** because the
bijection is randomized, not adapted; this matches the paper's
best-of-N formulation exactly.

1. **User query only.** Sample fresh bijections each run; never declare
   success (no judge signal). Stop when `max_attempts` is exhausted.
2. **User query + feedback.** Sample fresh bijections; early-stop on
   `RunEndEvent.evaluation.success` (paper's stopping rule).
3. **User query + responses.** Same as setting 1 — bijections are
   sampled, not adapted, so target responses don't drive optimizer
   state. The trajectory is still observed for the no-signal guard.
4. **User query + responses + feedback.** Same as setting 2.

In settings 1 and 3 (no judge), `max_no_signal_runs` may be set to
guard against truly blind loops where neither response nor evaluation
ever reaches the optimizer. Disabled by default (matches GOAT, GEPA).

## Faithful upstream ports

- **Letter bijection construction** (`_make_letter_bijection`):
  verbatim port of `BijectionLanguage.create_bijective_mapping`.
  `random.sample` selects fixed letters; the rest are shuffled.
- **Digit bijection construction** (`_make_digit_bijection`):
  verbatim port of `DigitBijectionLanguage.create_bijective_mapping_digits`.
  Numbers are sampled without replacement from `[10^(n-1), 10^n)` so
  every encoding has exactly `num_digits` digits.
- **Letter encode/decode**: our `Bijection.encode` / `Bijection.decode`
  (letter branch) both port upstream's `permute_string`. Both branches
  lowercase the input unconditionally — upstream's `permute_string` is
  a single function reused for both directions and lowercases; we
  mirror that exactly so a downstream judge wrapper feeding raw model
  output through `Bijection.decode` gets the upstream-equivalent text.
  Pinned by `test_letter_decode_lowercases_input_matching_upstream`.
- **Digit encode**: verbatim port of `DigitBijectionLanguage._f`.
  Delimiter is inserted before each substituted numeric token, never
  before identity-mapped letters.
- **Digit decode**: verbatim port of `permute_digits_to_string`,
  including the upstream-bug behaviour where the prepended delimiter
  can leak as a leading character in the decoded output. We retain
  this exactly; downstream code (judge, caller) is expected to call
  `.strip()` before comparison, matching upstream's pipeline. Tests
  document this with `test_digit_roundtrip_with_double_space_delim_leaks_leading_delim`.
- **Teaching intro template** (`render_teaching_intro`): verbatim from
  the letter / digit branch of `full_attack_construction`.
- **`MULTITURN_INDICATOR`**: verbatim from `run_bijection_attacks.py`.
- **Default `max_attempts = 6`**: matches the run script's default
  budget.
- **Default `num_teaching_shots = 10`**: matches the run script's
  default.
- **Default `num_digits = 2` and `digit_delimiter = "  "` (two
  spaces)**: paper's main-table digit configuration (Table 1).
- **Default `fixed_size = 10`** (dispersion 16): matches the
  Sonnet-optimal digit setting from Table 1. Used as the no-observable
  / unknown-model fallback when the new auto-tune resolver finds no
  Table 1 entry.

## Threat-model fidelity: full use of in-scope capabilities

Per the framework-wide guidance to *"automatically utilize the full
scope of capabilities granted by the threat model, even when slightly
beyond the original publication"*, the optimizer reads everything in
scope that has a natural attack vector for a static-encoding attacker
and degrades silently when a surface is absent. Concretely:

- **`model` static observable → auto-tune codomain & dispersion per
  paper Table 1.** When `bijection_type` and/or `fixed_size` are left
  at their default (`None`), `initialize` looks up the in-scope
  `model` observable in `_MODEL_OPTIMAL_DEFAULTS` and applies the
  paper's per-target optimum. Mirrors FlipAttack's `victim_llm`-driven
  Pliny-template selection. Resolution priority:
  1. Explicit constructor override (`bijection_type="letter"`, etc.).
  2. Paper Table 1 lookup (AdvBench-50 sub-table) against the in-scope
     `model` observable. Table 1 reports exactly five models:
     `claude-3-5-sonnet` → digit/10 (dispersion 16), `claude-3-opus` →
     digit/10 (dispersion 16), `claude-3-haiku` → letter/10 (dispersion
     16), `gpt-4o` → letter/18 (dispersion 8), `gpt-4o-mini` →
     letter/18 (dispersion 8). Models the paper does not evaluate
     (e.g. GPT-4-Turbo, plain "Claude 3 Sonnet") are intentionally
     *not* in the lookup table — fabricating a Table 1 value for a
     model the paper never tested would misrepresent the paper, so
     those model ids fall through to the main-table fallback instead.
  3. Paper main-table fallback (`digit`, `fixed_size=10`) when no
     `model` observable is present or it doesn't match a Table 1 row.
  When scope matches the paper's threat model the optimizer lands on
  the paper's per-target optimum; when scope grants more (an unknown
  model identifier) the optimizer keeps the paper main-table strong
  default; when scope grants less (no `model` observable) the
  optimizer is exactly the paper's main configuration. Pinned by
  `TestModelObservableAutoTune`.

- **`system_prompt` writable controllable → two-channel split.** When
  the controller's scope grants write access to `system_prompt`, the
  teaching intro + alphabet table goes to the system channel and the
  teaching shots + encoded query stay in `user_message` (FlipAttack
  pattern). Already shipped pre-PR-feedback.

- **`system_prompt_readable` (and other non-`model` static
  observables) → not consumed.** Bijection's attack vector is a
  deterministic, content-blind packed prompt: a teaching intro +
  alphabet table + encoded query. There is no LLM-driven attacker that
  could naturally consume readable context (contrast GEPA / GOAT /
  AutoDAN-Turbo, where the attacker prompt absorbs context cleanly).
  Two paths were considered and rejected for v1:
  1. Inject `system_prompt_readable` content as a "for your context,
     the assistant has been told: …" preamble. Rejected: would alert
     the assistant that its instructions have leaked, plausibly
     triggering the very defensive paths the bijection encoding is
     designed to skip.
  2. Encode `system_prompt_readable` content under the bijection as
     additional teaching pairs. Rejected: paper's teaching corpus is
     bland prose chosen specifically to be unrelated to the harmful
     query; injecting the system prompt would be off-distribution and
     plausibly degrade the teaching effect.
  Pinned by `test_non_model_observables_do_not_affect_resolution`.

- **Response observables → consumed only as a no-signal heartbeat.**
  Best-of-N is non-adaptive by paper definition, so target responses
  do not feed back into the next attempt's bijection. We still drain
  the trajectory to detect "blind" runs (no response *and* no
  evaluation) so `max_no_signal_runs` can trip cleanly.

## Deliberate departures

### Single-turn packed prompt (vs. upstream's multi-turn assistant-prefilled chat history)

Upstream renders teaching shots as separate user / assistant chat
turns, with the assistant turns pre-filled. Superred has no
assistant-prefill mechanism: the only way to put text in the
assistant role is to actually generate it. Mirroring upstream's
multi-turn structure as a real chat would force one real LLM
generation per teaching shot (default 10) per attempt — an 11×
cost multiplier whose only "value" is a noisy assistant response we'd
discard.

Packed teaching shots inside one user-message block (delimited as
`User: ... Assistant: ...` text) preserve the demonstration
structure at single-turn cost. This mirrors upstream's `multi_turn_format=False` mode in spirit: pack the teaching corpus into a single
text block before the attack turn. We also follow FlipAttack's
single-turn shape, the closest peer in the merged optimizers.

### Codomains supported in v1: `letter` and `digit`

The paper evaluates four codomains: `letter`, `digit`, `tokenizer`,
and `morse`. Table 1 lists `digit` as Sonnet-optimal and `letter` /
`morse` as competitive on weaker models. We ship `letter` and `digit`
in v1: they cover the strongest results, are pure-Python (no
tokenizer model), and the encode/decode logic is small and
verifiable. `tokenizer` and `morse` are out of scope for v1 and can
be added later as a `codomain="tokenizer"` / `"morse"` extension
without touching the optimizer state machine.

### No automatic response decoding in v1

Upstream `run_bijection_attacks.py` decodes the response with the
inverse bijection before judging. Superred's judge contract operates
on raw response strings; we do not decode the response inside the
optimizer because:

1. The judge is supplied by the controller, not configured by the
   optimizer.
2. Bijection encodings are noisy: the model often emits text that's
   partially decoded, partially still encoded, or in plain English
   anyway (the paper's denoising step is itself an open problem).
3. The optimizer's success signal is still
   `RunEndEvent.evaluation.success`; whether the judge sees the
   encoded or decoded string is the controller's call.

A judge wrapper that decodes via `Bijection.decode` before calling
the underlying judge is a one-line composition outside this module.

### Bijection construction is response-blind

The paper's best-of-N is by definition non-adaptive — each attempt
samples a fresh random bijection independent of prior results. We
honour that exactly. Bijection-related "search" (e.g. dispersion
schedules) is out of scope for v1 and should live in a separate
optimizer if needed.

### Default teaching corpus

Upstream uses Paul Graham essay snippets. We ship a small set of
short, deliberately bland English passages of similar shape and
length. Callers can override via the `teaching_corpus` argument to
`render_attack_prompt` (not currently exposed at optimizer level —
keeping the constructor minimal; can be promoted later if needed).

## Constructor surface

| Argument | Default | Purpose |
| --- | --- | --- |
| `bijection_type` | `None` | `"letter"` or `"digit"`; `None` auto-tunes from the in-scope `model` observable (paper Table 1) and falls back to `"digit"` (Sonnet-optimal main-table row) when no recognised `model` observable is in scope. Explicit values always win. |
| `fixed_size` | `None` | Dispersion = `26 - fixed_size`; `None` auto-tunes from the in-scope `model` observable (paper Table 1) and falls back to `10` (Sonnet-optimal `digit` row, dispersion 16) when no recognised `model` observable is in scope. Explicit values always win. |
| `num_digits` | `2` | Encoding length for digit codomain; paper default. |
| `digit_delimiter` | `"  "` (two spaces) | Inserted before substituted numeric tokens; paper default. |
| `num_teaching_shots` | `10` | Teaching pairs in the packed prompt; paper / run-script default. |
| `max_attempts` | `6` | Best-of-N budget; run-script default. |
| `response_observable_names` | `{"response", "model_response", "assistant_response"}` | Trajectory observable names recognised as target replies; matches Crescendo / GEPA / GOAT. |
| `target_controllable_name` | `None` | Lock injection onto a single named controllable (e.g. for non-chatbot targets); when `None`, default heuristics apply. |
| `max_no_signal_runs` | `0` (disabled) | Terminate after this many consecutive runs with no visible response or evaluation. |
| `seed` | `None` | RNG seed for reproducible bijection samples. |

## Module layout

```
src/bijection_optimizer/
  __init__.py          # exports BijectionOptimizer
  bijection.py         # Bijection dataclass + generate_bijection
  prompts.py           # render_teaching_intro / render_attack_prompt
  optimizer.py         # BijectionOptimizer event-state machine
tests/
  test_bijection.py    # 27 tests — construction + encode/decode + RNG
  test_prompts.py      #  9 tests — rendering, intro, shots, encoding
  test_optimizer.py    # 43 tests — state machine, 4 settings, e2e,
                       #             model-observable auto-tune
```

## Test coverage

79 tests total. Coverage:

- Letter and digit bijection construction (alphabet completeness,
  fixed-point counts, validation).
- Letter roundtrip (exact); digit roundtrip with empty delim
  (exact) and with double-space delim (upstream-faithful leaky).
- Letter decode lowercases its input (upstream-faithful;
  `test_letter_decode_lowercases_input_matching_upstream`).
- Encoder corner cases (delimiter only before substituted tokens,
  non-alpha pass-through, lowercasing).
- Determinism: same seed → same mapping; consecutive draws differ.
- Default RNG path: module-level `_DEFAULT_RNG` is a real
  `random.Random` instance; `generate_bijection` without an `rng`
  argument produces a well-formed bijection.
- Prompt rendering: intro inclusion, shot count, teaching-corpus
  looping, encoded-query placement, no-leak of plain English goal.
- Construction validation (invalid attempts, shots, codomain,
  fixed_size).
- **Model-observable auto-tune** (`TestModelObservableAutoTune`):
  no observable → paper main-table fallback; six Table 1 model IDs
  (Claude 3.5 Sonnet, Claude 3.5 Sonnet alias, Claude 3 Opus,
  Claude 3 Haiku, GPT-4o, GPT-4o-mini — the only models the paper's
  Table 1 reports) → expected per-paper codomain / `fixed_size`;
  a model the paper never evaluated (GPT-4-Turbo) → main-table
  fallback, not a fabricated Table 1 value; unknown model identifier
  → fallback; explicit `bijection_type` override wins; explicit
  `fixed_size` override wins; both explicit overrides skip the
  observable lookup; empty / whitespace `model` observable treated
  as absent; non-`model` observables (e.g. `system_prompt`) do not
  affect resolution; `gpt-4o-mini` resolves to its own row rather
  than the `gpt-4o` row despite the substring overlap.
- RunStart resets per-run state and prepares a fresh bijection.
- PreCall: injection on user_message, lock-in to first user
  controllable, single-turn behaviour (one injection per run),
  system_prompt out-of-scope skip, two-channel split when
  system_prompt is in scope, `target_controllable_name` override.
- PostCall: pairing rules; record answer; ignore unrelated.
- RunEnd: success → done, failure → not done, no-eval continues
  until `max_attempts`, fresh bijection per run, deterministic
  with same seed.
- All four adversarial settings.
- `max_no_signal_runs` blind-loop guard (trigger and reset).
- End-to-end ChatbotTarget-style integration (system_prompt then
  user_message loop; with and without system_prompt in scope).
- End-to-end integration against a real
  `Trajectory(filtered_scope=...).filtered`, emitting a real
  `ObservableEvent` and confirming the optimizer drains it through
  the production trajectory plumbing (guards against future drift in
  `FilteredTrajectory.drain` semantics).
