# Assumptions and deviations

## Upstream

| | |
| --- | --- |
| Project | [HackAPrompt](https://github.com/PromptLabs/hackaprompt) |
| Path | `hackaprompt/evaluator.py` (`Level*Evaluator`), `hackaprompt/utils.py` (filters) |
| Commit | `f0a035a` |
| Paper | Schulhoff et al., EMNLP 2023 (arXiv:2311.16119) |
| Licence | MIT (© 2023 hackaprompt team) |

## Ported byte-for-byte (level defenses)

`src/hackaprompt_target/_vendor/levels.py` reproduces, verbatim:

- the 11 level `format_prompt` templates (`Level0Evaluator` … `Level10Evaluator`);
- the input filters `contains_blacklisted_word`, `contains_pwned_letters`,
  `remove_non_emoji`.

Every level's prompt was verified **byte-identical** to upstream (a differential
check ran each level's `prepare` — template + filter/escape — against upstream's
own `format_prompt` + filters across varied inputs). Level 8's large template is
stored via `repr` so trailing whitespace survives exactly. Kept under `_vendor`
(ruff-excluded) so the upstream text is never reformatted.

## Ported logic (per-level orchestration)

`target.py` reproduces each level's `evaluate` body — the pre-model filter/escape,
the model call, and Level 6's two-step (a story turn, then the defensive
self-check turn) — minus the completer. Faithful details preserved:

- **Level 2 secret key.** Upstream generates a fresh `random_alphanumeric(k=6)`
  key per game session. The target re-arms the key **per run** (not per task), so
  an attacker cannot carry a memorized key across optimizer iterations; an optional
  `seed` makes it reproducible. The effective key is recorded in the result and is
  the judge's expected completion (target↔claim threading).
- **Levels 7/9 input filters** reject the attack before the model runs (the target
  reports `blocked=True`, model not called), exactly as upstream.
- **Levels 8/9/10 preprocessing** (`<`/`>` escaping; per-character backslash
  escaping; emoji-only filtering) is applied before the template, verbatim.
- **Levels 0-5 over-length cutoff.** Upstream's base `evaluate`
  (evaluator.py:74-86) rejects an input over 2000 tokens without calling the model
  (`blocked`); the levels that override `evaluate` (6-10) have no such cutoff. The
  target preserves this exactly — the check applies only to levels 0-5.
- **Model errors are caught per attempt.** Upstream wraps each attempt in
  `try/except` and returns a failed Response rather than aborting (evaluator.py:114-126);
  the target does the same (records `error=True`, the attempt fails) so a transient
  provider error does not stop the whole task.

## superred adaptation / deviations

- **litellm replaces the completer.** Upstream drives models through
  `completers.py` (pre-1.0 OpenAI SDK: `Completion`/`ChatCompletion`, plus a FlanT5
  HF space). The target sends the level prompt as a single user message via
  litellm — the modern equivalent of upstream's single-prompt completion. The
  MongoDB game DB and the Gradio UI are not ported.
- **Token count via litellm.** Upstream counts tokens with tiktoken; the target
  uses `litellm.token_counter` (the model's own tokenizer), falling back to a
  whitespace word count when unavailable (e.g. an unknown/stub model). Used for the
  levels-0-5 cutoff above and the informational `competition_score` sub-score.
- **Competition-score multiplier.** Upstream applies a 2× score multiplier for
  chat models (`completers.py`); the port's `compute_score` uses multiplier 1.0.
  This affects only the informational `competition_score` sub-score — the primary
  metric is binary attack success — so the model-specific multiplier is not threaded.
- **Optimizer surface.** `user_input` is the single Controllable (the attacker's
  turn), scoped to the untrusted-input boundary; passthrough runs no attack.

## Version scope

Ports **HackAPrompt v1** (the EMNLP-2023 11-level benchmark), the canonical cited
artifact — not the separate HackAPrompt 2.0 (2025) live competition.
