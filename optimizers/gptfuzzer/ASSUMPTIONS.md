# GPTFuzzer Assumptions

This module follows the GPTFuzzer paper and official implementation as closely as SuperRed's event model allows.

## What Matches The Official Work

- The bundled seed templates come from the official `GPTFuzzer.csv` file.
- The target prompt is built the same way: replace `[INSERT PROMPT HERE]` in a jailbreak template with the task goal.
- The five mutation operators are the official ones: crossover, expand, generate similar, rephrase, and shorten.
- Mutator prompts use the official wording when no extra SuperRed context is available.
- The default mutator policy is the official random-single policy with concatenation enabled.
- The default selector is the official MCTS-Explore selector, using the same reward and exploration formulas.
- Default budgets follow the official runner: `max_query=1000`, `max_jailbreak=1`, and `energy=1`.

## How This Maps To SuperRed

- One SuperRed run evaluates one synthesized GPTFuzzer prompt against the target.
- With the default `energy=1`, this is the same one-mutation, one-query loop used by the official runner.
- If `energy > 1`, extra mutations are queued across later SuperRed runs because a SuperRed target run has one active user query.
- Under a user-only threat model, the optimizer injects the full synthesized GPTFuzzer prompt into `user_message`, matching the official attack shape.
- If SuperRed grants writable `system_prompt`, the optimizer automatically uses it as a stronger framework capability: the template prefix before `[INSERT PROMPT HERE]` goes into `system_prompt`, and the task goal plus the template suffix goes into `user_message`.
- This keeps the user-only setting paper-faithful while making broader threat models meaningfully broader.
- Responses are read from the trajectory first. If no trajectory response is visible, the optimizer falls back to a matching PostCall answer.
- If `RunEndEvent.evaluation` is visible, `evaluation.success` is the authoritative success signal.
- A failed framework evaluation never counts as success, even if its numeric score is high.
- If no framework evaluation is visible but a model response is visible, the optimizer uses the official GPTFuzzer RoBERTa classifier, `hubert233/GPTFuzz`, to score that response.

## Intentional SuperRed Extensions

- In-scope static observables, such as target model identity or readable system prompt, are passed to the mutation model as bounded context.
- That context is only added when the framework exposes it. If the threat model does not expose static observables, mutator prompts stay identical to the official prompts.
- Static context is capped by `static_context_max_chars` so large observables do not crowd out the official mutator instruction.
- Writable system-prompt use is an automatic SuperRed extension. It splits the selected template at the first `[INSERT PROMPT HERE]` placeholder instead of duplicating the synthesized goal across channels. Disable it with `use_system_prompt_when_available=False` when you want the exact user-channel attack even in a broader scope.

## Runtime Predictor Notes

- The default response predictor is `hubert233/GPTFuzz`, matching the official implementation.
- The model is loaded lazily on the first response-visible run, so unit tests do not download model weights.
- If the RoBERTa dependencies or weights are unavailable, the optimizer can fall back to a lightweight refusal-string predictor. Set `allow_predictor_fallback=False` to require the official classifier and fail fast instead.
- Loading the scorer fails with no single exception type: an unreachable Hub raises `OSError`, a bad repo id raises `ValueError`, a config the installed `transformers` rejects raises a plain `Exception`, a missing dependency raises `ImportError`, a device mismatch raises `RuntimeError`. `RoBERTaPredictor` normalizes all of them to `PredictorUnavailableError` (a `RuntimeError` subclass, so the previous contract still holds) rather than asking callers to enumerate the set.
- The fallback therefore triggers on any primary-predictor failure, not on a chosen exception type. The optimizer scores responses from `on_event`, where an escaping exception is not a scoring failure but a data-loss event: the framework rejects the event, the controller sets `stop_reason="error"`, and the task's measurement is discarded even though the target already answered. Degrading the scorer is strictly better than that.
- The switch is latched for the predictor's lifetime, which the framework scopes to one task. That keeps every response in a task scored by the same classifier (mixing two scorers would corrupt the seed-selection signal) and avoids retrying a multi-second model load on every run. A later task builds a fresh predictor and retries the real model, so a transient outage costs one task's scoring fidelity, not the sweep's.
- A permanently unusable scorer is an operator error, not a runtime condition, and the per-task fallback cannot tell it apart from a transient one. `RoBERTaPredictor.load()` exists to surface it once at startup; the degradation warning names the cause, and `FallbackPredictor.fallback_active` / `.fallback_reason` make it auditable after the fact.
