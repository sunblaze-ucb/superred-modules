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
- If SuperRed grants writable `system_prompt`, the optimizer automatically uses it as a stronger framework capability: the synthesized jailbreak prompt goes into `system_prompt`, and the task goal goes into `user_message`.
- This keeps the user-only setting paper-faithful while making broader threat models meaningfully broader.
- Responses are read from the trajectory first. If no trajectory response is visible, the optimizer falls back to a matching PostCall answer.
- If `RunEndEvent.evaluation` is visible, `evaluation.success` is the authoritative success signal.
- A failed framework evaluation never counts as success, even if its numeric score is high.

## Intentional SuperRed Extensions

- In-scope static observables, such as target model identity or readable system prompt, are passed to the mutation model as bounded context.
- That context is only added when the framework exposes it. If the threat model does not expose static observables, mutator prompts stay identical to the official prompts.
- Static context is capped by `static_context_max_chars` so large observables do not crowd out the official mutator instruction.
- Writable system-prompt use is an automatic SuperRed extension. Disable it with `use_system_prompt_when_available=False` when you want the exact user-channel attack even in a broader scope.

## Known Difference

- The official implementation uses a finetuned RoBERTa predictor for response-only scoring. This module prefers SuperRed task feedback when available. When only a response is visible and no framework evaluation is present, it uses a lightweight refusal-string fallback rather than bundling a heavyweight model dependency.
