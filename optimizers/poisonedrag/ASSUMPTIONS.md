# PoisonedRAG Optimizer Assumptions

## Paper-Faithful Defaults

- Default attack path is `LM_targeted`. Set `max_attempts=1` for the released code's single-shot poison batch (paper-parity ASR).
- Defaults match the released code where SuperRed can use them: `adv_per_query=5` and `top_k=5`.
- Poison docs use the official black-box shape: `question + "." + corpus`.
- The official multi-context RAG wrapper and JSON joint-generation prompt are preserved in `prompts.py`.
- When framework feedback is not visible, success falls back to the released check: `clean_str(incorrect_answer) in clean_str(response)`.
- `official_adv_results_dataset` loads bundled official `nq`, `hotpotqa`, or `msmarco` attack results; `official_adv_results_path` can load a custom file.
- Budget use: with no explicit `max_attempts` the optimizer does not self-cap — it keeps attempting until success or the attack is undeliverable, letting the controller's run budget bound the loop, so granted capability is not left unused. An explicit `max_attempts=N` is a hard cap (use `1` for paper-parity). Each delivered LLM-generated attempt regenerates a fresh poison batch (temperature 1.0, so independent samples); static bundled or user-provided poisons are reused unchanged since the paper has no evolution step (repeats only help against a stochastic target generation/retrieval).

## SuperRed Mapping

- SuperRed owns target execution, retrieval, scope filtering, trajectory, and task evaluation.
- Writable corpus/context surfaces are preferred, then writable `system_prompt`, then writable user prompt, then runtime context PostCall surfaces.
- Static surface matching is tried first; then the optimizer can ask its LLM to classify any remaining in-scope controllables as corpus/context/user-prompt surfaces. The corpus/context label only affects routing, not the on-wire format.
- A doc-carrying (corpus or context) surface is poisoned once per run whether it is exercised as a `PreCall` or a `PostCall`, since a controllable may use either event.
- The on-wire format follows the controllable's value type, not its corpus/context label, on both `PreCall` and `PostCall`. On `PreCall`, a JSON surface receives the merged JSON payload (preserving an existing list/dict shape read from `event.request`, falling back to the metadata wrapper only when the target gives no usable schema); any other surface receives plain poison-context text. On `PostCall`, `event.answer` is the genuine CURRENT read content, not a write template, so a JSON surface cannot be merged deterministically the same way; the value is instead built by the shared `surface_llm.fill_value` formatter, which reads the surface's description and embeds the poison documents verbatim into a schema-matching value. A free-text surface still gets plain poison-context text on either event, with no LLM call. If the JSON formatter cannot produce a value, the optimizer declines that delivery (leaving the once-per-run corpus gate open) rather than emit text into a structured surface.
- A DTAP `env_inject:<server>` environment-vector surface (writes attacker data into a live backend the agent reads back later) is recognized by a static `env_inject` name hint alongside the other corpus hints, since its description names an `inject_*` tool call, which the shared LLM surface-role classifier can read as "execution" rather than data and drop.
- If only `user_message` is writable, the optimizer uses the official RAG wrapper in the user prompt. This is a capability fallback, not true database poisoning.
- If only `system_prompt` is writable, the optimizer can place the official RAG wrapper and poison contexts there. This is also a SuperRed capability extension.
- Framework `RunEndEvent.evaluation` is authoritative when visible; otherwise the optimizer reads response observables from trajectory first, then scoped PostCall answers.
- If an advertised writable surface does not fire, the candidate is not scored and the optimizer can try again. If only speculative runtime context poisoning was possible and no context event appears, the optimizer gives up to avoid empty retries.

## Limits

- HotFlip is not implemented because SuperRed optimizers do not receive retriever weights, tokenizers, gradients, CUDA state, or BEIR scores.
- Exact retrieval precision/recall/F1 remains target- or benchmark-owned. The optimizer only tracks visible poison-doc hits through `last_retrieved_poison_count` and `best_retrieved_poison_count`.
