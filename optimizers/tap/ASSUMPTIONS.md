# TAP Optimizer Assumptions

## Paper-Aligned Defaults

This module follows Tree of Attacks with Pruning (TAP): generate candidate
jailbreak prompts, prune off-topic prompts, test every surviving candidate
against the real target, judge each prompt/response pair on a 1-10 scale, prune
to the best branches, and repeat until score 10 or max depth.

Defaults match the original implementation where they map cleanly to SuperRed:

1. `branching_factor=1`
2. `root_nodes=1`
3. `tree_width=10`
4. `tree_depth=10`
5. `attack_top_p=0.9`
6. `attacker_max_tokens=500`
7. `evaluator_temperature=0.0`
8. `evaluator_max_tokens=10`
9. `max_attack_attempts=5`
10. `keep_last_n=3`

The attacker prompt, seed message, feedback message, on-topic judge, jailbreak
judge, JSON retry behavior, history truncation, and pruning fallback are kept
close to the original TAP code.

## SuperRed Mapping

Original TAP sends a batch of candidate prompts directly to the target. SuperRed
targets are event-driven, so this module maps that batch to a queue:

1. Generate and on-topic-prune candidates for the current depth.
2. Inject each surviving candidate into `user_message` in its own target run.
3. After all candidates at that depth have run, judge-prune and move deeper.

This is the main intentional framework adaptation. It lets every surviving TAP
candidate hit the real target without making a chatbot target treat separate
candidate tests as one continuous conversation.

## Feedback Sources

TAP reads target responses from the filtered trajectory first. If no response is
visible, it uses `RunEndEvent.evaluation` when available. If neither response nor
feedback is visible, the candidate is scored as 1.

`ControllablePostCallEvent` is not used as a response source and always returns
`ControllableNoInjection`. This preserves SuperRed's scope model: model responses
are available only through in-scope trajectory observables.

Framework success feedback is treated as a hard success signal. When both a
trajectory response and framework evaluation are visible, TAP keeps the response
text and also uses the framework score/rationale as feedback. This is a SuperRed
extension: the original TAP paper did not have an external task evaluator.

If a response is visible but framework evaluation is not, TAP uses the TAP judge
on the prompt and response together.

## Prompt Placement

TAP is a user-prompt attack in the original setting. If SuperRed only exposes
`user_message`, this optimizer keeps that behavior and injects TAP candidates
there.

If SuperRed also exposes `system_prompt`, the optimizer treats that as a
stronger threat model and uses it automatically. The attacker may return both
`system_prompt` and `prompt`: the optimizer injects the first into
`system_prompt` and still injects the second into `user_message`.

## Static Target Context

The original TAP attacker does not receive target metadata such as model name or
available injection points. SuperRed can expose that information through the
threat model before a run starts.

When extra in-scope static context is available, TAP includes it in the attacker
prompt. This is a SuperRed extension. If only the normal `user_message`
controllable is available and no static observables are exposed, no static
context is added and the prompt path stays paper-faithful.

## Target String

Original TAP includes a desired target opening string. SuperRed goals usually
only provide a goal description, so this module defaults `target_str` to
`"Sure, here is"`. Configure `target_str` for closer reproduction of a specific
TAP experiment.
