# TAP Optimizer Assumptions

## Original Attack Shape

This module follows Tree of Attacks with Pruning (TAP):

1. Start from one root attack stream.
2. Ask an attacker LLM to propose jailbreak prompts.
3. Prune prompts that are off topic.
4. Send every surviving prompt to the real target.
5. Judge each prompt and target response together on a 1-10 scale.
6. Keep the best branches and repeat until a score of 10 or max depth.

The defaults stay close to the original implementation:

1. `branching_factor=1`
2. `root_nodes=1`
3. `tree_width=10`
4. `tree_depth=10`
5. `attacker_max_tokens=500`
6. `evaluator_max_tokens=10`
7. `max_attack_attempts=5`
8. `keep_last_n=3`

The attacker prompt, initial message, feedback format, on-topic judge, and
jailbreak judge are written to match the original TAP flow closely while still
living inside the SuperRed optimizer interface.

## SuperRed Mapping

The original TAP code can send a batch of candidate prompts directly to a target
model. SuperRed targets are event-driven instead: the optimizer gets a
`ControllablePreCallEvent`, injects one user query, then receives any visible
model response on the trajectory and any task feedback on `RunEndEvent`.

Because of that, this module maps TAP's target batch onto a queue:

1. At a tree depth, TAP generates and on-topic-prunes candidates.
2. Each surviving candidate is tested in its own SuperRed run.
3. After all candidates from that depth have been tested, TAP judge-prunes the
   tree and moves to the next depth.

This is intentional. It lets every surviving candidate hit the real SuperRed
target without turning a chatbot target into one long multi-turn conversation
where later candidates see earlier candidate responses.

## Feedback Sources

TAP handles the four common SuperRed adversarial settings like this:

1. User query only, no model response, no task feedback: inject candidates, but
   score them as 1 because no target evidence is visible.
2. User query plus task feedback, no model response: use `RunEndEvent.evaluation`
   to stop on success and to rank candidates when possible.
3. User query plus model response, no task feedback: read the response from the
   filtered trajectory and score the prompt/response pair with TAP's judge.
4. User query plus task feedback and model response: use framework success as a
   hard success signal, and use the trajectory response for TAP judge scoring
   when success is not already true.

`ControllablePostCallEvent` is not used for scoring. It always returns
`ControllableNoInjection`. This keeps response reading aligned with SuperRed's
scope model: model responses come from trajectory observables when they are in
scope.

## Prompt Placement

TAP is a user-prompt attack. For chatbot-style targets with separate
`system_prompt` and `user_message` controllables, this optimizer does not inject
into `system_prompt`. It waits for `user_message` and injects the TAP candidate
there.

## Target String

The original TAP prompts include a target opening string, such as "Sure, here is
...". SuperRed goals usually provide only a goal description, so this module
defaults `target_str` to `"Sure, here is"`.

For closer reproduction of a specific TAP experiment, configure `target_str` to
match that experiment's desired target opening.

## Intentional Differences

The main intentional difference is the queued target testing described above.
It is a framework adaptation, not a change to TAP's search idea.

The optimizer also uses the controller-provided LLM client for attacker and
judge calls. It does not directly call the SuperRed target model from inside the
optimizer; real target access happens through PreCall injection and trajectory
feedback.

The framework feedback fallback is also a SuperRed addition. It lets TAP make
progress in scopes where the target response is not visible but task feedback is
visible.
