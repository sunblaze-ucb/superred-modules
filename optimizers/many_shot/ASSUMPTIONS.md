# ManyShot Optimizer Assumptions

## Original Attack Shape

The optimizer follows the Many-Shot Jailbreaking idea: build one long prompt
with many faux user/assistant examples, then append the real objective as the
final user turn.

Bundled examples come from `data/examples.json`. The optimizer preserves the
basic prompt shape: preamble, repeated `User:` / `Assistant:` examples, then
the target objective. If the dataset includes extra metadata fields such as
`category`, the prompt uses only `user` and `assistant`.

Examples are used in their source order by default. The default is one run with
`example_counts=[100]`.

The paper says the attack strings were randomized before formatting. This
module defaults to deterministic first-N example selection for reproducibility.
To run closer to the paper's ordering description, set
`shuffle_examples=True`; use `random_seed` when you need the shuffle to be
repeatable.

## Intentional Differences

The original work studies attack success as the number of examples increases.
This module can map that idea onto SuperRed staged runs if configured with a
count schedule:

1. Each run uses one configured example count.
2. Later runs can use larger example counts.
3. A useful sweep schedule is `[32, 64, 128, 256, 400, 512]`.

This staged schedule is a framework feature, not the default. For a single
paper-style 256-shot attempt, configure `example_counts=[256]`.

## Extra Example Generation

If `generate_extra=True` and the requested count is larger than the bundled
dataset, the optimizer asks the controller-provided `self.llm` to generate
more faux dialogues.

This path depends on the generation model. A safety-trained model may refuse,
sanitize, or produce weak examples. For stronger reproduction of the original
many-shot setup, prefer a large curated example set or a generation model that
is suitable for red-team data generation.

Extra generation is off by default. This keeps the default path limited to the
provided dataset rather than generating more examples inside the attack.

## SuperRed Feedback

ManyShot can use two feedback paths:

1. Filtered trajectory response observables: when the response is visible on
   the trajectory, ManyShot reads that response first and scores it with its
   own evaluator.
2. `ControllablePostCallEvent`: if no response observable is visible, ManyShot
   falls back to the post-call answer when that event is in scope.
3. `RunEndEvent.evaluation`: when the task evaluator provides framework
   feedback, ManyShot uses `evaluation.success` and `primary_score` to decide
   whether to stop.

The framework evaluation path is important for scopes where post-call response
events are not visible to the optimizer.

## Controllable Mapping

For single-channel targets, ManyShot injects the long prompt into the first
visible user-like controllable and injects only once per run.

For chatbot-style split channels, ManyShot does not inject into
`system_prompt`. It waits for `user_message` and injects the long prompt there.
This avoids turning the many-shot user prompt into a system prompt.

## Scoring

The local evaluator is a pragmatic SuperRed scoring helper, not an exact paper
metric. It asks the controller-provided LLM for a JSON score from 0.0 to 1.0.
When `RunEndEvent.evaluation` is available, that framework feedback can also
mark the attempt as successful.
