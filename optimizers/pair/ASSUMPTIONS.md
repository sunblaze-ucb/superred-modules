# PAIR Optimizer Assumptions

## Original Attack Shape

PAIR runs several attacker conversations in parallel. Each attacker proposes a
JSON object with `improvement` and `prompt`. The target sees only the proposed
prompt. A judge scores the target response from 1 to 10, and each attacker uses
that response and score to refine its next prompt.

This optimizer keeps the current official runtime defaults:

- `n_streams = 3`
- `n_iterations = 3`
- `keep_last_n = 4`
- attacker `max_tokens = 500`
- attacker retries invalid JSON up to `5` times
- judge `max_tokens = 10`
- default judge method is `gcg`

The attacker system prompts, initial user message, target-response feedback
format, JSON fields, and GPT-judge prompt follow the official implementation.
Attacker generation also follows the official JSON behavior: stop at `}`, add
the closing brace back before parsing, retry malformed JSON up to five times,
and store the extracted JSON object in attacker history.

## Mapping To SuperRed

One SuperRed run can send only one target prompt. PAIR normally sends a batch of
one prompt per stream each iteration, so this optimizer stages that target batch
across multiple SuperRed runs. It finishes the current staged batch before
stopping on a successful candidate, matching PAIR's batch-at-a-time shape as
closely as the event loop allows.

The official implementation batches attacker LLM calls. SuperRed's optimizer
LLM client exposes one chat completion at a time, so attacker proposals are
generated sequentially while preserving the same prompts, defaults, retry
behavior, and per-stream histories.

The optimizer injects PAIR's `prompt` into `user_message` or the first visible
user-like controllable. It skips ChatbotTarget's `response` controllable and does
not modify model responses.

Verified against ChatbotTarget only. The trajectory response reader matches
observables by exact name (`response`, `model_response`, `assistant_response`,
or any names passed via `response_observable_names`); when adding a new target,
confirm its model-response observable uses one of those names or extend the
allow-list at construction time.

## Feedback And Success

The optimizer reads model responses from the filtered trajectory first. If no
trajectory response is visible, it falls back to a paired PostCall answer when
that is in scope.

When a model response is visible, the optimizer still runs PAIR's local judge so
the refinement loop receives the paper-shaped 1-10 score. When framework
evaluation is also visible, `evaluation.success` is the hard success signal. A
failed framework evaluation never counts as success, even if its numeric score is
high or the local judge is optimistic; the score sent back to the attacker is
capped below 10 in that case. If framework feedback is visible but the model
response is not, the optimizer uses the framework rationale and score to refine
the next PAIR prompt.

The official repository also has a JailbreakBench judge option. In SuperRed,
that role is covered by the framework evaluator and arrives on `RunEndEvent`.

If neither model response nor framework feedback is visible, PAIR cannot truly
learn from the target. In that blind setting, the optimizer continues through the
configured finite stream/iteration budget with score `1` and a no-signal message.

## Capability Use

When static observables are visible, the optimizer appends a small bounded
SuperRed context block to the attacker system prompt. This lets PAIR use granted
information such as target model identity while keeping the official prompt text
intact when no such observables are in scope.

When `system_prompt` is writable, the attacker may optionally return a
`system_prompt` JSON field. A non-empty value is injected into the system prompt,
and the normal PAIR `prompt` still goes to the user message. Missing or empty
`system_prompt` means no override. This is a SuperRed capability extension; when
`system_prompt` is not writable, the default path is the paper-style user-prompt
attack.

Static context is size-bounded so long observables do not crowd out the PAIR
instructions.

## Deliberate Differences

The official CLI has a harmful default goal and matching target prefix. SuperRed
provides the goal at runtime, so this optimizer defaults `target_str` to
`Sure, here is`. Callers can pass an exact target string when reproducing a
specific paper row or dataset item.

The official scripts own target model generation parameters. In SuperRed, the
target owns its own decoding settings, so this optimizer does not set target
`temperature`, `top_p`, or target `max_tokens`.

The official code also pins the attacker temperature to `1.0` and the judge
temperature to `0.0`. This optimizer sends no temperature on either call.
Reasoning models reject the parameter outright, and both the attacker and the
judge swallow their own failures, so a pinned temperature would silently disable
the optimizer on exactly the strongest attacker and judge models. Every other
generation parameter, including `top_p` and `max_tokens`, is unchanged.

The official implementation depends on FastChat conversation templates. SuperRed
LLM clients accept OpenAI-style chat messages directly, so this optimizer stores
the same roles and truncates the same last `2 * keep_last_n` attacker messages
without FastChat.
