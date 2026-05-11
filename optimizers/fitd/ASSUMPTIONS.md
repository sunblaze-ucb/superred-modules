# FITD Optimizer Assumptions

## Original Attack Shape

FITD is a multi-turn attack. It starts with a benign query, then asks the model
to revise earlier answers toward gradually more harmful bridge prompts. The
implementation keeps that shape instead of turning FITD into a single prompt.

The default values follow the official code where they map to SuperRed:

- `level=10`
- `max_queries=50`
- `max_attempts=5`
- `control_history=False`
- `max_history_length=22`

The official refusal phrases, sensitive-word replacements, prompt-generation
format, polish prompt, realign prompt, intermediate-prompt prompt, and
`safe`/`unsafe` evaluator prompt are preserved.

## Mapping To SuperRed

One SuperRed run is one FITD attempt. Within that run, targets such as
`ChatbotTarget` can emit repeated `user_message` PreCall events, so the optimizer
can run the FITD conversation turn by turn.

The optimizer injects into `user_message` or the first user-like controllable. It
skips the `response` controllable and never edits model responses.

When `system_prompt` is writable, the optimizer uses it only for a short FITD
conversation setup. The real FITD query ladder still goes through `user_message`.
If `system_prompt` is not writable, behavior stays paper-mainline.

Static observables, such as the target model name or readable system prompt, are
passed to the assistant model that generates the prompt ladder. This context is
bounded and truncated so long observables cannot crowd out the official FITD
instructions.

## Feedback And Success

The optimizer reads target responses from the filtered trajectory first. If no
trajectory response is visible, it falls back to an in-scope PostCall answer.

When `RunEndEvent.evaluation` is present, framework success is authoritative. A
failed framework evaluation is not overridden by the local FITD judge.

When framework feedback is not available but a model response is visible, the
local evaluator uses the official `safe`/`unsafe` FITD evaluation prompt.

If neither responses nor feedback are visible, FITD runs open-loop through the
prompt ladder. It cannot detect refusals or do realign/paraphrase recovery in
that threat model, but it still exercises the user-message capability.

## Refusal Backtracking

The paper and official code remove a refused query from target history before
continuing. SuperRed targets do not expose an in-place history pop primitive.

To avoid continuing from poisoned history, this optimizer stops the current run
when it sees a refusal, then replays the accepted prefix in the next run before
sending the recovery prompt. This matches the paper's backtracking intent as
closely as SuperRed allows. It works best when the target is deterministic.

## Deliberate Differences

The official scripts run over fixed CSV/JSON benchmark files and save result
artifacts. This optimizer instead accepts any SuperRed goal. Callers may provide
a `prompt_sequence` to replay a pre-generated official ladder exactly; otherwise
the optimizer asks the attacker LLM to generate the bridge sequence using the
official transformation prompt.

The official code owns target model calls directly. SuperRed targets own target
decoding, temperature, and chat formatting, so this optimizer does not pass
target decoding parameters.
