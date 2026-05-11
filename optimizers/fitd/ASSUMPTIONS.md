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
format, polish prompt, realign prompt, intermediate-prompt prompt, similarity
judge, harmfulness judge, and `safe`/`unsafe` evaluator prompt are preserved.

Prompt-ladder generation follows the official transformation format and retries
invalid/refusal outputs. If generation never returns the official `Step N`
format, the optimizer raises instead of silently inventing a non-official ladder.
Callers can pass `prompt_sequence` to reproduce a stored ladder; matching the
official script, the final target prompt is appended to that sequence during
setup.

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

When framework feedback is not available but model responses are visible, the
local evaluator uses the official `safe`/`unsafe` FITD evaluation prompt. It
checks the final response first, then uses the official `find_most` judge prompt
to select and evaluate the most jailbreak-like prior output.

If neither responses nor feedback are visible, FITD runs open-loop through the
prompt ladder. It cannot detect refusals or do realign/paraphrase recovery in
that threat model, but it still exercises the user-message capability.

`max_queries` counts target turns plus helper-LLM calls made during recovery and
local evaluation. This is close to the official script's `total_query` counter,
while SuperRed's LLM budget remains the hard cost control. Prompt-ladder
generation happens before target interaction and is governed by the attacker LLM
budget rather than this target-loop query counter.

## Refusal Backtracking

The paper and official code remove a refused query from target history before
continuing. SuperRed targets do not expose an in-place history pop primitive.

To avoid continuing from poisoned history, this optimizer stops the current run
when it sees a refusal, then replays the accepted prefix in the next run before
sending the recovery prompt. This matches the paper's backtracking intent as
closely as SuperRed allows. It works best when the target is deterministic.

For SlipperySlopeParaphrase, the optimizer keeps the refused level pending after
a successful intermediate prompt. It uses the official similarity and
harmfulness judges to decide whether the intermediate prompt/answer should
replace the optimizer's previous prompt/answer, then retries the original next
level. SuperRed currently cannot rewrite the target's already-recorded assistant
message, so this replacement affects optimizer state and future prompts, not the
target's private conversation history.

## Deliberate Differences

The official scripts run over fixed CSV/JSON benchmark files and save result
artifacts. This optimizer instead accepts any SuperRed goal. Callers may provide
a `prompt_sequence` to replay a pre-generated official ladder exactly; otherwise
the optimizer asks the attacker LLM to generate the bridge sequence using the
official transformation prompt.

The official code owns target model calls directly. SuperRed targets own target
decoding, temperature, and chat formatting, so this optimizer does not pass
target decoding parameters.

The `control_history` option cannot physically trim ChatbotTarget's private
conversation state. It limits the accepted-prefix replay maintained by the
optimizer. Exact live-history truncation would require a target-side history
mutation primitive.
