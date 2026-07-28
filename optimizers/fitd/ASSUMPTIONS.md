# FITD Optimizer Assumptions

## Original Attack Shape

FITD is a multi-turn attack. It starts with a benign query, then asks the model
to revise earlier answers toward gradually more harmful bridge prompts. The
implementation keeps that shape instead of turning FITD into a single prompt.

The default values follow the official code where they map to SuperRed:

- `level=10`
- `max_attempts=5`
- `control_history=False`
- `max_history_length=22`
- `max_soften_retries=10`
- `max_slippery_retries=2`

The official refusal phrases, sensitive-word replacements, prompt-generation
format, polish prompt, realign prompt, intermediate-prompt prompt, similarity
judge, harmfulness judge, and `safe`/`unsafe` evaluator prompt are preserved.

The initial soften step follows the official `change_prompt` shape: it retries
helper-model refusals and runs the extra sentence-structure alignment pass after
each successful rewrite. The target-side "try the softened prompt and see if it
refuses" part is mapped onto SuperRed runs rather than performed inside the
optimizer, because targets are only callable through framework events.

Prompt-ladder generation follows the official transformation format and retries
invalid/refusal outputs. Parsing into the official `Step 1..N` format is the sole
authority for a usable ladder: a candidate is accepted whenever it parses, even
if its first line opens with a refusal-shaped preamble (e.g. "I understand.
**Step 1**: ..."). The start-anchored refusal regex is consulted only to
*classify* a candidate that already failed to parse, never to discard one that
parsed. This removes a class of false refusals where a compliant ladder was
thrown away because its preamble matched the refusal regex.

If generation never returns the official `Step N` format, the assistant raises
`PromptTransformationRefused` (a `ValueError` subclass carrying per-cause counts
of refusals / parse failures / empty responses) instead of silently inventing a
non-official ladder. Every underlying attacker-LLM call has already succeeded at
that point (a transient API error or budget exhaustion propagates out of the
loop as itself), so this signals the attacker model being unable to operate on
this goal, not an infrastructure fault. The optimizer therefore catches it in
`_handle_run_start` and ends the attempt with no injection, so the task is
recorded as a completed attacker failure rather than escaping as
`stop_reason="error"` (which would discard the whole task from the measurement).
Genuine infrastructure faults are left untouched: they escape and the controller
records them as `stop_reason="error"` (or `"budget_exhausted"` for budget).

Note: the same start-anchored refusal regex is still used to detect *victim*
refusals during a run (to drive realign/backtracking). That path is unchanged;
the parse-first rule above applies only to the attacker's own ladder generation.

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

The official script also has a `num_limit` query counter because it directly
owns target and helper-model calls. SuperRed already has framework-level LLM
budgeting, and this optimizer's loops are bounded by `level`, `max_attempts`,
`max_soften_retries`, and `max_slippery_retries`. To avoid a second, inaccurate
shadow budget, this optimizer does not expose a separate `max_queries` knob.

`BudgetExhaustedError` from the attacker LLM is never swallowed. It propagates
out of `on_event` (including during prompt preparation) so the controller
records `stop_reason="budget_exhausted"`; the optimizer does not convert a spent
budget into a `done` response, which would make a budget failure look like a
clean completion.

## Refusal Backtracking

The paper and official code remove a refused query from target history before
continuing. SuperRed targets do not expose an in-place history pop primitive.

To avoid continuing from poisoned history, this optimizer stops the current run
when it sees a refusal, then replays the accepted prefix in the next run before
sending the recovery prompt. This matches the paper's backtracking intent as
closely as SuperRed allows.

`ChatbotTarget` no longer forces deterministic decoding. Replay therefore
cannot assume the target will produce byte-identical answers. During replay, the
optimizer rebuilds its state from the fresh target responses when those
responses are visible on the trajectory or PostCall event. If responses are not
visible, it falls back to the cached accepted answers so the open-loop path can
still make progress.

For SlipperySlopeParaphrase, the optimizer keeps the refused level pending after
a successful intermediate prompt. If that intermediate prompt is also refused,
the optimizer generates one more bridge prompt, matching the official two-try
slippery-slope loop. It uses the official similarity and harmfulness judges to
decide whether the intermediate prompt/answer should replace the optimizer's
previous prompt/answer, then retries the original next level. SuperRed currently
cannot rewrite the target's already-recorded assistant message, so this
replacement affects optimizer state and future prompts, not the target's private
conversation history.

Slippery-slope recovery is a *between-levels* mechanism: it bridges toward the
next ladder level. It is therefore only applicable while a next level exists.
The post-sequence `final_align` pass runs after the ladder is exhausted
(`_level_index == len(prompt_sequence)`), where there is no next level to slip
toward. A refused `final_align` ends the attempt, exactly as an accepted one
does; it does not open a recovery that would index one past the end of the
sequence. (Handling it like a mid-ladder refusal was a bounds bug that crashed
the run and discarded the task.)

Recovery metadata is carried on each active turn: the recovery kind, retry
number, and whether a successful bridge should retry the original next level.
That makes replay safer under nondeterministic targets because the optimizer can
rebuild the same recovery state from fresh replay responses instead of relying
on scattered global flags.

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
