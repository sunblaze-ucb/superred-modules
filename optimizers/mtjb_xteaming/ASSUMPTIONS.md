# Assumptions and deviations — mtjb_xteaming

Ported from MT-JailBench `engine/attacks/xteaming` at commit `cb8184e0`.

## Flow control: no partial rewind

Upstream's `XTeamingFlowController` emits `CONTINUE`, `RETRY` (re-do the current
turn against the same partial conversation with a TextGrad-refined prompt), and
`JUMP_TO 1` (restart to rotate strategy when `use_multiple_strategies`). superred
exposes only a full between-run reset, not a partial rewind.

- `CONTINUE` advances the strategy plan (the next turn's prompt) — realized
  directly.
- `RETRY` is realized as **pre-send refinement**: the just-sent prompt is
  refined and sent as the next conversation turn, rather than re-sending at a
  rewound depth. The flow decision (turn-1 continues; the last plan step refines
  up to `max_refines_per_turn`; a turn that beats the historical best continues,
  else refines) is preserved. Because the conversation cannot rewind, every
  sent prompt — refined or planned — is a real, permanent turn, and the
  conversation history shown to the attacker reflects that.
- Strategy rotation via `JUMP_TO 1` (`use_multiple_strategies`) is not
  reproduced; a single strategy is used per task (matching MT-JailBench's own
  "1 strategy" simplification in the Mix composite).

## TextGrad refinement is optional and pre-send

Upstream refines with TextGrad unconditionally. Here the refine step:

- is imported lazily and only when `enable_textgrad_refine` is set and
  `textgrad` is importable; otherwise a would-be refine advances the plan
  (so the optimizer runs with no heavy dependency).
- runs in a worker thread (`asyncio.to_thread`) because TextGrad is synchronous;
  its backward engine bridges each model call back to the running event loop via
  `asyncio.run_coroutine_threadsafe`, so every call still goes through the
  constrained `self.llm`.
- injects the already-observed target response through a fixed-response engine
  (no target re-query), exactly as upstream does.

The TextGrad loss template and scoring policy are loaded verbatim from the
vendored `xteaming_prompt_generator.py` / `updater_utils.py`. The TextGrad role
descriptions are neutral labels authored here (graph metadata, not payload).

## Judge degrades instead of raising

Upstream's evaluator raises on an unparseable or out-of-range (not 1–5) judge
score. As an attacker self-evaluation this port retries `max_judge_retries`
times and then degrades to the lowest score (1), rather than crashing the run —
mirroring how `actor_attack` treats a failed rating as the refusal score. (The
strict, abstain-on-failure judge lives in the `mt_jailbench` claim.)

## Strategy selection

Upstream requires exactly 10 strategies and uses the first. This port keeps the
"exactly 10" requirement (retrying generation up to 5 times) and selects the
first strategy that carries the fields the turn prompts need; if none is valid
after 5 attempts the attack ends without sending (nothing was set up).

## Truncation fallback

Conversation-history responses are truncated with `tiktoken` when available
(upstream `truncate_response`); when it is absent a char-based bound is used so
history stays bounded offline (upstream returns the untruncated text on error).

## One conversation, one pinned surface

As with `mtjb_coa`, the optimizer pins the first eligible free-text controllable
and rejects other surfaces for the rest of the conversation; the reserved
`system_prompt` is never injected. No LLM surface classifier is run.

## Global turn cap

`max_total_turns` bounds the number of conversation turns (runs) per task so a
refine loop cannot run unbounded; the strategy's plan length drives the turn
structure within that cap.
