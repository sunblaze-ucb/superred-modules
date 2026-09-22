# Assumptions and deviations — mtjb_coa

Ported from MT-JailBench `engine/attacks/coa` at commit `cb8184e0`. Deviations
from upstream, and why each is necessary or safe.

## Flow control: no partial rewind

Upstream's `CoAFlowController` emits `CONTINUE`, `RETRY` (re-do the current turn
against the same partial conversation), and `JUMP_TO n` (rewind the target
conversation to an earlier turn). superred exposes only a full between-run reset
(`Target.reset_ephemeral_state`), never a partial rewind, so `RETRY` and
`JUMP_TO n>1` are not realizable.

- `RETRY` (regenerate the current round) is realized as **pre-send
  refinement**: instead of re-querying the target with a refined prompt, the
  next round's prompt is refined before it is sent, using the previous round's
  answer. This is exactly upstream's `_update_attack` step, moved earlier in the
  loop — the technique's optimization is preserved; only when it runs changes.
- Turn advance is sequential (`CONTINUE`). The three-way similarity DP that
  chooses between advancing, going back, and regenerating is not reproduced,
  because two of its three branches require a rewind. The binary-judge success
  check (`[[1]]` → stop) is preserved exactly.

## Mirror-target requery disabled

Upstream's evaluator computes `base_response_sem` by re-querying the target
model with the current prompt and *no* conversation history (a second target
client it constructs directly). A superred optimizer cannot query the target;
it only injects at controllables and reads the answer that comes back. The
requery is therefore disabled. `base_response_sem` / `posted_response_sem`
(which fed only the DP that is already gone) are not computed.

## Similarity: LLM-approximation only

Upstream can score similarity either with a separately deployed SimCSE endpoint
(`requests.post`) or with an LLM approximation. This port uses the
**LLM-approximation path only**, routed through `self.llm`, so no external
endpoint and no `sentence-transformers`/`requests` are needed. This matches the
recommended offline configuration. Similarity feeds only chain ranking at setup
and the pre-send refinement acceptance threshold.

## Upstream `{round}` bug fixed

Upstream's `_update_attack` formats the prompt-update user message with
`round=round`, passing Python's builtin `round` function (rendering
`"<built-in function round>"` into the prompt) instead of the current turn
number. This port passes the turn number, matching the clear intent and the
sibling system-prompt template that uses `round=turn`.

## One conversation, one pinned surface

CoA is a single multi-turn conversation. The optimizer pins the first eligible
free-text controllable it injects into and rejects other surfaces for the rest
of the conversation, so later rounds continue the same thread. Unlike
`actor_attack`, no LLM surface classifier is run; the name/value-type backstop
(never inject the reserved `system_prompt`; require a free-text value type) is
used directly.

## Chain generation cost

Upstream generates `n_init_chains` (default 3) candidate chains and ranks them
by semantic range, computing similarity for every round prompt. With the
LLM-approximation scorer this is several `self.llm` calls at setup. It is
faithful to upstream and bounded by `n_init_chains` and `max_turns`; a genuine
budget exhaustion propagates as `BudgetExhaustedError` (a spent run is never
reported as a quietly finished attack).

## Success criterion

Success is the upstream self-evaluation: the binary CoA judge returns `[[1]]`.
The independent 3-judge evaluation lives in the separate `mt_jailbench` security
claim, matching how superred separates attacker self-scoring from claim
scoring.
