# Assumptions and deviations — mtjb_mix

Ported from MT-JailBench `engine/attacks/mix` (and the shared `engine/`) at
commit `cb8184e0`. Mix is the heaviest port and carries the most deviations.

## Drives the vendored engine (not a native reimplementation)

CoA and X-Teaming are reimplemented natively in their packages. Mix instead
**runs upstream's own generator / outcome-evaluator / flow-controller verbatim**
for all five families, because reimplementing every family would duplicate the
whole ecosystem. The `engine/` subtree is vendored byte-for-byte and imported
lazily; the reconstructed engine loop lives in `_engine_bridge.py`.

## Model access shim

The vendored engine's `client.unified_llm_client` is replaced by a stdlib shim
(`_client_shim.py`) registered in `sys.modules` under that exact name. Its
`Conversation` is a faithful stdlib re-implementation; its `UnifiedLLMClient` is
a facade that ignores the configured model id and routes every call to superred's
constrained `self.llm`. No model SDK is imported.

## Sync engine driven from async superred

The vendored components are synchronous and expect to call the model directly.
Each engine step runs in a worker thread (`asyncio.to_thread`); the facade
bridges its synchronous calls back to the running event loop via
`asyncio.run_coroutine_threadsafe`. The LLMClient + loop are bound per task with
a `ContextVar` that `to_thread` copies into the worker thread, so concurrent Mix
optimizers do not interfere. `temperature` is never forwarded.

## textgrad / tiktoken gating

The vendored generator/updater import `textgrad` (via `xteaming.updater_utils`)
and `tiktoken` at load. So the Mix engine only runs with the `refine` extra
installed. Without it the optimizer imports fine and degrades cleanly: the engine
fails to start and the optimizer ends the run without sending. This path is
env-gated in the tests.

## No partial rewind

superred exposes only a full between-run reset. The reconstructed loop realizes
the engine's actions as follows:

- `CONTINUE` — commit the in-effect attempt, advance the turn, generate the next
  prompt (faithful).
- `RETRY` — upstream rewinds the conversation and re-sends a refined prompt at
  the same depth. Here the refined prompt is sent as the **next** conversation
  turn (no rewind); because the current turn then has more than one attempt, the
  generator's `refine_prompt` fires exactly as upstream intends. Every sent
  prompt is a real, permanent turn.
- `JUMP_TO` (including strategy restart via `JUMP_TO 1`) — a rewind that cannot
  be realized, so the attack ends. `max_restarts` therefore has no rewind effect.

The rest of the per-attempt bookkeeping (epoch/turn advance, `attempt_in_effect`,
`_conv_after_response`, `full_history`) mirrors `jailbreak_engine` faithfully.

## Mirror-target requery

Upstream's CoA (`base_response_sem`) and FITD (`update_fitd` refusal re-test)
sub-modes construct a second target client and re-query the victim. superred
optimizers cannot query the target; the facade's `UnifiedLLMClient` routes those
mirror calls to `self.llm` (the attacker model) instead. This is a poor proxy for
the victim and is a known fidelity caveat for the CoA/FITD roles specifically.

## Upstream simplifications carried forward

MT-JailBench's `plan_generation.py` already simplifies the composed families
(1 actor, 1 CoA chain, 1 X-Teaming strategy, and no target refusal re-check at
FITD plan time). Those simplifications are inherited unchanged (the code is
vendored verbatim).

## Bounds

`max_total_turns` caps conversation turns (runs) per task and is also passed as
the engine's `max_epochs`; `max_turns` is the planned turn count handed to the
generator. Together they bound the loop so a refine cycle cannot run unbounded.

## Testability

The generation/refine/judge path cannot be exercised offline (textgrad is absent
from the shared venv and no installs are permitted). The offline tests cover the
event-loop adapter with the engine bridge stubbed, role validation, import
without heavy deps, and vendor byte-identity; the real-engine test is env-gated
(`importorskip("textgrad")`) and skips in that environment.
