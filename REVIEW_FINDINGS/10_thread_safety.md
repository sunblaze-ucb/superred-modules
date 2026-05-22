# Thread Safety and Sync-to-Async Bridge Review

Scope: `runtime_wrapper.py`, `pipeline_bridge.py`, and the cross-cutting concurrency story between `AgentDojoTarget`, the controller's event loop, the `AgentPipeline.query` worker thread spawned by `asyncio.to_thread`, and the optimizer task.

## Summary

The bridge is structurally correct under the happy path: a fresh per-task `Target` (concurrency=4/8 stress-tested in `test_concurrent_isolation.py`) constructs a fresh `EventChannel`, a fresh `WrappedFunctionsRuntime` and a fresh pair of hooks, all bound to `asyncio.get_running_loop()` at the time `run()` was entered. The only `_await_event` users (`WrappedFunctionsRuntime`, `_CatalogEditHook`) capture `loop` per-instance — there is no shared loop across Target instances. Module-level state (`ALL_FUNCTIONS`, `TOOL_REGISTRY`, `READ_CTRLS`, the security tag singletons, suite seed YAMLs) is built once at import and used read-only, and `_seed_overrides` is a per-instance dict.

The main risks live at the error edges and protocol corners: `future.result()` has no timeout and will block the worker thread forever if the loop is torn down while the worker is mid-call; the channel's `set_error` only rejects already-queued envelopes, not the synchronous `run_coroutine_threadsafe` future that is in-flight; and `_try_apply` swallows mutation errors silently (correctness, not deadlock). One HIGH-severity gap, three MEDIUM, plus a handful of LOW/INFO items below.

## Findings

### F1. [HIGH] `_await_event` can hang the worker thread if the loop dies mid-call

**Locations**:
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/runtime_wrapper.py:197-200`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/pipeline_bridge.py:96-98`

Both `_await_event` implementations call `asyncio.run_coroutine_threadsafe(coro, loop).result()` with no `timeout` argument. `concurrent.futures.Future.result()` without a timeout blocks the worker thread indefinitely. Three failure modes:

1. **Loop closed mid-flight**: if the controller crashes, the event loop is cancelled, or someone calls `loop.close()` while the worker has a pending future, the coroutine `send_event(event)` is silently cancelled and `future.set_result` / `future.set_exception` is never called from inside the (now-stopped) loop. `future.result()` blocks the worker forever. Because the worker is the body of `asyncio.to_thread(pipeline.query, ...)` in `target.run`, the outer `await asyncio.to_thread(...)` likewise never returns; the controller's `_run_task` `finally` (lines 643-662) is queued behind it, so `channel.close()`, `optimizer.teardown()`, and `target.cleanup()` are all blocked. The whole worktree run is wedged.

2. **Channel error-poisoned after a send_threadsafe is queued**: `EventChannel.set_error` (channel.py:186-208) rejects already-queued envelopes via `item.reject(error)`, but the wrapper's coroutine doesn't reach `channel.send` synchronously — it is scheduled via `run_coroutine_threadsafe` and may not have been awaited yet. If `set_error` runs first, the coroutine then calls `channel.send`, which checks `self._error` (channel.py:150-151) and raises. That propagates through the future and `future.result()` re-raises — safe. But if `set_error` runs **after** the coroutine has entered `channel.send` and queued an envelope, the queued envelope is rejected correctly and the sender's future is set to the exception. Also safe.

3. **Optimizer task crashed during dispatch**: the controller wraps `optimizer.run` in `_optimizer_with_error_propagation` (controller.py:539-544), which calls `channel.set_error(exc)` and re-raises. This poisons the channel, but only flips the next `channel.send` to raise; an envelope already received by the optimizer for which `on_event` raised would have been rejected by `_dispatch` (optimizer.py:191) — the worker's future gets the exception. Safe.

The bare-loop-closed case (mode 1) is the unmitigated one. Mitigations:
- Pass a timeout to `future.result(timeout=...)` and re-raise on `TimeoutError`. A long but finite timeout (60-120s of no activity is unrecoverable) avoids permanent wedge.
- Or have the controller actively cancel the worker via the pipeline. The `concurrent.futures.Future` returned by `run_coroutine_threadsafe` supports `.cancel()` from any thread; the controller could keep a reference and cancel on shutdown. Not trivial because there can be many in-flight wrapper calls per run.
- At minimum, document this assumption in `ASSUMPTIONS.md`: "the controller must not close the loop while a target is running."

### F2. [MEDIUM] `_try_apply` swallows mutation exceptions other than `ValueError`

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/pipeline_bridge.py:100-115`

The `try/except ValueError` covers the documented rejection path from `ToolCatalog.apply_*`, but any other exception (`KeyError`, `TypeError`, `RuntimeError`, etc.) thrown by `method(payload)` propagates out of `_try_apply`, which runs on the worker thread inside `pipeline.query`. The exception escapes the `for ctrl, apply_method in ops:` loop and out of `_CatalogEditHook.query`, eventually escaping `AgentPipeline.query`. Target.run's `try/except Exception` (target.py:309-313) catches it and breaks the retry loop — semantically reasonable but classifies a malformed-injection bug as a model error. This is a **correctness** issue, not deadlock; an attacker payload that triggers a non-ValueError inside catalog code silently aborts the run.

Stronger guard: `try ... except Exception` with a clear log message would convert this from a silent abort into a logged failure-to-apply. Even better: explicitly validate payload types in `apply_*` so only `ValueError` is ever raised; this is the doc-stated contract.

### F3. [MEDIUM] `_MessageStreamHook._next_idx` and `WrappedFunctionsRuntime._tool_response_counter` rely on AgentDojo's sequential pipeline contract

**Locations**:
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/pipeline_bridge.py:171, 183-191`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/runtime_wrapper.py:171, 332-333`

Both counters are plain `int` attributes mutated without a lock. Today this is safe because `AgentPipeline.query` is strictly sequential (verified upstream at `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/agent_pipeline/agent_pipeline.py:168-180` and `tool_execution.py:46-115` — both iterate elements/tool calls in plain `for` loops, no `asyncio.gather`/threads). So one worker thread per target instance owns both counters for the lifetime of `run()`, and the `read+write` is read-modify-write by a single thread. Each Target instance owns its own counter pair (constructor-local), so concurrent Target instances don't share.

The risk is forward-compatibility:
- If upstream ever adds parallel tool execution (a known opportunity in agent frameworks), `_tool_response_counter` would race.
- If we ever splice a hook that itself spawns workers (e.g. a streaming LLM that emits messages concurrently), `_next_idx` would race.
- Brief Section 2.g's `concurrency >= 1` contract is per-Target-instance, not per-internal-thread.

Fix is cheap: wrap the increment in a `threading.Lock` or use `itertools.count()`/`atomic-int`. The cost (one lock per emit) is negligible; the protection from a silent ordering bug is worth it. Or assert in the docstring that the counter assumes sequential dispatch.

### F4. [MEDIUM] `WrappedFunctionsRuntime._trace.append` is technically a CPython GIL accident

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/runtime_wrapper.py:170, 214-216, 174-182`

`list.append` is documented as atomic under CPython's GIL — but Python 3.13 with `--disable-gil` (PEP 703 free-threaded build) drops that guarantee. The `trace` property's `list(self._trace)` snapshot also relies on the same atomicity. Same single-thread argument as F3 holds: today only the worker thread mutates, and the only reader is `target.run` post-`asyncio.to_thread` (after the worker has stopped), so the read-after-write ordering is enforced by `await`. But coverage on free-threaded CPython is implied by `requires-python = ">=3.11,<3.14"` and PEP 703 is on by 3.13t.

Same fix as F3: a `threading.Lock` around append and snapshot, or move to `collections.deque` (its `append` is documented thread-safe). A future-proofing note in the docstring acknowledging the GIL-dependency would also work.

### F5. [MEDIUM] `pipeline.query` exception path leaves wrapper hooks with stale `_next_idx`

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/target.py:301-313`

`AgentDojoTarget.run` retries `pipeline.query` up to 3 times. The `_MessageStreamHook` and `_CatalogEditHook` instances are reused across attempts (built once in `build_pipeline`, line 337-338, 343). If attempt 1 raises mid-conversation (e.g. an LLM 500), `_next_idx` may have advanced past messages that don't exist in attempt 2's reset-conversation. The next attempt re-runs `SystemMessage` → `InitQuery` → first LLM call → `msg_hook`, sees `messages[:]` with low indices and `self._next_idx > len(messages)`, and silently emits nothing (the `while` loop in pipeline_bridge.py:183 immediately exits).

Net effect: a retried run can produce zero `agent_trace_message_NNNN` events even though messages flow through the pipeline. Not a thread-safety bug, but a state-management bug that the bridge surfaces. The same applies to `_tool_response_counter` on the wrapper — although that's mitigated because the wrapper is created **once** in target.run line 279 and is the runtime passed back into each retry, so the trace and the responses counter stay consistent with each other (just monotonically ahead of any per-retry message indices).

Fix: reset `_next_idx = 0` at the start of each `pipeline.query` attempt, either by creating a new `_MessageStreamHook` per attempt or by exposing a `reset()` method on the hook.

### F6. [LOW] `loop` capture happens at run() entry; a controller using a different loop per task is fine, but a single Target reused across `asyncio.run` calls would fail

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/target.py:248`

`asyncio.get_running_loop()` returns whatever loop entered the coroutine. The `TargetFactory.singleton` pattern (controller.py:103-118) would re-use one Target instance across multiple tasks in one `Controller.run()` — fine, same loop. But if a user constructs an `AgentDojoTarget` outside the controller and reuses it across separate `asyncio.run(...)` calls, `loop` would be torn down between runs. Today this isn't a path the controller exercises (controller creates a fresh `WrappedFunctionsRuntime` per `Target.run` via the wrapper's `__init__`), so the issue is upstream of the test surface.

Worth a sentence in target.py's `run()` docstring: "the loop is captured per-call; the Target instance does not pin a loop across runs."

### F7. [LOW] Trajectory thread-safety check on `emit` from worker thread

**Location**: `/Users/simonsure/research/superred/superred/src/superred/core/types/trajectory.py:73-94`

`Trajectory.emit` is documented thread-safe and uses a `threading.Lock`. The `FilteredTrajectory._push` it forwards to also takes a lock. The worker thread calls `emit` via `self._emit(ObservableEvent(...))` (wrapper line 253, 276, 334; pipeline_bridge line 185); this is `trajectory.emit` after passing through `Target.run`'s `emit` arg. So observables from the worker thread land via locks: correct.

One subtle point: the controller's middleware pipeline for `send_event` does `compose(trajectory_recorder(trajectory), security_domain_filter(scope))(channel.send)`. The `trajectory_recorder` middleware calls `trajectory.emit(event)` and `trajectory.emit(response)` inside an `async def` (middleware.py:71-77), which is executed by the optimizer-thread-loop. Meanwhile the worker thread calls `trajectory.emit(observable_event)` via the `emit` callback. Both go through the same `threading.Lock`, so this is safe. INFO-level note: this is one of the two places where the trajectory is touched by both threads concurrently; the lock handles it.

### F8. [LOW] `EventChannel.send` is "must be called from event loop thread"; the wrapper's `_await_event` complies via `run_coroutine_threadsafe`

**Location**: `/Users/simonsure/research/superred/superred/src/superred/core/channel.py:133-160`

`EventChannel.send` is async and must run on the loop. The wrapper schedules `self._send_event(event)` (which is `compose(trajectory_recorder(...), security_domain_filter(...))(channel.send)`) on the captured loop via `run_coroutine_threadsafe`. That executes on the loop thread, so `await self._queue.put(envelope)` and `await future` happen on the loop. Worker thread blocks on `future.result()` (the concurrent.futures future), receives the EventResponse after the optimizer responds. Correct.

The deadlock concern in the prompt — "if the optimizer's `on_event` callback synchronously blocks on something owned by the worker thread, do we deadlock?" — is answered by the channel design. `on_event` is `async def`; if it does `time.sleep(forever)` or busy-loops, the loop thread is stuck and the worker's `future.result()` never returns. That's a programmer error in the optimizer, not a bridge bug, but worth noting in optimizer docs. If `on_event` does `await asyncio.to_thread(synchronous_blocking_call)` and that blocking call needs a resource held by the worker thread, the deadlock is real. Per the test pattern in `test_concurrent_isolation.py`, optimizers are expected to be pure async.

### F9. [INFO] `ALL_FUNCTIONS` is module-level but only contains immutable rebound `Function` instances

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/tool_registry.py:395-411`

`ALL_FUNCTIONS` is a `list[Function]`, `TOOL_REGISTRY` is a `dict[str, RegistryEntry]`. Both are populated once at import (`_build_registry()`), never mutated. Each Target's `ToolCatalog.from_seed(ALL_FUNCTIONS)` calls `tuple(seed_functions)` (tool_catalog.py:173), so the per-Target catalog's `_seed` is a fresh tuple of the same `Function` instances — but the `Function` objects themselves are reused by reference.

The `Function` objects carry `run` callables (the upstream tool bodies). These bodies receive the per-call env from the pydantic root, so different Targets calling the same `Function.run` with their own envs don't interfere. The only shared mutable state would be if a tool body mutated module-level state, which the upstream suites don't do for v1.

The `test_no_module_level_mutable_state_in_target_package` test (test_concurrent_isolation.py:310-353) enforces this audit. Good.

### F10. [INFO] `agentdojo.logging.LOGGER_STACK` is a `contextvars.ContextVar`

**Location**: `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/logging.py:16`

`AgentPipeline.query` and `ToolsExecutionLoop.query` call `Logger().get()` (agent_pipeline.py:176, tool_execution.py:145). `LOGGER_STACK` is a `ContextVar` with default `[]`; `Logger.get()` returns the top of the stack or `NullLogger()`. Because nobody in our port enters a `Logger` context, every call returns `NullLogger()` which has `log(*args, **kwargs): pass`. ContextVar values are per-task-per-thread, so two parallel Targets each get their own implicit `NullLogger`. No cross-contamination.

If upstream introduces a shared `Logger.__enter__` somewhere that the port doesn't override, parallel targets would push to the same context — but each `asyncio.to_thread` worker gets a copy of the contextvar context at creation, so it would still be isolated.

### F11. [INFO] `_CatalogEditHook` is shared across catalog phases within one pipeline; safe because reused on the same worker thread

**Location**: `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/src/agentdojo_target/pipeline_bridge.py:338-348`

Same `_CatalogEditHook` instance spliced both outside the `ToolsExecutionLoop` and inside it. The hook's `query` is called sequentially per turn. The hook's state (`_catalog`, `_wrapper`, `_send_event`, `_loop`) is all read-only after construction; the per-call ops are stateless. Safe.

## Strengths

- **Per-Target isolation is enforced by construction**: `WrappedFunctionsRuntime`, `ToolCatalog`, `CompositeEnvironment`, `_MessageStreamHook`, `_CatalogEditHook`, and the wrapper's `_trace`/`_tool_response_counter` are all constructed in `AgentDojoTarget.run()` (target.py:278-285, 295). The factory pattern in `TargetFactory.create()` ensures concurrent tasks never share these.
- **Loop captured per-run, not per-Target**: `loop = asyncio.get_running_loop()` (target.py:248) means a fresh Target reused for multiple `run()` calls would still bind to the current loop each time.
- **EventChannel's poison + reject design**: `set_error` poisons future sends and rejects queued envelopes (channel.py:186-208), so a dead optimizer turns into a clean exception on the wrapper's `future.result()` rather than a hang — as long as the optimizer's failure happens after the wrapper queued its send.
- **Trajectory thread-safety primitives are solid**: `threading.Lock` around all mutations, `FilteredTrajectory` is push-based with its own lock and holds no reference to the parent. Encapsulation via `__slots__` prevents accidental coupling.
- **`asyncio.run_coroutine_threadsafe` correctly bridges from worker thread to loop thread**: the loop thread services the queued coroutine (`channel.send` → `await future`) while the worker thread blocks on `future.result()`; the loop is never blocked by the wrapper, so other tasks (including the optimizer) continue.
- **The concurrent-isolation test suite exists and stresses N=4 and N=8 parallel targets** (`test_concurrent_isolation.py`). The fingerprint check covers per-task env, prompt, and overlay isolation — exactly the cross-contamination class that module-level state would surface.
- **`Function` and `Controllable` are frozen dataclasses**: even though they're shared by reference, callers cannot mutate them.

## Open questions

1. **Should `_await_event` use a timeout on `future.result()`?** A hard timeout would prevent the worker-wedge in F1 but adds a magic number. An alternative: have the controller keep a registry of outstanding `concurrent.futures.Future` objects and `.cancel()` them all on shutdown. Worth a design decision.
2. **`pipeline.query` retry semantics**: F5 noted that `_next_idx` doesn't reset between attempts. Is that the intended behavior (the trace is the cumulative attempt history) or a bug (each attempt should look like a fresh run)? Upstream behavior is "messages are re-built from SystemMessage + InitQuery on each attempt"; the hook's monotonic counter does not match. Clarify in `ASSUMPTIONS.md`.
3. **Free-threaded CPython (3.13t)**: project supports 3.13. F3, F4 may matter if the project ever runs on `python3.13t`. Today the venv at `.venv/bin/python` is GIL-enabled. Worth a CI note.
4. **`_MessageStreamHook` is instantiated once per `build_pipeline`** (pipeline_bridge.py:337). If a future code path constructs the pipeline once and runs it N times (cross-task reuse), `_next_idx` would carry across. Not a current path — `build_pipeline` is called inside `AgentDojoTarget.run()` (target.py:285) so it's per-run — but the docstring says "Construction is per-run" (pipeline_bridge.py:163) and that's worth pinning with an assertion or a test.
5. **What's the policy if `send_event` raises** (e.g. the channel was closed before the wrapper got there)? Today the exception flows through `future.result()` to the worker thread, then through `pipeline.query`, then caught by `target.run`'s broad `except Exception` (target.py:309), which logs and breaks the retry loop. Net effect: a single failed bridge call aborts the whole run. Is that the desired behavior, or should specific exception types (e.g. `RuntimeError("Cannot send on a closed EventChannel")`) be re-raised at the controller boundary so the controller's classification picks them up cleanly?
