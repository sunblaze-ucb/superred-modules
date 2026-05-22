# Runtime Wrapper Review

## Summary

`WrappedFunctionsRuntime` is a thin, well-organized subclass of `agentdojo.functions_runtime.FunctionsRuntime` that splices three side effects (trace recording, per-call event firing, observable emission) around `run_function`. The dispatch logic (canonical vs. attacker) and the sync-to-async bridge are correct in the happy path and faithfully preserve upstream behaviour, but several edge cases warrant fixes: the bridge can deadlock if the loop is the worker's loop, `_await_event` does not handle exceptions from `send_event` cleanly, the attacker-path inject value is treated as the agent-seen return type incorrectly, and `_execute_nested_calls` will recursively flow back through our wrapper (faithful, but undocumented). Several smaller findings cover serialization edge cases, refresh-time races, write-path emission inconsistency, and ergonomics.

## Findings

### [HIGH] `_await_event` will deadlock if called on the same loop's thread

`runtime_wrapper.py:197-200` calls `asyncio.run_coroutine_threadsafe(self._send_event(event), self._loop).result()`. The intended caller flow is `target.run` -> `asyncio.to_thread(pipeline.query, ...)` (`target.py:303-308`) so `pipeline.query` and its inner `runtime.run_function` are on a worker thread, distinct from the loop, which is correct. However the wrapper has no guard: if a future code path (e.g. running the pipeline directly from an async coroutine without `to_thread`, or unit tests that pass `loop=asyncio.get_event_loop()` and call `run_function` from a coroutine on that loop) ever happens, `future.result()` blocks the loop on a future that can only be completed by that very loop. This is a classic re-entrancy deadlock.

Recommended fix: at the top of `_await_event` assert `self._loop is not asyncio.get_running_loop_or_none()` (or, more simply, `threading.get_ident() != self._loop_thread_ident`). Capture `self._loop_thread_ident` once at construction or first call. Raise a clear `RuntimeError` so the misuse fails fast instead of hanging. Also worth a docstring note on the contract.

### [HIGH] `_await_event` does not bound the wait or surface cancellation cleanly

`runtime_wrapper.py:199-200` calls `future.result()` with no timeout. If the optimizer never responds (deadlock on the receiver side, a buggy `on_event` that raises before `respond`, the channel being closed from underneath the wrapper), the worker thread blocks indefinitely while holding no lock but tying up a thread from the default executor pool. There is also no handling for `concurrent.futures.CancelledError` should the loop be torn down while the wrapper is waiting; `future.result()` would then raise `CancelledError` synchronously and the wrapper would let that escape `run_function`, breaking AgentDojo's pipeline contract (which expects `run_function` to return `(value, error_str)` not raise).

Recommended fix: wrap `future.result()` in a try/except that catches `concurrent.futures.CancelledError` and any exception the channel might propagate via `EventEnvelope.reject` (see `superred/core/channel.py:88-102`). Convert those into a synthesized `ControllableNoInjection` so the canonical path still returns the legitimate value, or re-raise into the canonical error tuple. Optionally add a generous per-call timeout (e.g. 600s) configured at construction; an infinite block is rarely the right default.

### [HIGH] Attacker-path `agent_seen_value` type is mislabeled and may not match agent rendering

`runtime_wrapper.py:306-309`: when the optimizer responds with `ControllableInjection`, `agent_seen_value` is set to `response.value` (a `str`), then returned as `FunctionReturnType` (`(agent_seen_value, None)` at line 311). Downstream `ToolsExecutor` formats via `tool_result_to_str` (`agentdojo/agent_pipeline/tool_execution.py:22`), which only special-cases `BaseModel` and `list`; a `str` falls through to `str(tool_result)` and is rendered verbatim into the LLM prompt. ASSUMPTIONS.md A.3 explicitly affirms this design.

Two concerns nonetheless: (1) the canonical path returns the *raw* legitimate value (a `BankAccount`, a `list[Email]`, etc.) so the YAML formatter renders a structured doc, while the attacker-injection path returns a raw string — the rendered prompt shape differs. (2) When no injection arrives (line 309: `agent_seen_value = entry.fake_return`), `fake_return` is `Any` (could be a dict, BaseModel, list, etc.) and YAML formatter handles only the three branches above, so an attacker who registers `fake_return={"x": 1}` gets a YAML dict, but `fake_return=42` (an int) falls through to `str(42)`. Faithful, but worth pinning a test: an attacker's `fake_return=[{"k":1},{"k":2}]` works only if every item is `str|int|BaseModel` (line 33-38 of upstream's `tool_result_to_str` raises `TypeError` on `dict` list items). Tests in `test_runtime_wrapper.py:191-213` only exercise dict and string fake returns, never lists of dicts.

Recommended fix: document the prompt-rendering polarity asymmetry in ASSUMPTIONS.md A.3 (canonical = structured, attacker-injected = raw string). Add a test that registers a `fake_return=[{"k":1}]` and confirms the upstream `tool_result_to_str` raises (or that the wrapper coerces ahead of time). At minimum a `_serialize_for_event(value)` guard before returning attacker values would let the agent see consistent string content regardless of `fake_return` shape, and would remove the dict-list `TypeError` hazard.

### [MEDIUM] `_execute_nested_calls` re-enters our overridden `run_function` for every nested call

Upstream `FunctionsRuntime.run_function` calls `self._execute_nested_calls` (`functions_runtime.py:279`), which loops over `kwargs` looking for `FunctionCall`-typed arguments and recursively calls `self.run_function(...)` with `raise_on_error=True` (line 241-243). Because we override `run_function`, every nested call will (correctly) fire its own per-call event, emit its own observable, and append to `self._trace`. This is *faithful* (the nested call really did happen and the agent's intent should be visible), but it has two consequences worth flagging:

1. The trace order is depth-first: an outer call appears in the trace *before* its nested children (line 213-216: append happens up-front), which differs from `functions_stack_trace_from_messages` (which only walks the message log and would see only the top-level call). Layer-1 predicates that consume `function_call_trace` may see entries that upstream's `*_from_traces` helpers do not. Confirm whether the SecurityClaim predicates iterate the trace literally or use a parallel upstream-style extractor.
2. If a nested call's controllable is in scope, its per-read event fires and the optimizer can inject into the nested result, which then becomes the `kwargs` value for the outer call. This is more expressive than upstream and not obviously documented. Mention in ASSUMPTIONS.md.

Recommended fix: add an ASSUMPTIONS.md entry documenting both behaviours (eager trace + nested-event injection). Add a test that exercises a nested call (e.g. an agent that nests `get_iban` inside `send_money` args) and asserts both the trace order and the injection semantics on the nested call.

### [MEDIUM] `refresh_functions` is not thread-safe vs. concurrent reads

`runtime_wrapper.py:184-191` assigns `self.functions = {f.name: f for f in self._catalog.functions_for_runtime()}` in one shot. The call site is `_CatalogEditHook.query` (`pipeline_bridge.py:139`), which runs from the worker thread (inside `pipeline.query` -> `asyncio.to_thread`). The `ToolsExecutor` and OpenAILLM both read `runtime.functions` from the *same worker thread* in sequence within `pipeline.query` (no concurrency inside a single `pipeline.query`), so today this is safe.

The risk: if anywhere `target.run` ever runs internal branches concurrently (the Target interface explicitly allows this — see CLAUDE.md "Target run() can have internal parallelism") or if a future change parallelizes pipeline elements, two threads could read `self.functions` while a refresh is rebinding it. A dict assignment is atomic for the reference swap in CPython, but a midstream `runtime.functions.values()` iteration while another thread reassigns can briefly hold a stale view; worse, a *mutating* iteration (an LLM element that snapshot-iterates while the hook rewrites) could miss a brand-new attacker tool.

Recommended fix: hold a `threading.Lock` (or an `asyncio.Lock`) around `refresh_functions` and around reads of `self.functions` in the wrapper's own methods. Alternatively, document that callers must serialize pipeline turns; current behaviour relies on AgentDojo's strictly sequential turn structure and that invariant should be pinned in a test if any internal-parallelism target is added.

### [MEDIUM] `_serialize_for_event` does not cover several common shapes

`runtime_wrapper.py:80-104`:

- **`None` returns**: `isinstance(None, str)` is False, `isinstance(None, BaseModel)` is False, `json.dumps(None)` returns the literal string `"null"`. The agent then sees `"null"` as the event answer. For a tool returning `None` on success this conflates "no value" with "null literal". Not catastrophic but worth a unit test.
- **Nested dicts containing `datetime`**: `_json_fallback` only handles `BaseModel`, `isoformat`-bearing objects, and `enum.value`-bearing objects. A `dict` whose *values* are datetimes goes through `json.dumps` recursively, and the nested datetime IS handled because `default=_json_fallback` fires on the leaf. Confirmed correct.
- **`StrEnum` members that are also strings**: `isinstance(value, str)` catches `StrEnum` instances at the top of `_serialize_for_event` (`StrEnum` inherits from `str`), so the enum's underlying string value is returned verbatim. Correct but worth a comment.
- **Pydantic models nested inside `list`**: `json.dumps([model_a, model_b])` invokes `default=_json_fallback` per item, which returns `model.model_dump()` (a dict). Final output: a JSON list of dicts. Correct.
- **Sets, tuples, bytes**: `json.dumps({1, 2})` raises `TypeError`, falls to `repr({1, 2})` -> `"{1, 2}"` (not valid JSON). Bytes: `repr(b"x")` -> `"b'x'"`. Acceptable but the optimizer cannot round-trip these.
- **Cyclic references**: `json.dumps` raises `ValueError("Circular reference detected")`, falls to `repr(value)` which itself may recurse. A truly cyclic structure crashes with `RecursionError` instead of producing a string.

Recommended fix: add a unit test matrix (None, list[BaseModel], dict[str, datetime], StrEnum, bytes, set, tuple, cyclic). The cyclic guard is the only one likely to bite in practice; an explicit `try: ... except RecursionError: return repr(value)` would harden it. The `None`-vs-`"null"` ambiguity is worth a deliberate decision.

### [MEDIUM] Canonical write path's observable predicate is inverted from intent

`runtime_wrapper.py:275-281`: after a canonical call, if `function not in READ_CTRLS` *and* `error is None`, a `write_call:*` observable fires. The `READ_CTRLS` check is the actual read/write classification (since `READ_CTRLS` is keyed by every read tool name). However:

1. The classification should ideally consult `WRITE_FUNCTION_NAMES` (positive identification) rather than `not in READ_CTRLS` (negative identification). If a new canonical tool is added that is neither read nor write (none exist today, but the registry could in principle classify "neutral"), it would silently be tagged as a write. Tests in `tool_registry.py` enforce exhaustive read/write classification, so this is a latent rather than active bug, but the wrapper should depend on the positive set for clarity.
2. The block is gated on `error is None`. If a write tool errors (e.g. `send_money` with a bad recipient), no `write_call:*` observable fires. SecurityClaim predicates that look for "did the agent *attempt* a write?" miss this — only successful writes are surfaced. Brief Section 5.c says "every tool call (with the function name and args the agent emitted)" and "every return value" — attempts ought to be visible too. Compare with `_emit_agent_tool_response` (line 285), which DOES fire on errors (`error` is included in the payload), so the asymmetry is jarring.

Recommended fix: switch to `function in WRITE_FUNCTION_NAMES` for the classification check. Drop the `error is None` gate or move it into the observable payload (e.g. `content={"function": ..., "args": ..., "error": error}`) so attempts are visible to Layer-2 predicates that hunt for adversarial-write intent. Add a test that errors `send_money` and asserts the observable fires.

### [LOW] `_trace` records `FunctionCall(args=dict(kwargs))` losing `id` and `placeholder_args`

`runtime_wrapper.py:213-216`: the trace entry only carries `function` and `args`. Upstream `FunctionCall` has `id` and `placeholder_args` fields (`functions_runtime.py:48-51`) used by OpenAI/Anthropic for tool-call correlation. The wrapper never receives the agent's `tool_call.id` because `runtime.run_function` is called by `ToolsExecutor` with just `(env, tool_call.function, tool_call.args)` and no id (`tool_execution.py:103`). So this is an upstream limitation, not a wrapper bug. But `target.py:388-395` (`_function_call_to_dict`) serializes `fc.id` as `None` regardless, which is correct given the loss. Worth a brief comment in the trace-record docstring acknowledging that `id` and `placeholder_args` will always be `None` here.

### [LOW] `_emit_agent_tool_response` always emits even when the controllable is out of scope

`runtime_wrapper.py:317-342`: the observable is emitted unconditionally per tool call. The observable's security domain is `AGENT_TRACE_TOOL_RESPONSES_TAG` (per `observables.py:119-129`), and the trajectory's filter will hide it from an optimizer whose scope does not include that tag. So functionally this is correct (the observable is on the trajectory, the optimizer just won't see it). However, the observable carries `_serialize_for_event(value)` which may include the **post-injection** value the agent saw — if injection happened on a tag the optimizer DOES have, but the agent_trace_tool_responses tag is out of scope, the optimizer can't read the response, which is the intended polarity. Confirm by reading the trajectory filter contract — `Trajectory._filter` (`trajectory.py:84-94`) tests `scope_includes(scope, domain)`, so an optimizer with scope `{AGENT_TRACE_TOOL_RESPONSES_TAG}` sees it; one without it doesn't. Correct.

Edge case: if `_serialize_for_event(value)` is expensive (large list of pydantic models), it runs even when no consumer is in scope. Premature optimization to skip, but worth a comment.

### [LOW] `_serialize_for_event` re-serializes the injected string in `_emit_agent_tool_response`

`runtime_wrapper.py:338`: `_serialize_for_event(value)` where `value` may already be a string (the injected `response.value`). The fast path at line 87-88 returns the string verbatim, so cost is one `isinstance` check. Fine.

### [LOW] `_run_attacker` does not check for `error` paths from `_await_event`

`runtime_wrapper.py:292-311`: unlike `_run_canonical`, the attacker path never invokes any actual body, so there's no `error` to capture. The function returns `(agent_seen_value, None)` always. However, if `_await_event` raises (see HIGH on exception handling), the exception propagates out of `_run_attacker` and out of `run_function`, breaking the AgentDojo `(value, error_str)` contract. Same root issue as the HIGH finding above; fixing that fixes this.

### [INFO] `refresh_functions` could be a no-op or shrink, but it always rebuilds

The dict is fully reconstructed even if the catalog has not changed. Performance is negligible (74 entries today) but the operation could be made idempotent / skipped when `len(self._entries) == len(self.functions)` and keys match. Not worth fixing.

### [INFO] Trace can grow unbounded; no max-trace guard

`self._trace` is `list[FunctionCall]` that accumulates per-run. A pathological agent in a long-running run could OOM. Mitigated by AgentDojo's `ToolsExecutionLoop.max_iters` (default 15 per turn) and the outer 3-retry loop, so practical exposure is small. Not worth fixing.

## Strengths

- Clean dispatch: `run_function` -> `_run_canonical` / `_run_attacker` partition is easy to read, and the `entry is None` fallthrough to `super().run_function` preserves upstream's `ToolNotFoundError` shape verbatim.
- Trace is recorded up-front (line 213) so partial failures and exceptions still produce a meaningful trace; `target.py:324-326` captures it even when `pipeline.query` raises.
- Observable emission policy is layered correctly: `read_data_field:*` mirrors carry pre-injection legitimate values for the matching scope, `write_call:*` for the agent-trace tool-calls scope, `agent_trace_tool_response_NNNN` for the response scope. The triple lets a Layer-2 predicate reconstruct what the agent saw vs. what was legitimate.
- Test coverage in `test_runtime_wrapper.py` exercises every documented branch (canonical read with/without injection, canonical write, attacker register/replace, trace ordering, unknown tool, observable index monotonicity, error path). The `loop` fixture spins up a real background loop, matching production usage.

## Open questions

- Should the wrapper coerce attacker `fake_return` values through `_serialize_for_event` before returning so prompt rendering is uniform (`tool_result_to_str` only handles three return shapes well)? Today an attacker registering `fake_return=42` gets `str(42)`, but `fake_return={"x": 1}` gets the dict and yaml-renders it. The asymmetry may matter for faithfulness comparisons.
- Is the `_execute_nested_calls` re-entrancy expected to fire per-read events on nested calls, or should nested calls inherit the parent's injection decision? The current behaviour multiplies the optimizer's exposure on agents that use nested syntax. Confirm with the brief author.
- Should `refresh_functions` be transactional with respect to `run_function` calls in flight? AgentDojo today serializes pipeline turns, but a target that uses internal parallelism (e.g. branched search) could race. Pin the contract with a test or add explicit locking.
- The wrapper imports `READ_CTRLS` directly from `controllables.py`, which couples the runtime to the static read-controllable map. If a future attacker capability registers a *read*-shaped attacker tool (today only `registered`/`replaced` writes are modeled), the wrapper would miss the per-read event for it because `READ_CTRLS.get(function)` returns `None`. Is read-shaped attacker tool registration in scope for v2?
