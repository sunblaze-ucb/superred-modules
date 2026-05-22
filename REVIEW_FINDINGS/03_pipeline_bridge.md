# Pipeline Bridge Review

## Summary

`pipeline_bridge.py` constructs the upstream `AgentPipeline` with two spliced hooks (`_CatalogEditHook`, `_MessageStreamHook`) so the optimizer can mutate the tool catalog per turn and observe the chat-message stream in real time. The construction is faithful to AgentDojo's `no_defense` shape and the per-turn semantics of catalog mutation are coherent for the happy path. However, several real concerns surface: the `_build_llm` dispatch is far narrower than upstream (5 providers missing) without an extension path, the Anthropic `-thinking-` parser uses `partition` where upstream uses `split` and accepts inputs upstream rejects, the `_CatalogEditHook` swallows only `ValueError` and `JSONDecodeError` so unexpected exceptions inside `apply_*` will tear down the run, the hook's exception polarity loses provenance about which slot failed, the message-stream hook's serialiser does not survive nested `FunctionCall` arguments, the pipeline silently drops upstream's `tool_output_format` and `max_iters` knobs, and there are several smaller concerns around cursor monotonicity assumptions, no test coverage for the dual catalog-hook firing, and the `applied_any` semantics that trigger spurious refreshes.

## Findings

### [HIGH] `_build_llm` covers only 2 of the 7 upstream providers and has no extension hook

`pipeline_bridge.py:229-279` dispatches only on `openai/` and `anthropic/`. Upstream `agent_pipeline.py:70-125` supports seven providers: `openai`, `anthropic`, `together`, `together-prompting`, `cohere`, `google`, `local`, `vllm_parsed`. The docstring acknowledges the gap ("Cohere, Google, Together, vLLM" — though it elides `local`, `together-prompting`, and `vllm_parsed`) and the `NotImplementedError` message points at upstream's source.

Concrete consequences:

- **`together/...`**: upstream uses `openai.OpenAI` with `base_url="https://api.together.xyz/v1"` and `TOGETHER_API_KEY`. The bridge could trivially fall into the openai branch with `api_base` override, but currently raises.
- **`local/...`** and **`vllm_parsed/...`**: upstream uses an `openai.OpenAI` against `http://localhost:{LOCAL_LLM_PORT}/v1`. Same trivial alias possible; bridge raises.
- **`cohere/...`** and **`google/...`**: non-trivial bridges (different SDKs), but their absence means the port covers a narrower experiment surface than upstream.
- **`together-prompting`**: uses `PromptingLLM` (a separate element type), not bridgeable as a trivial alias.

This is the biggest faithfulness gap in the bridge. The brief promises an AgentDojo port; cutting providers without a registered escape hatch means experiments that picked a non-OpenAI model upstream cannot be replayed via this port. Today only `test_build_llm_unknown_provider_raises` exists; it pins the limitation but does not extend it.

Recommended fix: at minimum add the trivial openai-alias providers (`together`, `local`, `vllm_parsed`) since the only difference is `api_base`. Optionally expose an extension point — e.g. `LLM_BUILDERS: dict[str, Callable[[str, str | None, str | None], BasePipelineElement]]` that callers can register into for `cohere`/`google`/etc., or accept a pre-built `BasePipelineElement` instead of a model id and dispatch only when a string is passed. Document the chosen extensibility story in the module docstring.

### [HIGH] Anthropic `-thinking-` parsing diverges from upstream in three observable ways

`pipeline_bridge.py:264-272` uses `model_name.partition("-thinking-")`. Upstream `agent_pipeline.py:76-83` uses `model.split("-thinking-")` with `len(elements) != 2` guard.

Divergences:

1. **Multiple occurrences silently accepted.** Upstream raises `ValueError("Invalid Thinking Model Name")` if `model.split("-thinking-")` produces more than 2 elements. The bridge's `partition` always splits on the first occurrence; a model id like `anthropic/claude-3-thinking-100-thinking-200` becomes base=`claude-3`, budget=`100-thinking-200` and then fails inside `int(...)` with the misleading diagnostic `"thinking suffix must be an integer, got '100-thinking-200'"`. The failure mode and the error message both differ.
2. **Empty base or empty budget silently accepted by parser.** `anthropic/-thinking-1024` -> base=`""`, budget=`"1024"`, `AnthropicLLM(client, "", thinking_budget_tokens=1024)`. Upstream behaves the same way (split gives `["", "1024"]`, also len 2). Both build a broken client; the bridge could be friendlier than upstream here.
3. **No empty-budget error message tuning.** `anthropic/claude-3-thinking-` -> budget=`""`, `int("")` raises `ValueError: invalid literal for int() with base 10: ''`. The bridge wraps that to "must be an integer, got ''", which is acceptable. Upstream surfaces the raw `int("")` traceback.

The test `test_build_llm_anthropic_thinking_suffix_invalid_int` covers the simple `banana` case but neither the multi-suffix nor the empty-budget edge cases.

Recommended fix: switch to `split("-thinking-")` and raise the same `ValueError` upstream raises when `len(elements) != 2`, with a clear message including the original model id. Add tests for: (a) double `-thinking-` substring, (b) empty base, (c) empty budget, (d) trailing space inside budget (`"-thinking-1024 "`). Consider also asserting the parsed `base_model` is non-empty.

### [HIGH] `_CatalogEditHook._try_apply` only catches `ValueError` and `JSONDecodeError`

`pipeline_bridge.py:100-115`:

```python
try:
    method(payload)
except ValueError as exc:
    logger.warning("Catalog mutation rejected: %s", exc)
```

This is too narrow. The `apply_*` methods can plausibly raise:

- `TypeError` — e.g. if a payload value has the wrong shape inside pydantic `create_model` (the catalog wraps `create_model` in `try/except Exception` at `tool_catalog.py:132-135`, but the `Function(...)` construction at `tool_catalog.py:142-150` does NOT — a bad `return_type` value would surface as a pydantic `ValidationError` which is NOT a `ValueError` in pydantic v2; it's a `pydantic.ValidationError` that inherits from `ValueError` for compat in v1 but in v2 inherits from `Exception` only).
- `pydantic.ValidationError` — see above; pydantic v2's exception inheritance is not `ValueError`.
- `KeyError` — if any future `apply_*` does `payload["foo"]` directly without going through `_require_str`.
- `AttributeError`, `RuntimeError` — from deep inside the placeholder builder.

The docstring claims "Application errors (malformed payloads, unknown tool names, etc.) are logged and *not* propagated; the hook treats the LLM call's progress as more important than enforcing payload correctness." The intent is clearly "swallow anything malformed". But the implementation enforces ONLY `ValueError`, so the contract and the code disagree.

The test `test_hook_swallows_value_error_from_apply` only exercises the duplicate-register case. There's no test for an injection payload that causes a non-`ValueError` exception inside `apply_*`. The brief's promise that "the optimizer cannot crash the target via a bad catalog payload" depends on a wider catch.

Recommended fix: catch `Exception` and log with the controllable name plus the exception class, e.g. `logger.warning("Catalog mutation %s rejected (%s): %s", method.__name__, type(exc).__name__, exc)`. Include `logger.exception(...)` so the traceback lands in the logs at DEBUG. Add a test that injects a payload triggering a pydantic `ValidationError` (e.g. malformed `parameters_schema` that bypasses the existing `try/except Exception` guard) and asserts the run continues. Consider also documenting the catch breadth on the class docstring.

### [HIGH] `_message_to_jsonable` does not handle nested `FunctionCall` in tool-call args

`pipeline_bridge.py:195-221` reduces a `ChatMessage` TypedDict to a plain dict. The serialiser handles top-level `tool_calls`/`tool_call` by extracting `(function, args, id)`. For `args`, it does `dict(tc.args)` — but `FunctionCallArgTypes = str | int | float | bool | None | dict | list | FunctionCall` (`agentdojo/functions_runtime.py:54`). A `FunctionCall` argument value would be preserved verbatim as a `FunctionCall` pydantic model in the resulting dict, which is NOT JSON-serialisable. Downstream the trajectory persistence layer or any consumer that calls `json.dumps` on the observable content will crash.

Concrete case: an agent that emits a nested call like `send_money(recipient=get_iban(account="primary"), amount=100)` would produce a `FunctionCall` where `args["recipient"]` is itself a `FunctionCall`. Today no test exercises this; the only `tool_calls` test (`test_message_stream_hook_serialises_tool_calls`) uses `args={}`.

The other field omissions are minor:

- `FunctionCall.placeholder_args` (Optional[Mapping]) — never captured. Per upstream `functions_runtime.py:50`, this is used by some prompting LLMs for placeholder substitution. If it's non-None, it's silently dropped.
- `FunctionCall.id` — captured, good.
- `ChatToolResultMessage.tool_call` (singular) — captured at lines 214-220, good.
- `ChatAssistantMessage.content` — captured as the raw `list[MessageContentBlock]`. The content blocks themselves are TypedDicts (text/thinking/redacted_thinking) and serialise fine to JSON because they're dicts. Good.
- `name` field — extracted at line 200 but NOT present in any upstream `ChatMessage` variant. Harmless because of `if key in msg` guard, but the comment in the docstring "captures all ChatMessage variants" should clarify this is a defensive over-cover.

Recommended fix: recursively convert `FunctionCall` instances inside `tc.args` via a small helper, e.g.

```python
def _arg_to_jsonable(v: Any) -> Any:
    if hasattr(v, "function") and hasattr(v, "args"):  # FunctionCall
        return {"function": v.function, "args": {k: _arg_to_jsonable(av) for k, av in v.args.items()}, "id": v.id}
    if isinstance(v, dict):
        return {k: _arg_to_jsonable(av) for k, av in v.items()}
    if isinstance(v, list):
        return [_arg_to_jsonable(av) for av in v]
    return v
```

Then `"args": {k: _arg_to_jsonable(av) for k, av in tc.args.items()}`. Capture `tc.placeholder_args` if present. Add a test that nests a `FunctionCall` inside `args` and asserts the emitted observable payload is JSON-serialisable.

### [MEDIUM] Pipeline silently drops upstream's `tool_output_format` and `max_iters` knobs

`pipeline_bridge.py:346-348` constructs `ToolsExecutionLoop([ToolsExecutor(), msg_hook, hook, llm, msg_hook])` — `ToolsExecutor()` uses the default `tool_result_to_str` (YAML formatter), and `ToolsExecutionLoop` uses the default `max_iters=15`. Upstream `PipelineConfig.tool_output_format` lets users pick `"json"` to use `json.dumps` instead; upstream `from_config` honours it (`agent_pipeline.py:197-200`). Upstream `ToolsExecutionLoop.max_iters` can be tuned per call.

For an AgentDojo port whose stated goal is faithfulness, hard-coding both is a quiet deviation. Experiments that picked `json` formatting upstream cannot be replayed.

Recommended fix: extend `build_pipeline` signature with `tool_output_format: Literal["yaml", "json"] | None = None` and `tools_loop_max_iters: int = 15`, threading them through to `ToolsExecutor` (via `partial(tool_result_to_str, dump_fn=json.dumps)` per upstream) and `ToolsExecutionLoop`. Document the defaults. Add tests asserting the formatter is the requested function and that `max_iters` is set.

### [MEDIUM] Pipeline shape deviates from `no_defense` more than the docstring implies

Upstream `no_defense` (`agent_pipeline.py:202-206`):

```python
tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter), llm])
pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
```

The bridge:

```python
tools_loop = ToolsExecutionLoop([ToolsExecutor(), msg_hook, hook, llm, msg_hook])
pipeline = AgentPipeline([SystemMessage(...), InitQuery(), hook, llm, msg_hook, tools_loop])
```

Three deviations from the strict no-defense shape:
1. `hook` added before the outer `llm` (intentional — fires catalog edits before turn 0).
2. `msg_hook` added after the outer `llm` (intentional — emits sys+user+first-assistant batch).
3. Inner loop body grows from 2 elements to 5 (intentional, but worth pinning in the docstring as "deliberate departure from no_defense").

The docstring at `pipeline_bridge.py:307-322` shows the new shape but does not explicitly call out that this DIVERGES from upstream's no_defense. A reader trusting that "this is the no_defense baseline" (per the brief) might assume the additional elements are pure passive observers; in fact they (a) consume controllable events and (b) mutate `runtime.functions`. The first is observational; the second is causal. Worth making the deviation explicit.

There is also one subtle behaviour the docstring should explain: the **inner** `_CatalogEditHook` fires only when the loop body runs at all, which happens only when the prior assistant message had tool_calls. If turn 0 returns no tool calls, the inner hook never fires; total catalog firings = 1 (the outer hook). If turn 0 has tool calls and turn 1 doesn't, total = 2 (outer + one inner). The docstring "fires before every subsequent turn" elides this — "before every LLM turn that the pipeline actually performs after turn 0" is more precise.

Recommended fix: add a "Deviations from no_defense" subsection to the `build_pipeline` docstring listing the three additions and stating clearly that they consume events and mutate runtime state. Tighten the "fires before every subsequent turn" claim.

### [MEDIUM] No test covers the dual `_CatalogEditHook` firing pattern

The hook fires once outside the loop and zero-or-more times inside it. The tests at `test_pipeline_bridge.py:184-205` only exercise a single invocation. Specifically untested:

- Two consecutive `hook.query(...)` calls on the same instance (mimicking outer + inner firing) — does the hook handle being called twice without state corruption? Today the hook is stateless except for the `_send_event`/`_loop` references, so two calls should produce 8 events total. But there's no test pinning that.
- `applied_any` semantics across calls — if turn 0's outer firing applies a register, the catalog mutates; if turn 1's inner firing applies another register on a different name, both should succeed. Today untested.
- Combination with the `_MessageStreamHook` cursor in the same loop iteration — there's no integration-style test that fires both hooks in sequence and verifies the message-stream cursor advances correctly while catalog edits happen.

Recommended fix: add an integration test that constructs the pipeline (without a real LLM), feeds a synthetic message stream through the elements in `tools_loop.elements`, and asserts: (a) catalog events fire on both invocations, (b) `_next_idx` cursor stays monotonic, (c) registered tools from outer firing are still visible after inner firing.

### [MEDIUM] `applied_any` triggers spurious `refresh_functions()` even when every apply was silently rejected

`pipeline_bridge.py:131-139`:

```python
applied_any = False
for ctrl, apply_method in ops:
    ...
    if isinstance(response, ControllableInjection):
        self._try_apply(apply_method, response.value)
        applied_any = True
if applied_any:
    self._wrapper.refresh_functions()
```

`applied_any` is set whenever the optimizer returned an `Injection`, regardless of whether `_try_apply` actually mutated. If all four optimizer responses are `Injection` with malformed payloads, `_try_apply` silently logs and skips four times, then `refresh_functions()` runs uselessly. Performance is negligible (74 entries today), but the semantic is "we refreshed because we ATTEMPTED to apply" rather than "we refreshed because we ACTUALLY applied". For diagnostics and future profiling this can mislead.

A tighter contract: `_try_apply` returns a `bool` indicating whether the catalog actually mutated; `applied_any` becomes "any apply returned True".

Recommended fix: thread a return value through `_try_apply` so `applied_any` reflects actual mutations. Optionally skip the refresh entirely if the catalog state is unchanged. Trivial fix; modest clarity gain.

### [MEDIUM] `_MessageStreamHook` cursor assumes append-only message lists

`pipeline_bridge.py:183-191`: the cursor only advances and never re-reads earlier indices. This is correct for the current pipeline (every element only appends to the message list), but fragile under future defenses:

- `repeat_user_prompt` defense (upstream `agent_pipeline.py:249`) inserts an `InitQuery()` inside the tools loop body, which appends a user message — still an append, cursor handles it.
- `spotlighting_with_delimiting` defense rewrites the system message at construction time, before any cursor read — also fine.
- Hypothetical future defense that rewrites earlier messages (e.g. a content sanitiser that redacts an old tool result) would silently get bypassed by the cursor (we'd never re-emit the rewritten message and the optimizer would have the stale value).

The docstring at lines 156-165 doesn't note this assumption. The brief specifies only `no_defense` for v1 so this is currently safe, but if the port grows to cover other defenses, this assumption needs revisiting.

Recommended fix: add a docstring note: "Assumes the message list is append-only across the run; defenses that rewrite earlier messages would silently bypass the cursor". Optionally, if the port grows to cover other defenses, switch to content-hash-based diffing instead of an index cursor.

### [LOW] `_CatalogEditHook` request payload is identical for all four events

`pipeline_bridge.py:133`: `ControllablePreCallEvent(controllable=ctrl, request="catalog edit slot")`. The same `request` string is used for all four slots. The optimizer disambiguates by `event.controllable.name`, which works but loses self-describing semantics. A request payload of `f"catalog edit slot: {ctrl.name}"` or even a small JSON `{"slot": ctrl.name}` would make trajectory dumps more readable.

Recommended fix: include the slot name in the request, e.g. `request=f"catalog_edit:{ctrl.name}"`. Update the test assertions to match.

### [LOW] `_build_llm` does not validate that `model_name` is non-empty

`pipeline_bridge.py:248-252`: `"/" not in model_id` is checked but `partition("/")` yielding empty model_name is not. `_build_llm("openai/", ...)` returns an `OpenAILLM(client, "")` — the failure surfaces only at LLM call time.

Recommended fix: assert `model_name` is non-empty after the partition.

### [LOW] `_message_to_jsonable` does not deep-copy tool_call `args`

`pipeline_bridge.py:207`: `"args": dict(tc.args)` creates a shallow copy. Mutations to the original `tc.args` (e.g. upstream's literal_eval coercion at `tool_execution.py:99-101`) would not affect the emitted observable IF the emit happens before the coercion. Today `_MessageStreamHook` fires AFTER `ToolsExecutor` (inner pos 2 is after ToolsExecutor at inner pos 1), so the literal_eval coercion has already happened and the emitted args reflect the coerced values. That's probably the desired polarity (the optimizer sees what the runtime saw), but worth documenting.

Recommended fix: docstring note on `_MessageStreamHook` clarifying that args reflect the post-coercion state of the message list at the moment of emission.

### [LOW] `_CatalogEditHook` does not propagate `extra_args` mutations

`pipeline_bridge.py:140`: the hook returns `extra_args` verbatim. If a future pipeline element passes `extra_args["catalog_mutated"] = True` as a signal between elements, the hook would drop the signal. Today upstream only uses `extra_args["plan"]` in `planner.py`, not in any of the elements the bridge composes, so this is purely speculative. Worth documenting.

Recommended fix: none for now; if extension is needed, the hook could read/write a key like `extra_args["catalog_edit_count"]`.

### [INFO] `pipeline.elements` is exposed for assertion via test indexing

The tests at `test_pipeline_bridge.py:130-137` access `pipeline.elements` as a list. `AgentPipeline.__init__` accepts `Iterable[BasePipelineElement]` and stores as-is. Our caller passes a list, so iteration works, but if a future refactor inside upstream changes `elements` to a one-shot iterator (e.g. a generator), the tests would silently break (first `list(pipeline.elements)` exhausts; the actual `pipeline.query` then iterates the now-empty iterator). Not actionable today; just an observation.

### [INFO] Hook instances are not reused across runs

`build_pipeline` is called per-run from `target.py:285-295`, so each run gets fresh `_MessageStreamHook` (cursor=0) and `_CatalogEditHook`. No cross-run state leak. Good.

### [INFO] The hook's exception in `_await_event` propagates to abort the run

If `send_event` raises (e.g. optimizer's `on_event` raises and middleware rejects), `future.result()` re-raises in the worker thread. The exception propagates out of the hook's `query`, out of `pipeline.query`, and is caught by `target.py:309-313` which logs and breaks the 3-retry loop. This is the controller's documented contract (envelope reject re-raises). Worth a one-liner in the hook's docstring.

## Strengths

- Pipeline shape is faithful to the no_defense baseline modulo the deliberate hooks; the docstring at lines 307-322 documents the layout clearly.
- Per-run hook construction (no cross-run state) is correct and aligns with the controller's per-task fresh-target lifecycle.
- The single-`_MessageStreamHook`-instance design is elegant: one cursor walks the message list as the pipeline progresses, naturally producing monotonically-indexed observables without needing inter-element coordination.
- The pinning test `test_openai_llm_reads_runtime_functions_per_call` in `test_toolsexecutor_per_turn.py` is well-conceived; it source-inspects upstream rather than calling OpenAI for real, and would catch a regression in upstream caching behaviour at our next version bump.
- `test_toolsexecutor_per_turn.py` correctly proves the mid-loop catalog edit visibility contract end-to-end with the real upstream `ToolsExecutor`.
- Hook unit tests cover the four-event firing order, the register success path, the bad-JSON swallow, and the duplicate-register `ValueError` swallow — solid coverage of the documented behaviour.

## Open questions

- Should `_build_llm` accept a pre-built `BasePipelineElement` (e.g. `pipeline_model: str | BasePipelineElement`) to let callers extend to providers the bridge doesn't natively support? Today the only escape is to fork the bridge.
- The brief documents `no_defense` only — is there a roadmap for spotlighting, repeat_user_prompt, tool_filter, or transformers_pi_detector? If so, the cursor and hook splice points may need to be reconsidered.
- Should `_try_apply` widen its catch to `Exception` (current swallow contract loose) or narrow further (only `ValueError` per current behaviour, with a tightened pre-validation upstream)? The class docstring promises the former; the code does the latter.
- Should the four catalog slots be ordered differently to allow self-referential operations (e.g. unregister before replace so an attacker can clear the namespace before placing a shadow)? Today's fixed order prevents the same-turn pattern "unregister `tool_X`, then register a new `tool_X`" — the unregister happens AFTER the register, so the final state is the unregister wins. Worth confirming this is intentional.
- Should the optimizer be able to observe its OWN catalog mutations via a synthetic `ObservableEvent` after each hook fires? Today the only way to confirm a mutation landed is to inspect the next-turn `tool_catalog_listing` snapshot via the static observable. A live confirmation channel might tighten optimizer planning.
