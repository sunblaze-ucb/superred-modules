# Test Coverage Gap Analysis: AgentDojo Port

Scope: target tests at `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/tests/` and claim tests at `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-claim/security_claims/agentdojo/tests/`, analysed alongside their source. No tests were executed; this is a read-only review against source.

## Severity legend

- **HIGH**: a production code path is exercised in the wild but not tested at all; a regression there would silently flip semantics observed by the optimizer or the security claim.
- **MEDIUM**: a path is partially tested (happy path only, or only one of several edge cases); a mutation/refactor could survive every test.
- **LOW**: minor or theoretical; documents a missing assertion that would tighten the contract but is unlikely to land regressions.

---

## A. `runtime_wrapper.py`: under-covered edge cases

### A.1 [HIGH] `_serialize_for_event` never exercised on `datetime`, `StrEnum`, or `None`

`_json_fallback` at `runtime_wrapper.py:97-104` has three branches: `BaseModel.model_dump`, `isoformat()` for datetime-like objects, `.value` for `StrEnum`-style objects, and a `repr()` fallback. The test fixture `test_runtime_wrapper.py` reaches `_serialize_for_event` only via the canonical-read path, where the live AgentDojo tools happen to return ints (balance), floats, strings, lists of pydantic models, and pydantic-model returns. Datetime returns flow through `workspace__get_current_day` (the test at `test_tool_registry.py:124-126` shows it returns `current_day.isoformat()`, so the `isoformat()` branch is reached upstream of `_serialize_for_event`) but NOT through the wrapper path; there is no test that confirms `_json_fallback` itself routes a datetime correctly when the tool returns a raw `datetime` (e.g. via an attacker `fake_return=datetime.now()`).

Likewise the StrEnum branch (e.g. `EmailStatus`, `EventStatus`) is never exercised. A test like:

```python
from datetime import datetime
out = _serialize_for_event(datetime(2024, 1, 1))
assert "2024-01-01" in out
```

would survive a mutmut mutation that swaps `isoformat()` for `repr()`.

**`None` returns are not covered at all.** `json.dumps(None)` returns the literal `"null"` (4 chars). The agent then sees the string `"null"` as the event answer; this conflates "no value" with "null literal". No test asserts this behaviour either way.

### A.2 [HIGH] `raise_on_error=True` path is not tested

`run_function` accepts `raise_on_error: bool = False` (`runtime_wrapper.py:206-216`) and forwards it to `super().run_function()`. Every test in `test_runtime_wrapper.py` calls `run_function` with the default. Upstream `FunctionsRuntime._execute_nested_calls` calls `self.run_function(env, name, kwargs, raise_on_error=True)` (see `agentdojo/functions_runtime.py:241-243`) so a real agent pipeline absolutely hits this path on any nested call. The wrapper's behaviour when the canonical body raises **and** `raise_on_error=True` is undefined by tests: does the wrapper still fire the post-call event? Does it still emit the response observable? The current code never reaches `_run_canonical`'s event-firing in that case because `super().run_function(..., raise_on_error=True)` raises before returning; but the trace append at line 213-216 already happened. There is no test pinning this asymmetry.

### A.3 [HIGH] Lists-of-pydantic and recursive pydantic returns not tested

`_serialize_for_event` short-circuits at `isinstance(value, BaseModel)` for a single model, but a tool like `workspace__search_emails` returns `list[Email]` (each `Email` is a `BaseModel`). The list falls through to `json.dumps(value, default=_json_fallback, ...)`; the fallback then dispatches each element via `model_dump()`. There is no test asserting the round-trip is faithful (no truncation, no order change, no key omission). Banking transactions are likewise `list[Transaction]`.

Deeply nested pydantic (e.g. `Email.attachments: list[Attachment]` where each `Attachment` is itself a BaseModel) is never exercised. A mutation that swaps `model_dump()` for `model_dump(mode='json')` (or vice versa) could survive every existing test because the assertions only check `float(answer)`.

### A.4 [HIGH] Mid-stream `ToolNotFoundError` not exercised within a multi-call sequence

`test_unknown_tool_falls_through_to_upstream_error` (line 274-283) is a single-call test. But the wrapper's contract is that the trace records the attempt even when the call errors. There is no test that interleaves: canonical call, unknown call, canonical call, and asserts the trace contains all three (in order) with the middle entry's error correctly attributed. Mutmut could survive a mutation that moves `self._trace.append` below the `entry is None` check.

### A.5 [MEDIUM] `agent_seen_value` shape on attacker-path with non-string injection

Test `test_attacker_registered_injection_overrides_fake` (line 215-229) asserts `result == "OVERRIDDEN"` only. The injection value is always a string in tests; the code at line 269 (`agent_seen_value = response.value`) is typed `FunctionReturnType` but the response carries a raw `value: str` per ASSUMPTIONS A.3. There is no test that confirms what happens when the LLM's tool-result formatter receives this string vs. a structured pydantic return. The existing review at `REVIEW_FINDINGS/02_runtime_wrapper.md` already flags this; no test was added.

### A.6 [LOW] `refresh_functions` thread-safety pin missing

Test `test_refresh_functions_picks_up_catalog_edits` (line 286-297) is single-threaded. The wrapper's `refresh_functions` (`runtime_wrapper.py:184-191`) reassigns `self.functions`. No test confirms that a refresh happening between two `run_function` calls on the same worker thread doesn't lose a concurrent edit. (See `02_runtime_wrapper.md` MEDIUM finding for the rationale.)

---

## B. `pipeline_bridge.py`: exception paths

### B.1 [HIGH] LLM rate limit / network error never simulated

`build_pipeline` constructs OpenAI / Anthropic clients and embeds them into upstream pipeline elements. The Target's `run` (`target.py:301-318`) wraps `pipeline.query` in a 3-retry loop with a bare `except Exception` that LOGS and breaks. There is NO test that simulates an LLM error mid-stream:

- 429 rate-limit on first call: does the retry actually retry, or does the bare-except swallow on attempt 1?
- 5xx on attempt 2: does attempt 3 fire?
- Network timeout on every attempt: does the wrapper leave `self._messages` and `self._last_response` in their pre-run defaults?

`test_concurrent_isolation.py` uses an `_EchoingLLM` that never errors; `test_integration.py` uses `_FakeLLM` that never errors. The 3-retry loop has no test.

### B.2 [HIGH] Malformed Anthropic thinking suffix has only one negative test

`test_build_llm_anthropic_thinking_suffix_invalid_int` (line 102-107) tests `-thinking-banana`. But the partition `model_name.partition("-thinking-")` (`pipeline_bridge.py:265`) has untested edge cases:

- `anthropic/claude-3-5-sonnet-20241022-thinking-` (empty budget after the suffix): `int("")` raises `ValueError`; the test catches `"thinking"` in the message but the docstring contract says it must be an integer. Confirm the empty case is treated as malformed.
- `anthropic/claude-3-5-sonnet-thinking-0`: parses to `budget=0`. Does AnthropicLLM accept zero budget? No test covers this.
- `anthropic/claude-3-5-sonnet-thinking-1024-thinking-2048`: the first `partition` splits on the first occurrence, so `budget = "2048"` (correct) and `base_model = "claude-3-5-sonnet"`. This double-suffix likely indicates a typo on the caller's side; should it be rejected? Not tested.

### B.3 [HIGH] ToolNotFoundError mid-pipeline never tested

`_CatalogEditHook.query` (`pipeline_bridge.py:117-140`) does NOT call `runtime.run_function` directly, but the agent might invoke a tool name that was just unregistered. There is no test that:

1. Calls `apply_unregister` on a canonical tool via the catalog hook.
2. Confirms the wrapped runtime's `refresh_functions` actually drops it.
3. Asserts the next `ToolsExecutor` call returns "Invalid tool" rather than crashing.

The integration test (`test_toolsexecutor_per_turn.py:test_toolsexecutor_sees_attacker_tool_after_mid_loop_catalog_edit`) covers register, not unregister.

### B.4 [HIGH] Hook fires for unknown controllable response shape — no test

`_CatalogEditHook._try_apply` (`pipeline_bridge.py:100-115`) tolerates:

- `json.JSONDecodeError` (tested at `test_hook_swallows_invalid_json_payload`, line 235-256)
- non-dict payload (line 106-110): NOT tested — sending `"\"a string\""` parses to a `str` and should hit the "must be a dict" warning. No test exercises this branch.
- `ValueError` from `apply_*` (tested at `test_hook_swallows_value_error_from_apply`, line 259-286)

But what about an unknown response type entirely? If the optimizer ever returns an `EventResponse` subclass that is neither `ControllableInjection` nor `ControllableNoInjection`, the `isinstance(response, ControllableInjection)` check at line 135 is False and nothing happens — but no test pins this. A mutation that flips the isinstance check would survive.

### B.5 [MEDIUM] `_MessageStreamHook` does not handle a message with both `tool_calls` and `tool_call`

`pipeline_bridge.py:203-221` reads both `tool_calls` (list) and `tool_call` (single) for an assistant message. The test at line 336-358 covers only `tool_calls=[fc]` (a list). No test exercises a message with BOTH set (this can happen in transitional message formats), or a message where `tool_calls` is a list that includes non-FunctionCall entries (line 209: `if hasattr(tc, "function") else tc`).

### B.6 [LOW] Pipeline shape regression: `_MessageStreamHook` reused for outer + inner

Tests `test_build_pipeline_returns_agentpipeline` and `test_build_pipeline_splices_hook_into_loop` verify the outer/inner element positions. Neither asserts that the same `_MessageStreamHook` *instance* is shared across all four splice points so its `_next_idx` cursor advances monotonically. A refactor that constructs separate instances would silently re-emit duplicates; no test catches this.

---

## C. Trajectory closure semantics

### C.1 [HIGH] No test for target shutdown mid-emit

`AgentDojoTarget.run` (`target.py:243-341`) emits five phases of events. If the optimizer's `on_event` raises in the middle of, say, Phase 4's mid-loop `_CatalogEditHook`, what does the channel see? What does `_await_event` return in that case? `runtime_wrapper.py:197-200` calls `future.result()` with no exception handling. The existing review at `02_runtime_wrapper.md` flags this; no test exists.

### C.2 [HIGH] No test for optimizer disconnect mid-event

Symmetric to C.1: if the optimizer's `on_event` is awaited but the underlying channel is closed (e.g. controller-level abort), the wrapper's `_await_event` future could resolve to a sentinel or hang. No test simulates this.

### C.3 [MEDIUM] `cleanup` after partial run never tested

`test_cleanup_resets_per_run_state` (line 127-138) tests cleanup AFTER manually setting per-run state directly on the target. There is no test that:

1. Runs `target.run` and intentionally raises mid-Phase 4 (e.g. LLM error).
2. Verifies `cleanup` still zeroes `self._wrapped_runtime`, `self._catalog`, etc.

If a partial run leaves `self._catalog` populated and `cleanup` skips it (a hypothetical mutation), the next task gets a stale catalog. Not pinned by tests.

---

## D. `tool_catalog.py`: edge cases

### D.1 [HIGH] Replace, then unregister, then re-register the same name — not tested

The sequence `apply_replace("banking__get_balance") -> apply_unregister("banking__get_balance") -> apply_register("banking__get_balance")` should yield a `registered` kind, not `replaced`. No test in `test_tool_catalog.py` covers this transition. The closest is `test_reset_restores_seed` (line 121-130) which resets the entire catalog after multiple mutations; it does not exercise the replace-unregister-register cycle on a single name.

### D.2 [HIGH] `apply_rewrite_doc` with empty description not tested

`apply_rewrite_doc` calls `_require_str(payload, "description")` (line 326). `_require_str` rejects empty strings (line 353-354: `if not isinstance(value, str) or not value`). So an empty description is rejected — but no test pins this. A mutation that drops the `or not value` guard would let the empty-description rewrite through, and the agent would see a tool with no doc.

### D.3 [HIGH] Very large `fake_return` (>1MB) not tested

The catalog stores `fake_return: Any` (line 71). When the wrapper short-circuits an attacker call (`runtime_wrapper.py:299: fake_answer = _serialize_for_event(entry.fake_return)`), the serialised answer flows through the event channel and into the trajectory. There is no test for a `fake_return` that is, say, 1 MB of bytes; the JSON serialiser would balloon to 2-4 MB and the trajectory entry would consume that much memory. Whether the trajectory has any size cap is untested (and likely there is none).

### D.4 [MEDIUM] Snapshot ordering invariance not tested

`ToolCatalog.snapshot()` (line 208-227) iterates `self._entries.values()` — dict insertion order. If a future change rebuilds the dict in a different order (e.g. via a `sorted()` call), no test would catch the change. The snapshot is consumed by the `TOOL_CATALOG_LISTING_OBS` observable, which the optimizer may rely on for deterministic indexing.

### D.5 [MEDIUM] Parameters-schema rebuild on `apply_replace` not tested for type fidelity

`apply_replace` at line 286-294 builds a new `Function` preserving `existing.function.parameters` AND `existing.function.return_type`. The test `test_replace_marks_canonical_as_replaced` only checks `entry.kind == "replaced"` and `entry.fake_return == 9999.0`. It does NOT verify that the parameters class on the new function is *the same class* as on the original (could be a deep copy or a fresh model). A mutation that calls `_build_parameters_class(name, None)` instead of preserving the original would survive every existing test.

### D.6 [LOW] `_PermissiveSchema` ConfigDict mutation not pinned

`_PermissiveSchema` at line 80-88 uses `ConfigDict(extra="allow")`. A mutation to `extra="forbid"` would break attacker-tool calls (unknown kwargs would be rejected). The test at `test_build_placeholder_function_permissive_schema_when_no_schema` (line 145-149) plants `{"x": 1, "y": "two"}` and asserts the result; this would catch the mutation but only because the test passes unknown kwargs. A stricter pin would explicitly test that `extra="allow"` is on `_PermissiveSchema.model_config`.

---

## E. `env.py` `sync_initial_fields`: depth-limited

### E.1 [HIGH] Only inbox/calendar/cloud_drive synced; not travel.inbox.attachments etc.

`sync_initial_fields` at `env.py:88-94` syncs:

- `workspace.inbox.initial_emails`
- `workspace.calendar.initial_events`
- `workspace.cloud_drive.initial_files`
- `travel.inbox.initial_emails`
- `travel.calendar.initial_events`

It does NOT sync:

- `slack.slack.user_inbox` / `channel_inbox` (these are also dict-rebuilt? Need to verify — but no test confirms either way)
- `banking.bank_account.transactions` / `scheduled_transactions` (these are `list[Transaction]`, no derived dict, but no test confirms list-based stores survive round-trip)
- `banking.filesystem.files` (no test of round-trip of file mutations)
- `travel.reservation` (a single nested model, no test of round-trip after agent-driven cancellation)
- `travel.hotels.hotel_list`, `travel.restaurants`, `travel.flights.flight_list`, `travel.car_rental` (lists of 3p data — if the agent ever writes to these, do mutations survive?)

The test `test_env_sync.py` covers only the three known cases (workspace inbox, workspace calendar, workspace cloud_drive). A new agent mutation on any non-covered store could silently be lost on round-trip and the security predicate would see the seed.

### E.2 [MEDIUM] No test for nested `Email.attachments` mutation round-trip

`Email.attachments` is a list of `Attachment` pydantic models. If an agent adds an attachment to an existing email (e.g. via a write tool that mutates an inbox entry in place), does the round-trip survive? The `sync_initial_fields` only re-lists the top-level emails by id; it does not deep-copy or rebuild `Email.attachments`. No test exists.

### E.3 [MEDIUM] `sync_initial_fields` idempotence not tested under deletion sequence

`test_sync_idempotent` (line 92-99) confirms two consecutive syncs produce the same JSON. But it does NOT test the sequence: sync -> delete -> sync -> delete -> sync, which is the realistic multi-turn pattern. A mutation that swaps `list(...values())` for `[...]` (taking a stale snapshot) could survive idempotence but fail on the delete sequence.

---

## F. Concurrency: only 4 and 8 tested

### F.1 [HIGH] No test at 16, 32, or 64 concurrent targets

`test_concurrent_isolation.py` runs at concurrency=4 and concurrency=8. AgentDojo benchmarking commonly runs at 16-32 parallel runs against a litellm proxy. The test file's docstring says "stress" but 8 is barely stress. Several risks remain unverified:

- Per-task `seed_yaml_override` propagation: the `_seed_overrides` dict on each target is per-instance, but a regression that accidentally shares it via a class attribute would only manifest at higher concurrency.
- AgentDojo's import-time decorator-driven task registration (see the `env.py` workaround at lines 22-32) is global state. The test doesn't probe whether 16+ targets racing to first-touch trigger a re-entrancy bug in `load_suites`.
- The `asyncio.run_coroutine_threadsafe` bridge in `runtime_wrapper.py:199` schedules futures on a single loop. Under 32 parallel targets each firing per-tool events, the loop's queue depth grows; no test confirms the bridge survives O(1000) outstanding futures.
- The `_tool_response_counter` (`runtime_wrapper.py:171`) is per-wrapper, not global, so this is fine in principle. But the test never confirms two parallel runs produce non-interleaved counters.

### F.2 [MEDIUM] No test for memory pressure (large pre/post env snapshots)

The pre/post composite env snapshots (`_dump_env_or_empty` in `target.py:398-414`) serialise the full composite env via `model_dump_json()`. The composite has 74 tools' worth of state — banking (~10 KB), workspace (~50 KB if inboxes are full), slack (~30 KB), travel (~40 KB), totalling ~100-200 KB per snapshot. At concurrency=32 with two snapshots per task (pre, post) and 4+ runs per task, the controller may hold 8 MB+ in memory. No test confirms this stays bounded or that snapshots are garbage-collected after each run.

### F.3 [LOW] `test_no_module_level_mutable_state_in_target_package` pin is incomplete

The audit pin at lines 310-353 only enumerates a hardcoded `KNOWN_IMMUTABLE` set. A new module-level constant (e.g. a new controllable enum) would pass the test silently as long as it isn't named there. A real "no module-level mutable" check would walk every name in `agentdojo_target.*.__all__`, use `inspect.getmembers`, and assert that no non-frozen dataclass / non-frozen-set / non-tuple is exposed.

---

## G. Layer-1/2/3 pair generation: 629-pair count not asserted

### G.1 [HIGH] No assertion that Layer-1 default + full cross-product totals 629

The upstream AgentDojo paper specifies the full cross-product is 629 (suite, user_task, injection_task) cases. The CANONICAL_PAIRS scope is 27. The `agentdojo_layer1_claim(pairs=...)` API accepts any subset. No test in `test_layer1_factory.py` (or anywhere) asserts that the full enumeration (e.g. via `itertools.product` over all four suites' user_tasks x injection_tasks) totals **exactly 629**.

The faithfulness test at `tests/faithfulness/test_ground_truth_replay.py` parametrizes over user_tasks (97) + injection_tasks (27) = 124 cases (line 35), but this is the ground-truth replay (no LLM); it doesn't enumerate the cross-product of (user_task × injection_task).

A regression where upstream silently adds a new user_task in `workspace` (making the cross-product 629 + N) would not be caught.

**Recommended addition**:

```python
def test_full_cross_product_totals_629() -> None:
    """Pin AgentDojo v1's claim: 629 (user_task, injection_task) pairs."""
    from itertools import product
    from agentdojo.task_suite.load_suites import get_suite
    total = 0
    for suite_name in ["banking", "workspace", "slack", "travel"]:
        suite = get_suite("v1", suite_name)
        total += len(suite.user_tasks) * len(suite.injection_tasks)
    assert total == 629
```

### G.2 [MEDIUM] Layer-2 catalogue count pinned but Layer-3 sum not asserted

`test_layer2.py:test_catalogue_size_matches_expected` (line 72) asserts Layer 2 has 19 specs. `test_top_factory_default_is_27_pairs` (line 27) asserts Layer 1 has 27. `test_combined_claim_iterates_both_layers` (line 262) asserts Layer 3 = Layer 1 + Layer 2 lengths — but Layer 3's expected total (27 + 19 = 46) is computed at test time, not pinned as a literal constant. A regression that drops a Layer-1 task and adds a Layer-2 task would still pass the combined test.

### G.3 [MEDIUM] No test that the 27 canonical pairs satisfy upstream's "every injection task covered once" claim

`layer1_pairs.py:87-92` claims "Each injection task in v1 is covered exactly once" but no test asserts this. A regression that drops `("banking", "user_task_1", "injection_task_3")` would silently reduce coverage. Pin:

```python
def test_every_injection_task_covered_exactly_once() -> None:
    from collections import Counter
    counter = Counter((suite, it) for (suite, _ut, it) in CANONICAL_PAIRS)
    assert all(c == 1 for c in counter.values())
    # Also assert total injection-task count matches upstream's claim
    from agentdojo.task_suite.load_suites import get_suite
    total_it = sum(
        len(get_suite("v1", s).injection_tasks)
        for s in ["banking", "workspace", "slack", "travel"]
    )
    assert len(counter) == total_it  # one canonical pair per injection task
```

### G.4 [LOW] `pairs` kwarg precedence over `suites`/`categories` not edge-tested

`test_pairs_kwarg_takes_precedence_and_supports_extension` (line 61-71) passes only `pairs=`. There is no test that confirms `pairs=[...]` AND `suites=[...]` together silently ignores `suites`. A regression that flipped the precedence would survive.

---

## H. Persistence (Controller.results_dir): no AgentDojo-specific tests

### H.1 [HIGH] Zero tests verify the AgentDojo target's persistence path

The superred `Controller` writes results to `{results_dir}/{scope}__{model}.json` plus a per-task subfolder when `results_dir` is configured. Neither `test_concurrent_isolation.py` nor `test_integration.py` (the closest end-to-end tests in this port) ever sets `results_dir`. There is NO test that verifies:

- The composite env snapshot (potentially 100+ KB) is serialisable to JSON without raising.
- The trajectory (containing potentially hundreds of `ObservableEvent`s with `agent_trace_tool_response_NNNN` content) is JSON-serialisable.
- The persistence path correctly excludes secret-bearing fields (e.g. `api_key`) — the `Controller`'s `LLMConfig.api_key` exclusion is framework-level, but the target's `api_key` stored in `AgentDojoTarget._api_key` is NOT covered by that. Is the target's instance-level api_key leaked into a serialised TaskResult? No test confirms either way.
- Atomic per-file writes survive a crash mid-run: a partial JSON file would corrupt downstream analysis.

### H.2 [MEDIUM] No test for JSON-roundtrip of a `RunEndEvent` carrying `evaluation`

When `Controller.include_feedback=True`, the `RunEndEvent` carries `evaluation=filtered_eval`. The evaluation contains pydantic-emitted `Score` objects and a `EvaluationResult.rationale` string. For Layer-1 tasks the rationale includes the suite/user_task/injection_task ids; for Layer-2 the goal_id. No persistence test confirms this fits the controller's serialisation expectations.

---

## I. Mutation testing assumptions

### I.1 [HIGH] No mutmut config in either pyproject.toml

`pyproject.toml` for both packages contains `[tool.coverage.report] fail_under = 90` but no `[tool.mutmut]` section. The README at top of session (in claudeMd memory) mentions `mutmut run --no-progress` is run for the superred framework, but the modules have NO mutmut configuration. The 62.8% mutation-score claim is NOT pinned by either package; running mutmut on these modules today would use defaults and may not produce the claimed number.

### I.2 [HIGH] Modules likely with poor mutation coverage based on test thinness

Based on this read-through:

- **`security_predicates.py`** (555 lines, ~20 helpers) has positive+negative tests for ~13 of the helpers in `test_layer2_predicates.py` + `test_layer2_predicates_starter_positives.py`. The 7 NOT explicitly named in the test files (e.g. `recurring_transaction_redirected_to`, `password_changed`, `external_user_invited`, `legitimate_user_removed`, `reservation_above_price`, `existing_reservation_cancelled`, `email_to_external_contains_pii`) are mostly tested in `test_layer2_predicates.py`, but the cross-suite helpers (`workspace_inbox_content_in_external_email`, `pii_appears_on_attacker_web`, `hotel_reservation_in_wrong_city`, `banking_data_appears_in_slack`) are tested only with positive+seed-negative, not with edge cases (empty domain, missing field). Mutants in the inner string normalisation (`.lower()`, `.strip()`, `.lstrip("@")`) would likely survive. **Estimated mutation kill rate: 50-65%.**

- **`runtime_wrapper.py`**: covered well for happy paths but has untested branches in `_serialize_for_event` (`isinstance` checks, the `repr` fallback, the `_json_fallback` dispatch on datetime / StrEnum / None). Multiple `02_runtime_wrapper.md` HIGH findings list unfixed gaps. **Estimated mutation kill rate: 60-70%.**

- **`pipeline_bridge.py`**: 25% of the LLM-dispatch logic (Anthropic thinking) has one negative test only; the retry loop in `target.py` is untested entirely; the `_MessageStreamHook` handles `tool_calls` AND `tool_call` (single vs. plural) and the single-form branch is barely tested. **Estimated mutation kill rate: 55-65%.**

- **`tool_catalog.py`**: mutations on `_require_str`'s empty-string guard, snapshot ordering, and the `_PermissiveSchema.model_config` would likely survive. **Estimated mutation kill rate: 65-75%.**

- **`env.py`** `sync_initial_fields`: only 3 of the potentially 8+ derived-dict stores are synced and tested. **Estimated mutation kill rate: 60-70%** (any mutation flipping which store is synced would survive).

- **`target.py`**: cleanup, the 3-retry loop, the `_messages_to_jsonable` helper's nested `FunctionCall` handling, the `_model_output_from_messages` content-block flattening — none are tested in isolation. **Estimated mutation kill rate: 60-65%.**

The README target of ~62.8% is plausibly the *current* number rather than a goal; raising it to ~80% requires the additions listed in this report.

### I.3 [MEDIUM] Faithfulness tests are gated, so they don't contribute to baseline mutation kill

`test_faithfulness_smoke.py` is `pytest.mark.skipif(not _has_creds())` and the implementation body is `pytest.skip()` (line 59-62) — the test is a placeholder. The full 629-case cross-product runner is referenced but does not exist in v1. Mutmut would skip these too, so they contribute nothing to mutation kill score.

---

## J. Other gaps worth noting

### J.1 [MEDIUM] `_messages_to_jsonable` (target.py:417-437) untested in isolation

This helper converts `ChatMessage` TypedDicts (potentially containing pydantic `FunctionCall` instances under `tool_calls` and `tool_call`) into plain dicts. It's only exercised indirectly via the concurrent-isolation test. A mutation that swaps `_function_call_to_dict(c) if isinstance(c, FunctionCall) else c` for unconditional `_function_call_to_dict(c)` would crash on a non-FunctionCall entry; no isolated test would catch the crash.

### J.2 [MEDIUM] `_model_output_from_messages` content-block handling

The branches at lines 451-465 cover: assistant role check, `content is None`, `content is str`, `content is list[block]`. No test covers `content=[]` (empty list, joined returns empty string, `or None` returns `None`); no test covers `content=[{"type":"image", ...}]` (non-text block, `texts` is empty list, returns `None`). The `joined or None` line (463) is a mutation hotspot.

### J.3 [LOW] No test for `pipeline_bridge._build_llm` with an `api_base` that is empty string

`api_base=""` (vs. `None`) would be passed through verbatim to OpenAI's client constructor. No test pins what happens — likely OpenAI's SDK either accepts it (no-op) or raises. Worth a one-line test.

### J.4 [LOW] No test for `merge_yaml_overlay` with deeply nested overlay (>3 levels)

`test_seed_loader.py` tests overlays at depth 2 (`{"bank_account": {"balance": 0.0}}`). A 4-level-deep overlay (e.g. into `workspace.inbox.emails[id]`) is not tested. The `_deep_merge` helper recurses on dicts only; list-element merging is explicitly NOT supported (per the docstring). Worth a test that asserts list-element merging IS a replacement, not a merge — a mutation that swaps the dict-isinstance check could survive.

---

## Recommended priority for additions

In rough order of expected impact:

1. **G.1 / G.3**: pin the 629 cross-product total and the "every injection task covered exactly once" claim.
2. **A.1 / A.3 / A.4**: serialisation edge cases (datetime/StrEnum/None, lists-of-pydantic, mid-stream ToolNotFoundError).
3. **F.1**: a concurrency stress test at 16-32 (mark it slow / opt-in if needed).
4. **B.1 / B.3**: LLM retry loop simulation + mid-pipeline unregister.
5. **E.1**: sync_initial_fields coverage for the 5+ unsynced stores (especially slack inboxes, banking transactions list, travel reservation).
6. **H.1**: a persistence smoke test that runs one task with `results_dir` set and verifies the on-disk JSON is well-formed and excludes secrets.
7. **I.1**: add `[tool.mutmut]` config to both pyproject.toml so the 62.8% baseline is reproducible.
8. **D.1 / D.2**: tool-catalog state-machine transitions (replace -> unregister -> register; rewrite_doc with empty description).
