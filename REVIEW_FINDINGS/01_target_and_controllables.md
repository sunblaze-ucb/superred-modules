# Target + Controllables Review

## Summary

`AgentDojoTarget` correctly stitches the upstream `AgentPipeline`, the wrapped runtime, the catalog hook, and the composite environment into superred's event-driven Target contract, and the Controllable enumeration (53 entries: 1 system, 1 user, 4 catalog, 47 reads) has clean self-validating construction. The main concerns are faithfulness divergences in `run()`'s retry loop and model-output extraction (which silently differ from upstream `task_suite.run_task_with_pipeline`), a documented-but-bug-prone seed-overlay path through pydantic validators, dynamic Controllables fired at runtime that were never enumerated in `get_controllables()`, and a sizeable test-coverage gap on `Target.run()` itself.

## Findings

### [HIGH] `run()` retry loop swallows non-Abort exceptions that upstream re-raises

In `target.py:301-318` the retry loop catches `except Exception` and breaks, logging a `pipeline.query` failure to the module logger. Upstream `task_suite.run_task_with_pipeline` (`agentdojo/task_suite/task_suite.py:383-394`) catches only `AbortAgentError`; **every other exception propagates** so the framework can attribute the failure (e.g. `openai.BadRequestError` for context-length, `cohere.ApiError` for upstream-down) and apply the "skip with `security=True`" fallback in `benchmark.py:120-148`. The port's blanket `Exception` silently turns those into "no last_response, no extra trace events" with no error visible on the trajectory or `TaskResult.error`. Effect: faithfulness comparisons against upstream verdicts will diverge whenever any provider error hits, and Layer-1 predicates that key off "agent never produced output" cannot distinguish a real model refusal from a transient API failure.

Recommended fix: narrow to `except AbortAgentError as e` and, in that branch, set `self._env = e.task_environment` and `self._messages = e.messages` exactly as upstream does; let every other exception propagate to the controller's `_run_task` error handler (it already records the traceback on `TaskResult.error`).

### [HIGH] `AbortAgentError` payload (messages + env) is discarded

Tied to the previous finding but worth its own entry: even if you keep the broad-`Exception` catch, upstream's `AbortAgentError` carries `e.messages` (the abort message appended as a final assistant turn) and `e.task_environment` (the env state at abort time). The current `except Exception: ... break` (`target.py:309-313`) drops both. After an abort, `self._env` and `self._messages` retain whatever attempt 0 produced (or stay empty if attempt 0 itself aborted), so downstream predicates miss the abort message and see the wrong final env. This matters for AgentDojo's prompt-injection-detector defense scenario (covered by the `pi_detector` element in upstream) where the abort is *signal*, not noise.

Recommended fix: handle `AbortAgentError` explicitly: `except AbortAgentError as e: self._env = e.task_environment; self._messages = e.messages; model_output = _model_output_from_messages(e.messages)`. Match upstream's exact recovery sequence.

### [HIGH] `_model_output_from_messages` diverges from upstream extraction semantics

`target.py:440-465` returns `str | None` after concatenating text-typed content blocks. Three behavioral divergences from upstream `model_output_from_messages` (`agentdojo/task_suite/task_suite.py:70-75`):

1. Upstream raises `ValueError("Last message was not an assistant message")` when the final role is not assistant; the port silently returns `None`, which makes the retry loop spin three times on a tool-result-final message (e.g. when the LLM gives up mid-tool-call) where upstream would have crashed loudly.
2. Upstream returns `list[MessageContentBlock]` (the raw blocks); the port joins only the `type=="text"` blocks into a string. AgentDojo predicates that look for "any non-empty model output" treat `[]` as truthy and break the retry loop; the port returns `None` for an empty list AND for a list of only thinking-blocks, so it keeps retrying.
3. Upstream does not handle `content` being a bare string (it would crash on `last["content"]` access); the port's `if isinstance(content, str): return content` branch is reachable in non-spec'd shapes only and effectively dead code for the canonical pipeline.

Effect: `model_output` is `None` more often in the port than upstream, so the port retries more often and pays a triple cost for completions that upstream would have accepted; conversely, when upstream would have raised, the port retries silently, masking pipeline bugs.

Recommended fix: rename and align to upstream `model_output_from_messages`: return `last["content"]` (a `list[MessageContentBlock] | None`) and raise on non-assistant final messages. The downstream consumer (`self._last_response`) can call `agentdojo.types.get_text_content_as_str` for the agent-visible string.

### [MEDIUM] Dynamic per-call attacker Controllables are not declared in `get_controllables()`

`runtime_wrapper.py:107-127` builds a fresh `Controllable(name=f"tool_call:{entry.name}", ...)` for every attacker-managed call. These never appear in `target.get_controllables()` (which only returns the 53 enumerated entries from `controllables.py:280`). Optimizers are initialized with the filtered controllable list (`controller.py:500-502`) and may reasonably assume that list is exhaustive. When the agent later calls a `registered`/`replaced` tool, the optimizer receives a `ControllablePostCallEvent` whose `controllable` is an instance it has never seen and whose `security_domain` (`TOOL_CATALOGUE_ADDABLE_TAG` or `TOOL_CATALOGUE_TAG`) IS in the system tree but might or might not be in the optimizer's scope. Two consequences:

- The `security_domain_filter` middleware correctly gates by the embedded tag, so out-of-scope events still get `ControllableNoInjection`. That's fine.
- An optimizer that asserts `event.controllable in self.known_controllables` (a reasonable defense against malformed events) will crash. The contract that "controllables list is exhaustive" is broken without warning.

Recommended fix: either (a) document the dynamic-controllable behavior on `Target.get_controllables` and on the relevant Controllable singletons so optimizers know to expect them, or (b) construct the per-attacker-tool controllables eagerly at register/replace time, store them on the catalog entry, and expose them through `get_controllables()` (more invasive but more honest). Option (a) plus a docstring on `CONTROLLABLES` saying "additional Controllables of the form `tool_call:{name}` may appear at runtime when attacker-managed tools are invoked" is the lighter touch.

### [MEDIUM] Seed YAML overlay does not reach `initial_*` source lists

`seed_loader.merge_yaml_overlay` (`seed_loader.py:113-118`) merges the overlay into `sub.model_dump()` then calls `type(sub).model_validate(merged)`. The Inbox/Calendar/CloudDrive pydantic models declare `@model_validator(mode="after")` that REBUILDS `emails`/`events`/`files` from `initial_emails`/`initial_events`/`initial_files`. An overlay like `{"inbox": {"emails": {"new_id": {...}}}}` writes the merged `emails` field, then the validator overwrites it from the unchanged `initial_emails` list — **the overlay is silently dropped**. The same bug shape that `env.py:sync_initial_fields` exists to fix on the round-trip side.

Effect: Tasks attempting to replay AgentDojo's per-task `init_environment` mutations via the seed-overlay API will see their changes lost for the three derived-dict models (Inbox, Calendar, CloudDrive), but other overlays (bank_account.balance, etc.) work fine. Sneaky because the surface API succeeds without error.

Recommended fix: in `merge_yaml_overlay`, after computing `merged`, also propagate writes into the corresponding `initial_*` lists when the overlay targets `inbox.emails`/`calendar.events`/`cloud_drive.files`. Alternatively, document loudly that the only safe path is to write `initial_emails`/`initial_events`/`initial_files` directly in the overlay; the `_seed_yaml_override_spec` description (`config_specs.py:67-76`) currently says nothing about this trap.

### [MEDIUM] `cleanup()` does not drain `WrappedFunctionsRuntime._tool_response_counter` and other emit-side counters live on the discarded wrapper

`target.py:347-362` resets `_wrapped_runtime = None`, which is correct because a fresh wrapper is built each `run()`. But two implicit assumptions ride on this:

- The `_tool_response_counter` only matters within a single run because indices restart at 0000 each new wrapper; this is fine.
- The `_MessageStreamHook` instance built inside `build_pipeline` is also discarded along with the pipeline. Its `_next_idx` cursor lives only on that pipeline instance, so per-run isolation holds.

What's missing: there is no test that re-running the same `AgentDojoTarget` (via the controller's multi-run loop) gets fresh `agent_trace_tool_response_NNNN` and `agent_trace_message_NNNN` indices starting at 0000 on every run. The test in `test_runtime_wrapper.py:381` checks the monotonic property within ONE wrapper. A multi-run regression would be caught only by the four-target concurrency test, which spawns four targets rather than re-running one target three times.

Recommended fix: add a `test_target.py::test_run_indices_reset_per_run` that runs the target twice (with a stubbed `_build_llm`) and asserts the second run's first observable is `agent_trace_tool_response_0000`, not `0006` or similar. This is a 30-line test using the fake-LLM machinery already in `test_concurrent_isolation.py`.

### [MEDIUM] `_dump_env_or_empty` mutates the env passed in

`target.py:398-414` calls `sync_initial_fields(env)` which mutates `env` in place (`env.py:88-95`). For the post-env path this is harmless — by the time it's queried, the run is done. For the pre-env path it is also harmless because `pre_env` is a separate deep-copy. But the docstring on `_dump_env_or_empty` says nothing about the side effect, and `sync_initial_fields` returns the same env for chaining — easy for a future maintainer to assume it's a pure function.

Effect: low; the mutation is idempotent (re-syncing produces the same lists). But if someone refactors to share `env` between in-flight tools and the query path, the mutation could land on a still-active env and confuse the upstream pydantic validators.

Recommended fix: rename `sync_initial_fields` to make the in-place mutation obvious (`mutate_to_sync_initial_fields_in_place`), or have it deep-copy before mutating. The current chaining-convenience return is a footgun.

### [LOW] Unused `from pydantic import BaseModel` in target.py

`target.py:44` imports `BaseModel` but the symbol is never used in the module. Removed when running `ruff --select F401`.

Recommended fix: drop the import.

### [LOW] `_function_call_to_dict` drops empty `placeholder_args` differently from upstream

`target.py:388-395` outputs `"placeholder_args": dict(fc.placeholder_args) if fc.placeholder_args else None`. An empty mapping `{}` is falsy in Python, so the serialized form is `None`. AgentDojo's `FunctionCall.placeholder_args` is `Mapping | None` per `functions_runtime.py:50`; an explicit empty mapping has different semantics from `None` (ground-truth pipelines distinguish "no placeholder" from "placeholder with zero args"). Practical impact is essentially nil for v1 but is a subtle round-trip-faithfulness fault if a Layer-1 predicate ever serializes/deserializes the trace.

Recommended fix: `"placeholder_args": dict(fc.placeholder_args) if fc.placeholder_args is not None else None`.

### [LOW] Eager `FunctionCall` in trace loses tool_call.id

`runtime_wrapper.py:214-216` records the call as `FunctionCall(function=function, args=dict(kwargs))` with no `id`. Upstream's `functions_stack_trace_from_messages` (`agentdojo/task_suite/task_suite.py:60-67`) pulls the FunctionCall objects straight from the assistant messages, which preserves `id` (set by the LLM provider). For v1 predicates this doesn't matter (none use the id), but the ASSUMPTIONS doc claims "the wrapped runtime's trace mirrors AgentDojo's functions_stack_trace_from_messages" — that's not literally true on the `id` field.

Recommended fix: thread the tool_call's id through. The `ToolsExecutor.query` (upstream) iterates `messages[-1]["tool_calls"]` and calls `runtime.run_function(env, tool_call.function, tool_call.args)`. The id is on `tool_call.id` at that point but is lost crossing the API boundary. Two options: (a) plumb `tool_call_id` as a kwarg or via `extra_args` (more invasive), or (b) accept the divergence and update ASSUMPTIONS.md to call it out (consistent with the existing C.* drift entries).

### [LOW] `_attacker_call_ctrl` description is internally inconsistent for one of two paths

`runtime_wrapper.py:107-126` builds a Controllable whose description always says "Per-call event for the attacker-{entry.kind} tool `{entry.name}`". For a `replaced` entry this is fine. For a `registered` entry the wording is correct too. The issue is subtler: when the optimizer responds with `ControllableInjection` on this transient Controllable, the response references `controllable=event.controllable` (the transient instance). Two events for the same tool invocation across different turns yield two DIFFERENT Controllable instances with the same `name`. Optimizers comparing controllables by identity rather than name will treat them as distinct.

Recommended fix: cache `_attacker_call_ctrl(entry)` keyed on `(entry.name, entry.kind)` so repeated calls return the same instance. The cache should clear when the entry is unregistered/replaced.

### [INFO] `trace_capture.py` is dead code

`src/agentdojo_target/trace_capture.py` contains only a docstring and `from __future__ import annotations`. The functionality it describes is implemented eagerly in `runtime_wrapper.WrappedFunctionsRuntime._trace`.

Recommended fix: delete the file, or replace its body with a one-line re-export of `WrappedFunctionsRuntime.trace` for callers who imported `from agentdojo_target.trace_capture import ...`.

### [INFO] `get_observables()` rebuilds the seed env on every call

`target.py:215-237` calls `_build_seed_env_with_overrides()` which calls `load_composite_seed()` which performs four YAML reads + parses + pydantic validations every time the controller initializes an optimizer (`controller.py:505`). At 100 tasks under one scope, that's 400 YAML reads. Not a correctness issue, but the seed env is functionally constant per configured target instance, so an `@lru_cache` keyed on `tuple(sorted(self._seed_overrides.items()))` would knock 90%+ off the cold-start cost. Worth scoping if the perf budget tightens.

## Strengths

- The five-phase `run()` decomposition makes the lifecycle scannable, and the docstring at the top of `target.py` describes the order precisely.
- `_build_read_controllables` (`controllables.py:235-266`) does both directions of consistency checking: missing entries in `READ_QUADRANT_MAP` AND stale entries are both fatal at import time, so a tool-list drift can't silently land.
- Security-domain assignment for the four catalog Controllables (register on the addable tag, replace/unregister/rewrite on the broad tag) correctly models the capability subsumption from the brief, and `test_tool_catalog_register_is_weakest_capability` pins it.
- `cleanup()` carefully separates per-run state (env, catalog, wrapper, messages, trace) from per-task config (system_prompt, user_prompt, seed_overrides), so the multi-run loop reuses task configuration without picking up stale env mutations.

## Open questions

- Should the per-call attacker Controllables (`tool_call:{name}`) appear in `get_controllables()` so optimizers can discover them up front, or is the implicit "you'll see these only if you used the catalog Controllables to spawn them" contract intentional? Either way the docstring on `CONTROLLABLES` should say so.
- Is the divergence between upstream's `model_output_from_messages` (returns `list[MessageContentBlock] | None`, raises on non-assistant last message) and the port's text-only string extraction acceptable for the faithfulness pairs, or should the port match upstream exactly and only string-coerce at the `last_response` query boundary?
- The seed-overlay-vs-initial-list trap (finding [MEDIUM] on `merge_yaml_overlay`): is the expectation that Tasks write `initial_emails`/`initial_events`/`initial_files` in their overlays, or should the merger auto-propagate? Whichever, it deserves a one-line warning in `_seed_yaml_override_spec`'s description and a test pinning the intended path.
- The 3-retry loop in `run()` was kept "for faithfulness" (per ASSUMPTIONS.md E.1), but the wrapped runtime accumulates per-call events across retries — duplicates are recorded as legitimate observations. Should the wrapper RESET trace + counters at the start of each attempt to make retries idempotent from the optimizer's perspective, or do we want optimizers to see the retry pattern?
