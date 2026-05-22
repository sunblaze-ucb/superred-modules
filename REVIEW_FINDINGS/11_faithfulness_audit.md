# 11. Faithfulness Audit: Port vs Upstream AgentDojo

Forensic comparison of `agentdojo_target/` and `security_claim_agentdojo/` against `agentdojo==0.1.35`. Each finding is classified JUSTIFIED (in ASSUMPTIONS.md), UNJUSTIFIED (HIGH/CRITICAL), or COSMETIC.

Methodology: per-file read of port sources alongside `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/`. Cross-referenced against `targets/agentdojo/ASSUMPTIONS.md` and `security_claims/agentdojo/UPSTREAM_PREDICATE_AUDIT.md`.

---

## Summary table

| # | Area | Finding | Classification |
|---|---|---|---|
| F1 | System prompt | Verbatim `default` from upstream YAML via `load_system_message(None)` | JUSTIFIED (E.2) |
| F2 | 3-retry loop | `for attempt in range(3)` preserved in `target.py:301` | JUSTIFIED (E.1) |
| F3 | `model_output` heuristic | Port's `_model_output_from_messages` returns `str|None`; upstream returns `list[MessageContentBlock]|None`. Port's loop-break semantic diverges for thinking-only or empty-list content. | **UNJUSTIFIED (HIGH)** |
| F4 | `except Exception: break` around `pipeline.query` | Upstream catches only `AbortAgentError` (dead code under `no_defense`); port catches everything. | **UNJUSTIFIED (MEDIUM)** |
| F5 | `_run_canonical` calls `super().run_function` | Tool bodies invoked verbatim with identical `raise_on_error` propagation | OK / FAITHFUL |
| F6 | `Inbox.get_unread()` side effect | Upstream method called untouched via super().run_function; `email.read = True` mutation preserved | JUSTIFIED (F.1) |
| F7 | `get_webpage` audit trail | Upstream tool called untouched; `web.web_requests.append(url)` preserved | JUSTIFIED (F.2) |
| F8 | Pydantic models (BankAccount, Inbox, Calendar, Web, etc.) | Used unmodified from upstream; CompositeEnvironment nests them by reference, no subclassing | OK / FAITHFUL |
| F9 | 5-tuple return from `AgentPipeline.query` | All 5 components (`_q, _runtime, new_env, messages, _extra`) destructured; `env` and `messages` propagated correctly | OK / FAITHFUL |
| F10 | LLM provider support | `_build_llm` raises `NotImplementedError` for non-openai/anthropic with clear message | JUSTIFIED (implicit in ASSUMPTIONS preamble; pipeline_bridge.py:274-279) |
| F11 | `Depends` extractor rebinding | Bound to `lambda env: env.<suite>.<attr>` callable; semantics preserved; tool bodies unchanged | JUSTIFIED (C.3) |
| F12 | Tool name prefix `{suite}__{tool}` | Used everywhere; layer1 task strips prefix before passing to upstream `*_from_traces` | JUSTIFIED (C.2) |
| F13 | Predicate dispatch order | Upstream: `*_from_traces` first (unconditional), fallback to `*()` when None. Port wraps both in `try/except Exception` before checking None. | **UNJUSTIFIED (MEDIUM)** |
| F14 | `function_call_trace` source | Recorded eagerly from `WrappedFunctionsRuntime.run_function`, not from message stream. Diverges from upstream `functions_stack_trace_from_messages` for tool calls rejected by `ToolsExecutor` before reaching the runtime (empty function names, names absent from runtime). | **UNJUSTIFIED (LOW)** |
| F15 | `Controllable.value_type` strings | ASSUMPTIONS A.3 says "every read Controllable declares `value_type='string'`"; code uses `value_type="json"`. SYSTEM_PROMPT_CTRL and USER_PROMPT_CTRL use `value_type="text"`. | **DOCUMENTATION DRIFT** (UNJUSTIFIED for the as-written ledger, but cosmetic for behavior) |
| F16 | `pre_environment` snapshot timing | Captured AFTER Phase 1 (system prompt event), Phase 2 (user prompt event), and Phase 3 (env build + override apply); BEFORE pipeline.query. Aligns with upstream's `pre_environment = task_environment.model_copy(deep=True)` after `init_environment`. | OK / FAITHFUL |
| F17 | `init_environment` mutation path | Port: serialised via `compute_init_env_overlay` to JSON, applied as YAML overlay during `_build_seed_env_with_overrides`. Upstream calls `user_task.init_environment(environment)` directly. Functionally equivalent. | OK / FAITHFUL (mechanical) |
| F18 | `sync_initial_fields` before JSON round-trip | Required because `Inbox`/`Calendar`/`CloudDrive` rebuild dicts from `initial_*` in pydantic validator | JUSTIFIED (C.4; comment in env.py incorrectly references "§C.3") |
| F19 | LLM temperature | Default 0.0 preserved via upstream `OpenAILLM`/`AnthropicLLM` constructors | OK / FAITHFUL |
| F20 | Pipeline shape | Adds `_CatalogEditHook` and `_MessageStreamHook` to upstream's `no_defense` skeleton. Otherwise identical. | JUSTIFIED (B.1, F.3) |
| F21 | `ToolCatalog.functions_for_runtime` refresh | Mutates `self.functions` dict in place per turn; upstream `OpenAILLM.query` and `ToolsExecutor.query` re-read `runtime.functions` per call so mid-loop edits propagate | JUSTIFIED (B.2) |
| F22 | Composite env loads all 4 suites unconditionally | `seed_loader.load_composite_seed()` always loads all suites | JUSTIFIED (C.5) |
| F23 | Single composite env exposing all 4 suites simultaneously | `CompositeEnvironment` with banking/workspace/slack/travel sub-attributes | JUSTIFIED (C.1) |
| F24 | Polarity `security_result -> attack_succeeded` | Layer-1 wraps `bool(security_result)` as `EvaluationResult.success` and `primary_score.value` | JUSTIFIED (D.1) |
| F25 | Defense variant | Only `no_defense` baseline ported | JUSTIFIED (E.3) |
| F26 | Per-tool injection granularity | Each readable tool maps to one Controllable; injection replaces entire return | JUSTIFIED (A.2) |
| F27 | `value_type` for read ctrls | Code uses `"json"`; ASSUMPTIONS A.3 documents `"string"` | **DOCUMENTATION DRIFT** in ASSUMPTIONS A.3 |

---

## Detailed findings

### F1 - System prompt (JUSTIFIED, E.2)

`agentdojo_target/system_prompt.py:20` calls `load_system_message(None)`, which is upstream's `agent_pipeline/agent_pipeline.py:52`. With `None` the function returns `system_messages["default"]` from `data/system_messages.yaml`. The YAML default is the 6-line "Emma Johnson / Blue Sparrow Tech / Don't make assumptions..." block. Layer-1 `configure_target` (layer1_task.py:131) re-applies it explicitly. Verbatim.

### F2 - 3-retry loop (JUSTIFIED, E.1)

`agentdojo_target/target.py:301` `for attempt in range(3)`. Upstream `task_suite/task_suite.py:383` `for _ in range(3)`. Same count, same break condition (`if model_output is not None: break`).

### F3 - `model_output_from_messages` heuristic divergence (UNJUSTIFIED, HIGH)

Upstream `task_suite.py:70-75`:

```python
def model_output_from_messages(messages) -> list[MessageContentBlock] | None:
    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message was not an assistant message")
    return messages[-1]["content"]
```

Returns the raw content **list** (truthy and not-None for any non-None content), or raises if the last message is not assistant.

Port `agentdojo_target/target.py:440-465` returns `str | None`. It joins all text-typed blocks and returns the joined string or None. Specifically:
- If content is a non-empty list with NO `type=="text"` blocks (e.g. only `thinking` blocks), upstream returns the list (not None -> break), port returns None (continues retrying).
- If content is `[]` (empty list), upstream returns `[]` (not None -> break), port returns None (continues).
- If the last message is non-assistant, upstream raises ValueError (the calling loop catches `AbortAgentError` only, so this propagates and crashes the run). Port returns None (continues retrying).

Real-world impact: the OpenAI/Anthropic clients in upstream typically populate `content` with at least one text block on the final assistant turn, so divergence rarely fires in practice. But under "thinking budget" model configurations where the only assistant content is thinking blocks, the port retries up to 3x while upstream would stop after 1.

Not flagged in ASSUMPTIONS. Recommendation: either match upstream's `messages[-1]["content"]` return shape (and adjust the break condition accordingly), or document this divergence.

### F4 - Broad exception catch around `pipeline.query` (UNJUSTIFIED, MEDIUM)

Port `agentdojo_target/target.py:309-313`:

```python
except Exception:
    logger.exception("AgentDojo pipeline.query raised on attempt %d", attempt + 1)
    break
```

Upstream `task_suite/task_suite.py:385-390`:

```python
try:
    _, _, task_environment, messages, _ = agent_pipeline.query(prompt, runtime, task_environment)
except AbortAgentError as e:
    task_environment = e.task_environment
    messages = e.messages
```

Upstream catches **only** `AbortAgentError` (raised by `pi_detector` defense; dead code under `no_defense`). Any other exception propagates and crashes the run. Port catches all exceptions and breaks out of the retry loop, preserving whatever messages were captured before.

Behavioral effects:
- Under no_defense (the only mode ported), AbortAgentError never fires, so upstream's try/except is essentially `try: ... except: pass`. Port's behavior differs only when SOME OTHER exception is raised — port suppresses, upstream would crash.
- Port retains messages from prior attempts via `self._messages` initialised to `[]` and updated inside the try block. Upstream's try block doesn't update messages on Abort.

Recommendation: add an ASSUMPTIONS entry. Either narrow the catch to `AbortAgentError` (and accept that real exceptions crash the run) or justify the broad catch as "the controller treats target-side exceptions as a failed run via `stop_reason='error'`, so target.run should swallow and log."

### F5 - `_run_canonical` raise_on_error semantics (FAITHFUL)

`runtime_wrapper.py:242` `result, error = super().run_function(env, function, kwargs, raise_on_error)` - delegates raise_on_error verbatim. The wrapper does **NOT** wrap the upstream runtime body in a try/except; if `raise_on_error=True`, errors propagate up exactly as upstream would. Faithful.

### F6 - `get_unread_emails` side effect (JUSTIFIED, F.1)

`runtime_wrapper.py:242` calls `super().run_function`, which invokes upstream's `get_unread_emails` (`tools/email_client.py:148`), which calls `inbox.get_unread()`. The latter (`email_client.py:120-124`) iterates `emails` and sets `email.read = True` on each. Mutation is on the actual pydantic `Email` instance held by `Inbox.emails[id]`, so the env state is updated in place. ControllableInjection replaces only the **agent-visible return value** (not the env state). Faithful.

### F7 - `get_webpage` audit trail (JUSTIFIED, F.2)

Same delegation path. `web.web_requests.append(url)` happens in upstream `web.py:41`. Faithful.

### F8 - Pydantic model usage (FAITHFUL)

`env.py:33-36` imports `BankingEnvironment`, `SlackEnvironment`, `TravelEnvironment`, `WorkspaceEnvironment` directly from upstream `default_suites.v1.*.task_suite`. `CompositeEnvironment` nests them as typed fields. No subclassing, no field reshaping, no validator overrides. Verbatim.

### F9 - 5-tuple return passthrough (FAITHFUL)

`target.py:303-308`:

```python
_q, _runtime, new_env, messages, _extra = await asyncio.to_thread(
    pipeline.query, effective_user, self._wrapped_runtime, self._env,
)
self._env = new_env
self._messages = messages
```

Upstream `AgentPipeline.query` returns `(query, runtime, env, messages, extra_args)`. Port captures `env` and `messages`; ignores the rest. Faithful.

### F10 - Provider error path (JUSTIFIED, implicit)

`pipeline_bridge.py:248-279` validates `provider/model` form, raises `NotImplementedError` for non-openai/anthropic with a clear message pointing to upstream `get_llm` for extension. ASSUMPTIONS preamble notes "openai, anthropic" only. Recommendation: bump this into an explicit ASSUMPTIONS entry under section E so the contract is unambiguous.

### F11 - `Depends` rebinding (JUSTIFIED, C.3)

`tool_registry.py:224-247` `_rebind_string_dep` and `_rebind_callable_dep`. String deps become `lambda env: getattr(getattr(env, suite), attr)`. Callable deps wrap upstream to receive the suite sub-env. Tool bodies are not modified; the rebinding only changes what env they receive.

### F12 - Tool name prefix (JUSTIFIED, C.2)

`tool_registry.py:199-216` defines `prefixed_name` (`{suite}__{tool}`) and `split_prefixed`. Layer-1 task `layer1_task.py:200-201` calls `name.removeprefix(prefix)` before passing FunctionCalls to upstream `*_from_traces`. Trace-prefix-stripping is suite-filtered: only this-suite calls are passed (cross-suite calls are silently dropped). This is a conscious choice — upstream predicates expect only their suite's tool names — but no upstream documentation specifies what to do with cross-suite calls in a single trace. Implicit but consistent with ASSUMPTIONS C.1/C.2.

### F13 - Predicate dispatch try/except (UNJUSTIFIED, MEDIUM)

Upstream `task_suite.py:281-311` dispatch sequence:

```python
def _check_user_task_utility(...) -> bool:
    out = task.utility_from_traces(...)
    if out is not None:
        return out
    return task.utility(...)
```

No exception handling. Any `Exception` from `utility_from_traces` propagates and crashes the run.

Port `layer1_task.py:237-256`:

```python
try:
    traced = self._user_task.utility_from_traces(model_output, pre_env, post_env, traces)
except Exception:
    traced = None
if traced is not None:
    return bool(traced)
try:
    return bool(self._user_task.utility(model_output, pre_env, post_env))
except NotImplementedError:
    return False
```

Two-tier swallowing. The port catches:
1. **Any exception** from `utility_from_traces` -> fall back to `utility()`. Lenient.
2. **NotImplementedError** from `utility()` -> return False. This is REQUIRED for `slack.UserTask11` (`UPSTREAM_PREDICATE_AUDIT.md` flags this: `utility()` raises NotImplementedError unconditionally; only `utility_from_traces` is implemented). For UT11 the upstream code path lands on `utility()` only when `utility_from_traces` returns non-None — port matches.

Difference 1 (broad except in tier 1) is not justified. A predicate bug in `utility_from_traces` (e.g. a KeyError because the trace has unexpected shape) would crash upstream but pass through to `utility()` in the port. This is silent-failure territory.

Recommendation: narrow the tier-1 catch to specific exception types or document the divergence.

### F14 - `function_call_trace` source (UNJUSTIFIED, LOW)

Upstream `functions_stack_trace_from_messages` (`task_suite.py:60-67`) iterates assistant messages and collects every `tool_call` from `message["tool_calls"]`. This INCLUDES calls that:
- Have empty function names (rejected by ToolsExecutor at `tool_execution.py:75-84`)
- Reference unknown tools (rejected at `tool_execution.py:86-95`)

Both cases produce error `ChatToolResultMessage`s but never invoke `runtime.run_function`. Upstream's trace records them anyway via the assistant message.

Port `runtime_wrapper.py:213-216` appends to `self._trace` only inside `run_function`. Tool calls rejected by ToolsExecutor never reach the wrapper. Therefore port's `function_call_trace` is missing these entries.

Real-world impact: small. v1 predicates' `*_from_traces` methods generally check `fc.function == "<name>"` or substring matches and don't care about malformed entries. But this is a behavioral divergence.

Recommendation: add a fallback path that derives the trace from messages on demand (e.g. a separate query slot `function_call_trace_from_messages`), or document the divergence.

### F15/F27 - `value_type` documentation drift (DOCUMENTATION DRIFT)

ASSUMPTIONS A.3 (line 31) says:

> every read Controllable in `controllables.py` declares `value_type="string"`; every catalog Controllable declares `value_type="json"`.

But `controllables.py:212-232` (`_make_read_ctrl`) sets `value_type="json"` for read controllables. `SYSTEM_PROMPT_CTRL` and `USER_PROMPT_CTRL` use `value_type="text"`.

This is documentation drift in ASSUMPTIONS A.3, not a behavior change. The framework's `Controllable.value_type` is documented as a free-form label. But the ASSUMPTIONS ledger is incorrect on its own contract. Recommendation: update A.3 to match the code, or change the code to match the ledger.

### F16 - `pre_environment` capture timing (FAITHFUL)

Port `target.py:298` captures `pre_env` after the env is loaded with overlays and after Phases 1+2 (system/user prompt events) but BEFORE `pipeline.query`. Upstream `task_suite.py:374` captures right after `init_environment(environment)` returns. Both are immediately before the agent gets a chance to mutate state. Faithful.

### F17 - `init_environment` replay path (FAITHFUL, mechanical)

Upstream: calls `user_task.init_environment(environment)` directly to get the mutated env. Port: calls the same `user_task.init_environment` inside `compute_init_env_overlay` (layer1_bridge.py:71), serialises the mutated env as JSON via `model_dump_json`, sets it as the `seed_yaml_override__{suite}` config slot, and the target's `merge_yaml_overlay` re-applies it as a deep-merged dict into the base seed.

This works because `init_environment`'s mutations are pydantic-model-state and the JSON round-trip plus deep-merge faithfully reproduces them (modulo the `initial_*` issue, which the port handles via `sync_initial_fields` — see F18). Equivalent in behavior, though mechanical.

Note: there is a subtle potential issue. `compute_init_env_overlay` calls `init_environment(baseline.model_copy(deep=True))` against the **default-injected** baseline (with injection_vectors substituted). Upstream's flow calls `init_environment(environment)` where environment is also the default-injected baseline. Same input, same function -> same output. Faithful.

### F18 - `sync_initial_fields` requirement (JUSTIFIED, C.4)

Port adds the sync call before serialising via `target.query("post_environment_snapshot")`. Upstream doesn't need this because it never serialises the env to JSON during a run. Port's JSON round-trip across `Target.query` -> `Task.evaluate` would otherwise discard derived-dict mutations. ASSUMPTIONS C.4 covers this. 

Minor doc nit: env.py:86 comment says "see ASSUMPTIONS.md §C.3" — the actual section in ASSUMPTIONS.md is C.4.

### F19 - LLM temperature (FAITHFUL)

`pipeline_bridge.py:258` `OpenAILLM(client, model_name)` and `:273` `AnthropicLLM(client, model_name)`. Both default temperature to 0.0 per upstream constructors (`openai_llm.py:181`, anthropic constructor similarly). Faithful.

### F20 - Pipeline shape (JUSTIFIED, B.1 + F.3)

Upstream no_defense: `[SystemMessage, InitQuery, llm, ToolsExecutionLoop([ToolsExecutor, llm])]`. Port adds:
- `_CatalogEditHook` BEFORE each LLM call (twice: once outer, once inside the loop) — for B.1 catalog editability.
- `_MessageStreamHook` AFTER each LLM call and AFTER the ToolsExecutor — for F.3 observable streaming.

Hook instances are reused across splice points so message-stream cursor advances monotonically. Faithful with documented additions.

### F21 - Catalog refresh propagation (JUSTIFIED, B.2)

`_CatalogEditHook.query` (pipeline_bridge.py:117-140) calls `self._wrapper.refresh_functions()` after applying any catalog mutation. `WrappedFunctionsRuntime.refresh_functions` (runtime_wrapper.py:184-191) replaces `self.functions` dict atomically. Upstream's `OpenAILLM.query:197` and `ToolsExecutor.query:103` re-read `runtime.functions` per invocation, so mid-loop updates take effect on the next turn without patching upstream code. Verified by code-reading and pinned by `tests/test_toolsexecutor_per_turn.py` (per ASSUMPTIONS B.2).

### F22-F26 - Other JUSTIFIED items

- F22 (C.5): all 4 suites loaded unconditionally. Trivially in code at seed_loader.py:43-46.
- F23 (C.1): composite env structure. env.py:40-63.
- F24 (D.1): `float(security_result)` polarity. layer1_task.py:151-154.
- F25 (E.3): no_defense only. build_pipeline only constructs the no_defense shape.
- F26 (A.2): per-tool injection. controllables.py builds one `READ_CTRLS[name]` per readable tool.

---

## Layer-2 (security_claim_agentdojo/layer2_*)

ASSUMPTIONS G.1 covers Layer-2 as bespoke (not faithful to AgentDojo, by design). No faithfulness comparison applies: predicates are hand-authored deterministic checks against the composite env. The implementation is self-contained and does not wrap any upstream predicate.

Note: Layer-2 `evaluate()` (layer2_task.py:124-133) uses `except Exception: violation = False`. Same swallowing pattern as Layer-1 (F13). Consistent with Layer-1, equally lenient.

---

## Notable "FAITHFUL" details worth recording

1. **`CompositeEnvironment.model_copy(deep=True)`** — port relies on pydantic's deep copy, which iterates all nested fields including `Inbox.emails: dict[str, Email]`. Verified to produce truly independent env instances; pre_env mutations are not visible in post_env (the env diff is well-formed).

2. **`Function.parameters` for attacker-managed tools** — port's `_PermissiveSchema` (tool_catalog.py:80-88) uses `ConfigDict(extra="allow")` so attacker-supplied tool calls don't get rejected by pydantic validation. This is needed because the attacker chooses both the registration schema and the agent's later call format; upstream doesn't model this case.

3. **`Tool catalog snapshot for query_specs.tool_catalog_snapshot`** — uses `entry.function.parameters.model_json_schema()` per-entry. Identical to what upstream `OpenAILLM._function_to_openai` would emit for the agent's tool-list prompt. Faithful.

4. **Trace ordering** — port records trace in `run_function` invocation order, which is interleaved with the agent's tool_calls. Upstream's `functions_stack_trace_from_messages` follows assistant-message order, but within each assistant message the tool_calls are in LLM-emitted order — same as the runtime invocation order. Equivalent.

---

## Recommendations

1. **F3 (model_output)**: align return shape with upstream OR add ASSUMPTIONS entry.
2. **F4 (broad except)**: narrow catch to `AbortAgentError` OR add ASSUMPTIONS entry explaining the controller-side stop_reason="error" path.
3. **F13 (predicate try/except)**: narrow tier-1 catch OR add ASSUMPTIONS entry.
4. **F14 (trace source)**: add a fallback `function_call_trace_from_messages` query slot OR add ASSUMPTIONS entry.
5. **F15/F27**: reconcile ASSUMPTIONS A.3 with code's actual `value_type` strings.
6. **F10**: lift implicit "openai, anthropic only" into an explicit ASSUMPTIONS entry under section E.
7. **F18 (doc nit)**: fix `env.py:86` reference from "§C.3" to "§C.4".

None of the findings invalidate the port's core threat-modeling contract: on-demand injection (A.1), composite env with all 4 suites (C.1), tool catalogue editability (B.1), default system prompt (E.2), 3-retry loop (E.1), polarity convention (D.1). All are JUSTIFIED.

The UNJUSTIFIED items are corner cases in:
- LLM output shape (F3) — fires only with thinking-only or empty content
- Exception suppression (F4, F13) — masks port-side bugs but won't change verdicts on a normal run
- Trace completeness (F14) — drops malformed LLM tool_calls that ToolsExecutor would have rejected anyway

Documentation drift (F15, F18 nit) is cosmetic but worth tightening before publication.
