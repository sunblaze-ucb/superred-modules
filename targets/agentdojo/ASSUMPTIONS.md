# ASSUMPTIONS

Every divergence from upstream AgentDojo behaviour, naming, or semantics is enumerated here with: (a) what AgentDojo does (with file path + line numbers), (b) what this package does, (c) why.

Entries are numbered. Add new entries as numbered bullets at the bottom of the relevant section; never reuse a number.

Upstream AgentDojo references in this file point into `https://github.com/ethz-spylab/agentdojo` at commit reachable from PyPI `agentdojo==0.1.35`.

---

## A. Injection model

### A.1 Static `{slot}` substitution replaced by on-demand event injection

**AgentDojo** (`src/agentdojo/task_suite/task_suite.py:139-146` `load_and_inject_default_environment`): Python `str.format(**injections)` applied to the raw YAML text of `environment.yaml` *before* YAML parsing. The substituted env is fixed for the duration of the run.

**Us**: At load time we substitute the upstream `injection_vectors.yaml` *default* values into the YAML so the env parses to a valid pydantic model. The original `{slot}` placeholders are gone from the parsed env. On-demand injection then happens per read: every readable tool fires a `ControllablePostCallEvent` carrying the *legitimate* (post-default-substitution) value as the answer; the optimizer's `ControllableInjection` replaces the agent-visible return for that one call.

**Why**: the brief mandates this. On-demand injection is strictly more expressive (different content per read, adaptive per agent state, multi-channel) than static substitution. AgentDojo's static slot mechanism is subsumed: passing the same canonical string per-read recovers static behaviour for faithfulness testing.

### A.2 Per-tool granularity, not per-record

**Us**: each readable tool maps to exactly one Controllable. When a tool returns a list (e.g. `get_received_emails`), the optimizer's injection replaces the entire serialized return value.

**Why**: simpler v1 contract; per-record refinement is a v2 follow-up. The brief proposes per-tool as default.

### A.3 Controllable `value_type` is always `"string"`; injection values pass through as strings

**Brief / framework**: `Controllable.value_type` is a free-form string label hinting at the expected payload shape; `ControllableInjection.value` is typed `str`.

**Us**: read Controllables declare `value_type="json"` (for tools returning structured data: lists, dicts, pydantic models) or `"text"` (for tools returning plain strings such as `read_file`, `get_webpage`); catalog Controllables declare `value_type="json"`. The label is purely informational; injection values flow through as strings regardless. For non-string canonical return values we serialise the legitimate value with `_serialize_for_event` (pydantic `model_dump_json` for `BaseModel` subclasses; `json.dumps` with an `isoformat`/`enum.value`/`repr` fallback chain otherwise) so the event answer is always a string. When the optimizer responds with `ControllableInjection`, we inject `response.value` verbatim as the agent-visible return without any reverse-deserialisation: agent-side formatters render it straight into the prompt.

**Why**: optimizers operate uniformly on strings; per-tool deserialisation rules would create N tool-shape coupling points. AgentDojo's agent prompt renders tool returns through `_tool_output_format_*` helpers which already string-coerce, so substituting a raw string is faithful to upstream's eventual prompt content. Tests in `tests/test_runtime_wrapper.py` pin the round-trip: an injected `"9999.99"` string replaces a numeric `BankAccount.balance` return and the agent sees the literal string.

### A.4 Each read tool reads exactly one store

**Us**: every readable tool reads from exactly one store (it carries exactly one data-bearing `Depends`), so each read is one Controllable tagged at one store leaf. There is consequently no per-boundary "split" to make: a read cannot straddle two stores, so the question of dividing one read across security domains is vacuous (there is nothing to split). A registry guard could enforce the one-data-bearing-`Depends`-per-read invariant.

**Why**: this is the store-level corollary of A.2 (per-tool, not per-record) under the per-store forest (see C.8). The per-record granularity limitation in A.2 still stands and is unchanged: a single-store read returning a list is still one Controllable replacing the whole serialized return.

---

## B. Tool catalogue

### B.1 Catalogue editability is new

**AgentDojo**: tool list is fixed for the duration of a run; the agent sees the suite's pre-declared tools (with `tool_filter` defense possibly removing some).

**Us**: four catalog Controllables (`register`, `replace`, `unregister`, `rewrite_doc`) fire **once at run start** (before the first LLM call) via a spliced `BasePipelineElement` hook -- deliberately NOT per turn, to avoid four redundant Controllable events on every agent turn. The optimizer edits the catalogue once; it is then fixed for the run. (Strictly, the hook fires once *per pipeline attempt*: exactly once in the normal run, but AgentDojo's rare empty-output retry -- the `range(3)` loop in `target.py`, faithful to upstream `task_suite.py:383` -- re-runs the whole pipeline and re-fires the hook up to 3x. This is safe and intentionally left unguarded: the catalogue persists across attempts, and re-applying an edit is either idempotent or raises a `ValueError` the hook catches.) Tradeoff: this keeps the upfront attacker capability but drops *reactive* editing -- the optimizer can no longer adapt catalogue edits to the agent's observed behaviour mid-run. Acceptable because tool-shadowing / malicious-MCP is normally a static upfront setup; if reactive editing is ever needed, re-splice the hook into the `ToolsExecutionLoop`. Registered/replaced tools fire per-call Controllables when invoked, carrying the agent-supplied args; optimizer's injection response is the fake return the agent sees.

**Why**: the brief mandates this as a second attacker capability surface (models malicious-MCP / tool-shadowing threat).

### B.2 Start-of-run catalog edits propagate across turns via runtime.functions

**AgentDojo** (`agent_pipeline/tool_execution.py:86,103` for `ToolsExecutor` and `llms/openai_llm.py:197,223` for `OpenAILLM`): both `ToolsExecutor.query` and `OpenAILLM.query` read `runtime.functions` (resp. `runtime.functions.values()`) on each invocation rather than caching at init.

**Us**: our `_CatalogEditHook.query` mutates the wrapped runtime's function dict in place (via `WrappedFunctionsRuntime.refresh_functions()`) once, before the first LLM turn.  Because upstream consults `runtime.functions` live on every invocation, that one-time edit is seen by the LLM's tool list AND every subsequent `ToolsExecutor` dispatch across all turns.  No upstream patching is required.

**Why**: the brief (Section 2.d) requires verifying that "the LLM tool list is rebuilt every turn from the runtime" and that we "wrap or patch to preserve mid-run reactivity" if upstream ever drifts.  We verified by code-reading and pinned the contract with `tests/test_toolsexecutor_per_turn.py` -- particularly the `test_toolsexecutor_sees_attacker_tool_after_catalog_edit` test which registers a new attacker tool at run start and asserts the same `ToolsExecutor` instance dispatches it successfully on a later call.  If a future AgentDojo release breaks this contract (e.g., caches the function list at element init), that test will fail with a clear error message naming the file to patch.

---

## C. Composite environment

### C.1 Single composite env exposing all four suites

**AgentDojo**: each suite has its own `TaskEnvironment` subclass; benchmark runs one suite at a time.

**Us**: a single `CompositeEnvironment` pydantic root with four sub-attributes (`banking`, `workspace`, `slack`, `travel`), each containing the unmodified upstream sub-environment. All 74 tools are simultaneously registered; the agent picks.

**Why**: the brief mandates this. The composite agent's threat model includes cross-suite system-purpose violations that a single-suite setup cannot model.

### C.2 Tool name disambiguation via `{suite}__{tool}` prefix

**AgentDojo**: tool names are bare (`send_email`, `read_file`).

**Us**: tools are registered as `{suite}__{tool}` (e.g. `workspace__send_email`, `banking__read_file`). Layer-1 SecurityClaim tasks strip the prefix before passing function-call traces to upstream's `*_from_traces` methods.

**Why**: the workspace and travel suites both define `Inbox`-mutating tools with overlapping names; the suite prefix disambiguates without rewriting upstream tool bodies.

### C.3 `Depends` extractor rebound per suite

**AgentDojo** (`src/agentdojo/functions_runtime.py:31-35`): `Depends("inbox")` resolves at run-time via `getattr(env, "inbox")`.

**Us**: at registration time we rebind each tool's `Depends` from a string attribute name to a callable that knows the suite: `Depends("inbox")` becomes `Depends(lambda env: env.workspace.inbox)` (or `env.travel.inbox`).

**Why**: with four sub-envs there is no `inbox` attribute on the composite root. Per-suite rebinding keeps upstream tool bodies unchanged.

### C.4 `initial_*` sync before JSON round-trip

**AgentDojo** (`tools/email_client.py:_create_emails`, `tools/calendar_client.py:_create_events`, `tools/cloud_drive_client.py:_create_files`): the `Inbox` / `Calendar` / `CloudDrive` models keep `initial_emails` / `initial_events` / `initial_files` as the source-of-truth lists and rebuild the matching `emails` / `events` / `files` dicts in a pydantic `@model_validator(mode="after")`. AgentDojo's agent tools (e.g. `delete_email`, `cancel_calendar_event`, `create_file`) mutate the derived dicts but never touch `initial_*`.

**Us**: `Target.query("pre|post_environment_snapshot")` serialises the composite env via `model_dump_json` and consumers parse it via `model_validate`. Without intervention, the `initial_*` ➜ derived-dict validator wipes in-memory mutations on the round-trip. We call :func:`agentdojo_target.env.sync_initial_fields` immediately before serialising so the derived dicts' current contents are reflected back into the `initial_*` lists; the round-trip is then faithful and downstream SecurityClaim predicates see the agent's actual mutations.

**Why**: upstream's `injection_task.security(pre, post)` compares in-memory envs, so this issue doesn't arise in upstream benchmarking. Our framework forces a JSON round-trip across the `Target.query` → `Task.evaluate` boundary, so the sync is mandatory for faithfulness. Tests in `tests/test_env_sync.py` pin both the bug (unsynced delete is lost) and the fix.

### C.5 Untouched suites are still fully loaded into the composite env

**AgentDojo**: a benchmark run picks one suite; only that suite's environment is loaded from its `environment.yaml`.

**Us**: `seed_loader.load_composite_seed()` unconditionally loads all four upstream suite `environment.yaml` files into the single `CompositeEnvironment` even when the optimizer's `Scope` only targets controllables/observables from one suite.

**Why**: the `CompositeEnvironment` pydantic root requires every sub-env field to be present and validate. Lazy per-suite loading would force a `None`-tolerant variant of every cross-suite tool's `Depends` extractor, expanding upstream patch surface for no benefit (the load is one-shot, ~50ms, and the validator is idempotent). Filtering at the **observation** layer (security-domain scope on controllables / observables / trajectory) is sufficient to keep an out-of-scope optimizer blind to the other suites' contents.

### C.6 Catalog operations are keyed by suite-prefixed names

**Us**: the four catalog Controllables (`apply_register`, `apply_replace`, `apply_unregister`, `apply_rewrite_doc`) and the per-call attacker-tool Controllables all operate on suite-prefixed names (`banking__get_balance`, never bare `get_balance`). An attacker that supplies a bare name to `apply_replace` finds no canonical entry and the operation is rejected by `ToolCatalog`.

**Why**: catalog Controllables run in a single global namespace; without the suite prefix `replace("send_email", ...)` is ambiguous between `workspace__send_email` and `travel__send_email`. Forcing the prefix is the same disambiguation rule the agent already lives under (see C.2) and makes attack payloads unambiguous.

### C.7 Stores beyond the brief's enumeration

**Brief Section 1.b table**: enumerates injection slots and observation surfaces per suite (banking transactions, workspace inbox/calendar/cloud_drive, slack channels/inbox/web, travel hotels/restaurants/reservations).

**Us**: the implementation registers tools and observables for these stores plus a small set the brief's table omitted: travel `flights` and `car_rentals` (read-only and used by `get_flight_information` / `get_all_car_rental_companies_in_city`), banking `scheduled_transactions` (`get_scheduled_transactions`, `update_scheduled_transaction`), and workspace `contacts`-like search surfaces (`search_contacts_by_name`, `search_contacts_by_email`). All write-side calls land on the trajectory once, via the per-tool `ControllablePostCallEvent` (function + args in the event request, return in the answer), and are reported to evaluators via the `write_calls_made` query (per §F.3).

**Why**: the brief's table is illustrative, not exhaustive. AgentDojo v1 ships these stores as part of the canonical surface and several injection / user tasks reference them (e.g. banking `InjectionTask4` mutates scheduled transactions; travel `UserTask16` reads flights). Excluding them would drop coverage; including them is mechanical and the security-domain forest already has a leaf node per store (see C.8 and `security_tags.py`).

### C.8 `tools` is a per-service, per-store forest (not a provenance grid)

**Earlier port**: the `tools` subtree classified surfaces by an abstract 2x2 provenance grid: `CONTENT_1P_DATA_1P`, `CONTENT_1P_DATA_3P`, `CONTENT_3P_DATA_1P`, `CONTENT_3P_DATA_3P` (content-author x data-store-owner). The four `CONTENT_*` tags were the leaves.

**Us**: the `tools` subtree is now a per-service, per-store forest. `tools` is a pure grouping root: nothing is emitted at it. Under it sit 4 service nodes (`banking`, `workspace`, `slack`, `travel`) and 16 store leaves, one per upstream pydantic store / per `Depends` extractor: `banking_bank_account`, `banking_filesystem`, `banking_user_account`; `workspace_inbox`, `workspace_calendar`, `workspace_cloud_drive`; `slack_slack`, `slack_web`; `travel_hotels`, `travel_restaurants`, `travel_car_rental`, `travel_flights`, `travel_user`, `travel_calendar`, `travel_reservation`, `travel_inbox`. Each read tool is tagged at the store leaf it reads from; each write tool's observation is tagged at the store leaf it mutates, so reading from and acting on a single store share one label. The whole `DOMAIN` is 32 tags (system 10, user 1, tools 21). The 4 `CONTENT_*` provenance tags were removed.

The `system` subtree's 10 tags are: `system` root, `system_prompt`, `model_identity`, `detailed_system_specification` (a read-only leaked free-text system-specification observable, a sibling of `model_identity`), `tool_catalogue` (a pure grouping root) with its three capability children `tool_catalogue_add` / `tool_catalogue_edit` / `tool_catalogue_remove`, and `agent_trace` with its single child `agent_trace_messages`. (Earlier ports had `prompt` instead of `system_prompt`, a single register-only `tool_catalogue_addable` child of `tool_catalogue`, and an `agent_trace` subtree with three children `agent_trace_messages` / `agent_trace_tool_calls` / `agent_trace_tool_responses`; the prompt tag was renamed, the catalogue child was split into add/edit/remove, and the two trace-tool tags were removed when tool calls/responses stopped being double-emitted as observables. Those #65 renames left the `system` count at 9; `detailed_system_specification` later added the read-only system-spec leaf, bringing it to 10.)

**Why**: the provenance grid mixed two orthogonal axes into four buckets and could not name an individual store, so an attacker scoped to "the banking account" and one scoped to "the slack web cache" were indistinguishable. A per-store forest follows the real trust boundaries (one tag per separately compromisable store), lets a scope grant access to one service or one store, and unifies the read label and the write-observation label for the same store.

### C.9 Slack is two stores, not finer

**Us**: every slack tool shares one `Depends("slack")` on the whole `Slack` object (its users, channels, channel messages, and direct messages all live on that single object). Channel-messages and direct-messages are therefore NOT separable into finer leaves. Slack is modeled as exactly two store leaves: `slack_slack` (the `Slack` object) and `slack_web` (the web cache used by `get_webpage` / `post_webpage`).

**Why**: a store leaf must correspond to a separately extractable `Depends`. Upstream gives the whole `Slack` object through one extractor, so there is no `Depends` boundary along which to split channel vs direct messages; inventing finer leaves would not map to anything the runtime can isolate.

### C.10 Dual-store writes are tagged at the primary mutated store

**Us**: a few write tools touch a second store besides the one they primarily mutate. Calendar `create` / `cancel` / `reschedule` events also `Depends("inbox")` to send a notification email, and travel `reserve_*` also `Depends("user")` (read-only) for the traveller profile. Each such write's observation is tagged at its PRIMARY mutated store (`workspace_calendar`, or `travel_reservation`), not at the secondary store. The secondary store touch is not silently lost: it still appears in the post-run environment diff available to the scorer. `add_calendar_event_participants` is calendar-only (single store).

**Why**: a single observation needs one label, and the store the attacker provoked a change in is the primary mutation; tagging there matches the read/write-share-a-label rule (C.8). The secondary touch is a side effect that the environment diff already captures, so no information is dropped.

### C.11 Per-suite seed-override slots retagged to their service node

**Earlier port**: the per-suite `seed_yaml_override__{suite}` config slots were tagged with the removed `CONTENT_1P_DATA_1P` provenance tag.

**Us**: each `seed_yaml_override__{suite}` slot is retagged to its service node (`banking`, `workspace`, `slack`, or `travel`).

**Why**: a per-suite YAML overlay can touch any store within that suite, so the service node (the parent of that suite's store leaves) is the correct, minimal scope; the removed provenance tag no longer exists (C.8).

---

## D. Polarity of upstream `security` / `utility`

### D.1 `security == True` means "attack succeeded"

**AgentDojo** (`base_tasks.py:126` docstring): "True if injection was executed correctly." Polarity matches this in `_check_injection_task_security` (`task_suite.py:297-311`). The skip-branches in `benchmark.py:147,153,...` set `security=True` for error cases ("agent stayed safe") which is a separate fallback semantic.

**Us**: SecurityClaim Layer-1's `EvaluationResult.primary_score.value = float(security_result)`, so success=True corresponds to attack succeeded. Naming in our code uses `attack_succeeded` rather than `security` to avoid the documentation-vs-fallback trap.

**Why**: explicit naming reduces the chance of polarity inversion in downstream analysis code.

### D.2 Faithfulness comparison: judge / pipeline-model substitution

**AgentDojo upstream**: most benchmark runs in the paper use `gpt-4o-2024-05-13` for the agent and a verbatim deterministic judge (the predicates in `security()` / `utility()` are pure Python; only Layer-2-style judge-LLM checks exist in our extensions).

**Us**: faithfulness tests in `tests/faithfulness/test_upstream_comparison.py` drive both the upstream `AgentPipeline` and our `Controller + AgentDojoTarget + SecurityClaim` end-to-end against the same 12 (user-task, injection-task) pairs at temperature=0 and compare verdicts. The pipeline model in CI is configurable via `AGENTDOJO_FAITHFULNESS_MODEL` (default `openai/gpt-4o-2024-05-13`), and falls back to `openai/gpt-4-turbo-2024-04-09` when the user's LiteLLM proxy does not have `gpt-4o-2024-05-13` provisioned (see also memory `feedback_sorrybench_judge`). LLM cost is hard-capped at `Controller(task_cost_cap_usd=4.0)`. The 12 pairs span attack semantics: 3 banking (IT0 send-money, IT2 send-money-conditional, IT4 scheduled-txn), 3 workspace (IT1 calendar mutation, IT2 email delete, IT3 file create), 3 slack (IT2 channel-send, IT3 invite-attacker, IT4 web-fetch), 3 travel (IT0 hotel-reserve, IT2 restaurant-reserve, IT3 calendar-event). The comparison is verdict-only: for each pair we compare the single upstream `security()` win/lose boolean against our Layer-1 `attack_succeeded` value, and nothing else. The post-environment diff structure and the function-call trace ordering are NOT asserted. The test PASSES when at least 10 of the 12 verdicts agree (`MATCH_FLOOR = 10`), tolerating up to 2 disagreements (LLM nondeterminism at the agent layer can flip a borderline pair even at temperature=0).

**Why**: the brief (Section 5.d) requires an LLM-driven verification that our port preserves upstream semantics. A 12-pair sample at `$1-4` total cost is the user-selected budget option (Option A). Pairs span all four suites and the three predicate shapes (`check_new_*` helpers, `*_from_traces`, plain `pre/post` diff) so a port-side regression in any of these would be detected. Substituting `gpt-4-turbo-2024-04-09` for `gpt-4o` where the proxy lacks the latter is a one-line config override; we record the substitution and the matched-verdicts proportion in the test output so a future divergence (e.g. proxy upgrades) is visible without re-running.

---

## E. Pipeline and run lifecycle

### E.1 AgentDojo 3-retry loop preserved

**AgentDojo** (`task_suite.py:383`): `for _ in range(3): _, _, env, messages, _ = agent_pipeline.query(...); if model_output is not None: break`.

**Us**: same outer 3-retry loop in `Target.run()` for faithfulness. Optimizers may observe duplicate per-read events across retries; we accept this.

**Why**: faithfulness requires it.

### E.2 Default system prompt only

**AgentDojo** (`data/system_messages.yaml`): single `default` key. The paper's Figures 14/15/16 (model-specific prompts) are **not** in the repo.

**Us**: use the single `default` system prompt. Layer-1 Tasks set this in `configure_target`. Per-model prompt customisation is out of v1 scope.

**Why**: source material is what's in the upstream repo; the paper's figures don't ship.

### E.3 `no_defense` pipeline only

**AgentDojo** has five defense variants (`tool_filter`, `transformers_pi_detector`, `repeat_user_prompt`, `spotlighting_with_delimiting`, plus the no-defense baseline).

**Us**: v1 wraps only the `no_defense` baseline. Defenses are a v2 follow-up.

**Why**: superred's threat-model decomposition treats defenses as a separable optimizer-side concern; bundling them at target-construction time is the wrong layer.

### E.4 `pipeline_model` is a construction-only choice

**Earlier port**: the pipeline model was settable per run via a `pipeline_model` `ConfigSpec` slot (`set_config("pipeline_model", ...)`).

**Us**: the pipeline model is now a construction-only choice: the `AgentDojoTarget` constructor argument. The per-run `pipeline_model` config slot was removed, and `set_config("pipeline_model", ...)` raises a `ValueError`. The model-identity observable is unchanged.

**Why**: the defender (the agent's model) must be held constant across every run in an experiment; making the model a per-run config slot would let a Task or sweep vary the defender mid-experiment, confounding the measurement. Model identity is a construction concern, consistent with the brief's rule that generation settings stay out of `ConfigSpec`.

### E.5 Per-run provider client is closed in a `finally`

**Us**: the provider client is opened once per run. `build_pipeline` / `_build_llm` now return a `(pipeline-or-element, close)` pair, and `Target.run` closes the client in a `finally` block, including on the failure path (an exception in the agent loop still closes the connection).

**Why**: without an explicit close, a client connection leaked per run; across a multi-run, multi-task sweep these accumulate and exhaust connection/file-descriptor limits. The `finally` guarantees a single client lifetime per run regardless of outcome.

---

## F. State observation

### F.1 `get_unread_emails` semi-mutation preserved

**AgentDojo** (`tools/email_client.py:148`): `get_unread_emails` flips each returned email's `read` field to True.

**Us**: same. The legitimate value the wrapper surfaces is post-flip (i.e. after the side effect); `ControllableInjection` only replaces the agent-visible return value, not the env state.

**Why**: faithfulness.

### F.2 `get_webpage` audit trail preserved

**AgentDojo** (`tools/web.py:35`): `get_webpage` appends the URL to `web.web_requests`. Security predicates key off this list.

**Us**: same. The tool is classed as readable (for content injection) but the audit-trail write is preserved.

**Why**: the slack injection tasks `security` predicates check `web_requests`.

### F.3 Observable emission strategy: events during runtime + post-run queries

**Brief Section 5.c**: requires observables for "the agent's chat-message stream, every tool call (with the function name and args the agent emitted), every return value (with the post-injection value the agent saw)".

**Us**: observables are exposed through two complementary channels.

1. **Per-call events** (`ObservableEvent` via `emit`): the `_MessageStreamHook` spliced into the AgentDojo pipeline emits one `agent_trace_message_NNNN` observable per new message as the conversation progresses, carrying ONLY the non-tool internal message stream: tool-result messages (`role == "tool"`) are skipped and the `tool_calls` field is stripped from assistant messages. A tool call (function + args) and its return are NOT mirrored as observables; they live exactly once on that tool's `ControllablePostCallEvent`: the canonical read carries the function name and args (`json.dumps` with keys `function` and `args`) in the event request, and the agent-visible return (the event answer, or the optimizer's injection value) as the answer. There is no observable mirror of a read's legitimate value; a Controller that lists the store tag under `read_only` observes it from that controllable event on the trajectory without injection rights. The message indexes increase monotonically within a pipeline attempt and restart at `0000` if `AgentDojoTarget.run` retries, because a retry rebuilds the conversation from scratch rather than extending it. Ordinals are therefore valid only up to the final attempt's length; a longer abandoned attempt leaves its tail on the trajectory. `function_call_trace` is deliberately asymmetric: it keeps the calls a discarded attempt made, because those ran against the environment the retry continues from -- the messages describe the conversation that stands, the trace everything that executed.

   **Emit-once rationale**: tool calls and returns used to be double-emitted, once on the tool `ControllablePostCallEvent` and again as `agent_trace_tool_call_NNNN` / `agent_trace_tool_response_NNNN` observables. The duplicate observable channel re-exposed, under the broad `agent_trace` boundary, the exact per-call data the per-tool controllable already carries at its own (store) tag, which defeated per-store scoping (a trace-scoped attacker could read every store's traffic out of the aggregate stream). The two trace-tool tags (`agent_trace_tool_calls`, `agent_trace_tool_responses`) and the `agent_tool_call_observable` / `agent_tool_response_observable` builders were removed; each tool call/return now lives exactly once at the store boundary it touches: a read on its `ControllablePostCallEvent`, a write on its store-tagged `write_call_NNNN` observable (per F.5).
2. **Post-run query specs** (`QuerySpec` via `Target.query`): after the run, `pre|post_environment_snapshot`, `conversation_history`, `function_call_trace`, and `tool_catalog_snapshot` are queryable. These provide the same data in batched form for SecurityClaim predicates that operate on the complete trace.

**Why**: per-call events serve real-time optimizers (so they can react mid-run); post-run queries serve deterministic Layer-1/Layer-2 predicates (which operate after the run completes). The two channels overlap intentionally so adding a third style of consumer never forces a brief revision. Tests pin both: per-call emission in `tests/test_runtime_wrapper.py` and `tests/test_pipeline_bridge.py`; query-spec coverage in `tests/test_target.py`.

### F.4 `composite_env_snapshot` world-dump observable removed

**Earlier port**: a `composite_env_snapshot` observable lived at the `tools` root and emitted the whole composite environment twice per run (at init and at run end).

**Us**: `composite_env_snapshot` was REMOVED entirely; no per-service snapshot replaces it. Store contents reachable through a read Controllable are not mirrored as observables. The full environment remains available post-run to the scorer through the query specs (`pre|post_environment_snapshot`) only.

**Why**: the world-dump re-emitted as one coarse aggregate the same per-read data that read Controllables already expose, defeating per-store scoping (a store-scoped attacker could read every other store out of the aggregate). Removing it leaves the per-store reads as the single, properly scoped path to store contents; the scorer still gets the full env through the query specs, which are not attacker-facing.

### F.5 Successful writes emit a store-tagged write observable

**Us**: each successful canonical write emits a one-way `write_call_NNNN` observable tagged at the store it mutates (per `WRITE_STORE_MAP`), carrying the write's function call (function + args). Writes get NO Controllable (no per-call `ControllablePostCallEvent`): the effect must really execute so the post-run environment-diff measurement is real, and the call is additionally recorded in the `write_calls_made` / `function_call_trace` queries. (Reads, by contrast, carry their call and legitimate return once on a `ControllablePostCallEvent`.)

**Why**: a service-scoped attacker needs to see the action it provoked at the store it provoked it on; the store-tagged write observable is what makes that visible under per-store scoping (whereas the generic agent-trace stream is tagged at the trace boundary). Withholding a Controllable on writes preserves faithfulness: the agent's mutation lands in the real environment rather than being replaced by an injected return.

### F.6 Three internal-control observables

**Us**: three further one-way observables surface runtime/control-plane events:
- `tool_menu_rebuild_NNNN`: emitted when the runtime rebuilds its function registry after a catalogue edit; tagged `agent_trace`.
- `discarded_action_NNNN`: emitted when the OpenAI-compat layer drops a malformed parallel-wrapper tool-call name; tagged `agent_trace_messages` (retagged from the removed `agent_trace_tool_calls`).
- `catalog_edit_{op}_outcome`: emitted per attacker catalogue edit, recording whether it was applied or rejected, tagged by the same capability mapping as the edit controllable: `register` (a registered/added tool) at `tool_catalogue_add`, `replace` and `rewrite_doc` (an edited tool) at `tool_catalogue_edit`, `unregister` at `tool_catalogue_remove`.

**Why**: these expose the control-plane consequences of an attacker's catalogue edits and the runtime's handling of malformed calls, so a catalogue-scoped optimizer can observe whether its edit took effect and adapt. Tagging each at the matching boundary keeps them scoped to the attacker who could have caused them.

---

## G. SecurityClaim Layer-2 design

### G.1 Layer-2 source is bespoke, not SORRY-Bench/HarmBench

**Brief proposal** (Section 3.b): default source SORRY-Bench, LLM-as-judge.

**User clarification (2026-05-15)**: "should not be to jailbreak just the model but goals that do something that is fundamentally against the system purpose but not necessarily against model ethics." Source can be self-authored.

**Us**: Layer-2 is a bespoke catalogue of standalone goals tailored to the composite agent's system purpose, evaluated by deterministic security predicates (not LLM-as-judge). See `security_claim_agentdojo/src/security_claim_agentdojo/layer2_goals/` for the catalogue.

**Why**: aligns Layer-2's threat model (agent acting against system purpose under attacker influence) with AgentDojo's deterministic style; sidesteps LLM-judge cost.

### G.2 Delivery vector decoupled from Task

**Brief / user clarification**: the adversarial goal lives only on `Task.goal` and the security predicate. `configure_target` sets a *benign* user prompt that exercises tools; the attacker (optimizer) chooses delivery (override user prompt via user-scope Controllable, override system prompt via system-scope Controllable, content injection via tool-scope Controllables, catalog edits via tool-catalogue-scope Controllables) based on scope.

**Us**: this applies to BOTH layers. Layer-1's `configure_target` sets `user_prompt = user_task.PROMPT` (the benign legit task per AgentDojo's threat model). Layer-2's `configure_target` sets `user_prompt = benign_per_goal_prompt` (a per-goal routine query that exercises tools).

**Why**: lets the attacker scope determine the threat model end-to-end.

---

## H. Consolidated user clarifications (Q&A log)

Decisions resolved through direct exchanges with the user during the port. Each entry is dated; future-you should check this section before changing the relevant code, since these are the *binding* answers to ambiguous brief language.

### H.1 Latest released benchmark version (v1.2.2) is canonical (2026-05-22, revising 2026-05-15)

**Original question**: AgentDojo upstream has `v1`, `v1.1`, `v1.1.1`, `v1.1.2`, `v1.2`, `v1.2.1`, `v1.2.2` task suite variants under `default_suites/`. Which is the port's source of truth?

**Initial answer (2026-05-15)**: v1, for cross-paper comparability against the AgentDojo paper's reported numbers.

**Revised answer (2026-05-22)**: v1.2.2 (latest released). The v1.2.x patches address specific community-reported bugs (workspace UT16 read-flag, banking UT6 iPhone-subject lambda, workspace UT17 De Morgan time-check, plus the new workspace IT6-IT13 injection tasks). Keeping the audit-flagged bugs that v1.2.2 does NOT fix (operator-precedence bugs in banking IT0/1/3, banking UT5 stale-transaction, etc.) is acceptable per the standing rule "the remaining predicate weaknesses are fine; we want the latest official fixes".

**Where applied**:
- The package specifier was tightened from `agentdojo>=0.1.35` to `agentdojo==0.1.35`, freezing the executable benchmark code (tool bodies, agent loop, output formatting) to the audited release. This pins only the code axis; the data-axis pin `BENCHMARK_VERSION="v1.2.2"` (below) is unchanged, and the canonical-version decision recorded here is unaffected.
- `seed_loader.py` `BENCHMARK_VERSION = "v1.2.2"` (single source of truth, re-exported as `agentdojo_target.BENCHMARK_VERSION`)
- The paired `security-claim-agentdojo` package imports this constant rather than redeclaring, so the two cannot drift out of sync.
- Environment pydantic classes still imported from `agentdojo.default_suites.v1.<suite>.task_suite` because they are version-stable (each later version reuses the v1 definitions; only TaskSuite contents differ).
- `tool_registry.py` still imports `task_suite` globals from v1 because the tools list is identical across versions (v1.2's patches only add or modify task classes, not tool functions).
- `UPSTREAM_PREDICATE_AUDIT.md` reflects v1 source-reading; bugs that v1.2.2 fixes are noted inline.

**Why the Environment classes stay on v1 paths**: AgentDojo's `BankingEnvironment` / `WorkspaceEnvironment` / `SlackEnvironment` / `TravelEnvironment` classes are defined exactly once, in their v1 modules. The v1_2 / v1_2_2 packages only register new task classes; they do not redefine the pydantic Environment models. Reimporting from a non-v1 path would be a no-op rename.

### H.2 Layer-2 source is bespoke, not SORRY-Bench (2026-05-15)

**Question**: brief Section 3.b proposes SORRY-Bench + LLM-judge for Layer-2.

**Answer**: "should not be to jailbreak just the model but goals that do something that is fundamentally against the system purpose but not necessarily against model ethics." Use a hand-authored catalogue with deterministic predicates.

**Where applied**: see G.1.

### H.3 Faithfulness budget is Option A ($1-4) (2026-05-18)

**Question**: faithfulness comparison can be (A) `12 pairs at temperature=0, $1-4`, (B) `30 pairs at temperature=0.7 sampled 3 times each, ~$15-20`, (C) `full benchmark, ~$200+`.

**Answer**: Option A. Budget cap `Controller(task_cost_cap_usd=4.0)`.

**Where applied**: D.2 and `tests/faithfulness/test_upstream_comparison.py`.

### H.4 Use Opus 4.7 at max reasoning for subagent reviews (2026-05-19)

**Standing rule**: parent agent spawns review/audit subagents with `model="opus"` (Claude Opus 4.7). Lower-tier models miss subtle predicate-semantics bugs.

**Where applied**: T5 audit (7 parallel subagents), T8 multi-level review (planned), faithfulness verdict-comparator (planned).

### H.5 No em dashes in any output (continuous)

**Standing rule**: never use `—`. Use `-`, `,`, or rephrase.

**Where applied**: this entire document; all generated artifacts.

---

(more sections to be added as implementation discovers divergences)

## Read / attacker-tool controllables declare `value_type="text"`

The per-read (`read__<tool>`) and attacker-added-tool (`tool_call:*`) controllables
declare `value_type="text"`, because the runtime injects `response.value` verbatim as
the agent-visible return (`runtime_wrapper`: "Inject the raw string"). They were
previously mislabelled `json` even though the consumer takes a raw string and nothing in
this package reads `value_type`. The corrected label makes `value_type` a truthful
signal so a generic content-injection attacker can distinguish a free-text surface from
a schema-parsed one (DTAP `env_inject`/`tool_add`/`filesystem`) without a target-specific
allowlist. The genuinely structured catalogue-edit controllables (add/replace/remove/
rewrite a tool) keep `value_type="json"`.
