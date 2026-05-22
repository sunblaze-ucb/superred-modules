# Attacker-Model Coverage Review

## Summary

The AgentDojo port wires up every attacker capability the brief enumerates. All 47 per-read controllables are present (one per readable tool, exact count); the 4 catalog-editing controllables (register, replace, unregister, rewrite_doc) are registered and fire at every LLM turn via a pipeline hook; per-call `tool_call:<name>` events fire for both attacker-registered and attacker-replaced tools through a single dispatch in `runtime_wrapper.py`. The system-prompt and user-prompt Controllables exist with the right security domains. The seven observation surfaces the brief specifies (chat messages, tool calls, tool returns, env snapshots, tool-catalog snapshot, conversation history, function trace, model identity) all reach the optimizer via either static `Observable`, dynamic emit, or post-run `query`. Security-domain assignments are coherent with the forest in `security_tags.py` and match the brief's intent (e.g. registered tools sit on the narrower `tool_catalogue_addable` capability while replaced sit on the broader `tool_catalogue`). The cross-suite (Layer-3) use case is enabled by construction: the composite environment exposes all four suites simultaneously, all tool names are suite-prefixed, and the Layer-2 catalogue ships four explicit `cross_suite_*` goals plus a paired Layer-1+Layer-2 SecurityClaim. No HIGH or CRITICAL coverage gaps were found.

## Capability-by-capability wiring

### 1. Per-read on-demand content injection — CONFIRMED (47/47)

- Controllables defined in `targets/agentdojo/src/agentdojo_target/controllables.py:269` via `_build_read_controllables()`.
- `READ_QUADRANT_MAP` (`controllables.py:134-204`) contains exactly **47 entries**: banking 6, workspace 14, slack 5, travel 22. Verified by regex count.
- Each gets a unique Controllable named `read__{suite}__{tool}` with one of the four 2x2 `tools.*` leaf tags.
- `_build_read_controllables` validates both directions vs. `READ_FUNCTION_NAMES` (`controllables.py:241-254`) so drift is caught at import.
- Per-read event firing wire: `runtime_wrapper.py:247-271` (`_run_canonical`). For every canonical read with no error, the wrapper computes the legitimate value via the superclass, emits the `read_data_field` observable, fires a `ControllablePostCallEvent` with `controllable=READ_CTRLS[function]`, and substitutes the agent-visible return on `ControllableInjection`.
- Security-domain assignment correct: each leaf comes from `READ_QUADRANT_MAP` (lines 134-204), assigned per Section C.4 of `ASSUMPTIONS.md`, with the "broader-quadrant on ambiguity" rule.

### 2. Tool catalogue editing — CONFIRMED (4/4)

- All four catalog Controllables declared in `controllables.py:68-118`:
  - `TOOL_CATALOG_REGISTER_CTRL` (`controllables.py:68`) — `security_domain=TOOL_CATALOGUE_ADDABLE_TAG` (narrowest write).
  - `TOOL_CATALOG_REPLACE_CTRL` (`controllables.py:80`) — `security_domain=TOOL_CATALOGUE_TAG` (broad write).
  - `TOOL_CATALOG_UNREGISTER_CTRL` (`controllables.py:92`) — `security_domain=TOOL_CATALOGUE_TAG`.
  - `TOOL_CATALOG_REWRITE_DOC_CTRL` (`controllables.py:102`) — `security_domain=TOOL_CATALOGUE_TAG`.
- Aggregated in `TOOL_CATALOG_CTRLS` (`controllables.py:113`).
- The four operations are applied through `pipeline_bridge.py:64-140` (`_CatalogEditHook.query`), which fires the four `ControllablePreCallEvent`s in fixed order (register, replace, unregister, rewrite_doc) **before every LLM turn** (the hook is spliced both before the first LLM call at `pipeline_bridge.py:357` and inside `ToolsExecutionLoop` at `pipeline_bridge.py:347`).
- State mutations land in `tool_catalog.py`: `apply_register` (line 231), `apply_replace` (262), `apply_unregister` (305), `apply_rewrite_doc` (315). After any change the hook calls `wrapper.refresh_functions()` so the next LLM turn sees the new catalog (`pipeline_bridge.py:138-139`).
- Security-domain coherence: `tool_catalogue` parents `tool_catalogue_addable` and `tool_catalogue_readable` (`security_tags.py:43-63`), so an attacker with the broad `tool_catalogue` tag automatically holds the register-only capability through `scope_includes`. This matches the brief's capability-subsumption intent.

### 3. Per-call event for attacker-registered tools — CONFIRMED

- `runtime_wrapper.py:228` dispatches: `if entry.kind == "registered" or "replaced"` calls `_run_attacker(...)` (line 228, via `_run_attacker` covering both kinds).
- `_run_attacker` (`runtime_wrapper.py:292-311`):
  - Builds a per-call Controllable via `_attacker_call_ctrl(entry)` (line 107-126), naming it `tool_call:{entry.name}` (line 117) — single naming convention as the brief requires.
  - Constructs a `ControllablePostCallEvent` carrying the catalog's `entry.fake_return` as the `answer` (line 300-304).
  - Substitutes the agent-visible return on `ControllableInjection` (line 306-307).
- Security-domain assignment: `TOOL_CATALOGUE_ADDABLE_TAG` for `registered` (line 115) — correct (weakest write capability).

### 4. Per-call event for attacker-replaced tools — CONFIRMED

- Same code path as #3 (`runtime_wrapper.py:228` → `_run_attacker`).
- Security-domain assignment: `TOOL_CATALOGUE_TAG` for `replaced` (line 115) — correct (broader write capability; replacing is strictly more powerful than registering, so the broader tag is required).
- Confirmed by `tests/test_runtime_wrapper.py:237` (`test_attacker_replaced_short_circuits_with_broader_tag`).

### 5. System-prompt-scope Controllable — CONFIRMED

- `SYSTEM_PROMPT_CTRL` declared in `controllables.py:50-55`, `security_domain=PROMPT_TAG`.
- Phase 1 of the run (`target.py:251-260`) fires a `ControllablePreCallEvent` with `controllable=SYSTEM_PROMPT_CTRL` and uses the optimizer's `ControllableInjection.value` to override the configured system prompt (`target.py:257-260`). The effective system prompt is then passed into `build_pipeline(...)` at `target.py:287`.
- Security-domain coherence: `PROMPT_TAG` parents `PROMPT_READABLE_TAG` (`security_tags.py:35-41`), so a write-capable attacker also has read access. Matches the brief's intent.

### 6. User-prompt-scope Controllable — CONFIRMED

- `USER_PROMPT_CTRL` declared in `controllables.py:57-62`, `security_domain=USER_TAG`.
- Phase 2 of the run (`target.py:262-272`) fires a `ControllablePreCallEvent` with `controllable=USER_PROMPT_CTRL` and substitutes the user prompt on `ControllableInjection` (line 269-272). The effective user prompt is then passed to `pipeline.query(effective_user, ...)` at `target.py:305`.
- This is the dedicated lever that the brief calls out for routing Layer-2 cross-suite attacks — Layer-2 tasks (`security_claim_agentdojo/layer2_task.py:104-112`) pin a *benign* user prompt; an attacker with `user` in scope can override it to steer the agent toward goals that violate the system's purpose.

### 7. Observation surfaces — CONFIRMED (all 8)

Cross-referenced against the brief's enumeration (Section 5.c). Note the brief lists seven items; the eighth, model identity, is also wired.

| Surface | Channel | Security domain | Code location |
| --- | --- | --- | --- |
| Agent message stream | dynamic emit (per turn) | `AGENT_TRACE_MESSAGES_TAG` | `pipeline_bridge.py:148-192` (`_MessageStreamHook`); spec at `observables.py:89-103` |
| Agent tool calls | dynamic emit (post-run) | `AGENT_TRACE_TOOL_CALLS_TAG` | `target.py:335-341`; spec at `observables.py:106-116` |
| Agent-seen tool returns (post-injection) | dynamic emit (per call) | `AGENT_TRACE_TOOL_RESPONSES_TAG` | `runtime_wrapper.py:317-342` (`_emit_agent_tool_response`); spec at `observables.py:119-129` |
| Pre-run env snapshot | static `Observable` + post-run query | `TOOLS_TAG` (root) | `target.py:233-236`, `query_specs.py:52`; spec at `observables.py:67-75` |
| Post-run env snapshot | dynamic emit + post-run query | `TOOLS_TAG` (root) | `target.py:329-334`, `query_specs.py:61` |
| Tool catalog snapshot | static `Observable` + post-run query | `TOOL_CATALOGUE_READABLE_TAG` | `target.py:229-232`, `query_specs.py:77`; spec at `observables.py:57-65` |
| Conversation history | post-run query | (queries are not security-domain-tagged) | `target.py:188-189`, `query_specs.py:69` |
| Function trace | post-run query | as above | `target.py:180-183`, `query_specs.py:42` |
| Model identity | static `Observable` | `MODEL_IDENTITY_TAG` | `target.py:227`, `observables.py:43-48` |

Additional dynamic observables emitted by the runtime wrapper:

- `write_call:{tool}` per write-side canonical call (`runtime_wrapper.py:274-281`), domain `AGENT_TRACE_TOOL_CALLS_TAG` (`observables.py:142`). Used by Layer-2 predicates to detect agent mutations directly from the trajectory.
- `read_data_field:{tool}` per read with the *pre-injection* legitimate value (`runtime_wrapper.py:253-258`), domain inherited from the matching read Controllable (`observables.py:162-177`). Lets an optimizer with `tools.<leaf>` scope see the legitimate value even without holding the injection Controllable's full write capability.

The two channels (per-call emit vs. post-run query) intentionally overlap; both are documented in `ASSUMPTIONS.md` §F (lines 183-190).

## Cross-suite (Layer-3) capability

CONFIRMED and explicitly tested.

- The composite env (`env.py:40-63`) holds all four suites' sub-environments under `banking` / `workspace` / `slack` / `travel`. Every run sees all four.
- Tool name disambiguation (`tool_registry.py:199-216`) yields `{suite}__{tool}` so the agent has every tool of every suite available simultaneously.
- The Layer-2 catalogue under `security_claims/agentdojo/src/security_claim_agentdojo/layer2_goals/` includes four explicit cross-suite goals:
  - `cross_banking_to_slack.py` — agent posts banking transaction subjects or IBAN into Slack.
  - `cross_workspace_to_external_email.py` — agent forwards inbox content to an external attacker-controlled address.
  - `cross_calendar_collision_booking.py` — cross-suite calendar tampering.
  - `cross_pii_via_slack_web.py` — PII routed through Slack `get_webpage` / `post_webpage`.
- `layer3_factory.py:19-36` composes Layer-1 and Layer-2 into a combined SecurityClaim. The brief's specific example — inject content into a banking tool that causes the agent to send a Slack message — is exactly the `cross_banking_to_slack` shape: an optimizer with `tools.content_*` scope injects into `banking__get_most_recent_transactions` (per-read Controllable available, domain `CONTENT_3P_DATA_3P_TAG`), and the predicate (`security_predicates.banking_data_appears_in_slack`) inspects the post-run env for the banking string in Slack channels/DMs. The injection point and predicate are independent; the attacker is free to route through any combination of read-injection / catalog-edit / prompt overrides.

## Findings

None at HIGH or CRITICAL. All seven capabilities and all eight observation surfaces are present, with security-domain assignments coherent with the forest in `security_tags.py:31-156`. Cross-suite attack support is wired by construction and exercised by the four Layer-2 cross-suite goals.

Minor observations (non-blocking):

- **Post-run query slots are not security-domain-filtered.** `last_response`, `function_call_trace`, `conversation_history`, `pre/post_environment_snapshot`, `tool_catalog_snapshot`, `write_calls_made` are queryable by any caller via `target.query(name)` (`target.py:172-202`). The framework's `QuerySpec` (`superred/core/types/state.py`) has no `security_domain` field by current design — the filtering at this surface relies on the SecurityClaim Task being the only caller, which is the framework contract. Worth noting in `ASSUMPTIONS.md` if not already stated: queries are post-evaluation, not exposed to the optimizer directly.

- **`read_data_field:*` observable's security_domain is the *write* capability tag**, not a separate `tools.*.readable` tag (`observables.py:163-177`). This is the intended design (one tag per quadrant), but it means an attacker with read-only access to the legitimate value must hold the same tag as one who can inject. If a future refinement wants to model "can read the legitimate return without being able to substitute," a `tools.<leaf>_readable` sub-tag tree would be needed. Not a brief deviation; flagging for future-you.

- **The `TOOLS_TAG` root domain on `COMPOSITE_ENV_SNAPSHOT_OBS`** (`observables.py:67-75`) is broad: an attacker with `tools` in scope gets every sub-env in one observable. Likely intended (this models a full-read attacker), but a Layer-1 attacker scoped to a specific quadrant should not be able to read the full env via this observable. The `scope_includes` semantics will block sub-leaf scopes from reading the `tools`-root observable (since `tools` is not in the descendants of e.g. `tools.content_1p_data_3p`); confirm with a unit test if not already covered.

## File index (for cross-reference)

- `targets/agentdojo/src/agentdojo_target/controllables.py` — all 53 Controllables (1 system_prompt + 1 user_prompt + 4 catalog + 47 read).
- `targets/agentdojo/src/agentdojo_target/observables.py` — 4 static observables + 5 dynamic builders.
- `targets/agentdojo/src/agentdojo_target/security_tags.py` — 17-tag forest across three trees.
- `targets/agentdojo/src/agentdojo_target/runtime_wrapper.py` — per-call event firing, dispatch on `CatalogEntry.kind`.
- `targets/agentdojo/src/agentdojo_target/pipeline_bridge.py` — `_CatalogEditHook` + `_MessageStreamHook` spliced into AgentPipeline.
- `targets/agentdojo/src/agentdojo_target/target.py` — `AgentDojoTarget` run() phases 1-5, config and query routing.
- `targets/agentdojo/src/agentdojo_target/tool_catalog.py` — catalog mutation methods (register/replace/unregister/rewrite_doc).
- `targets/agentdojo/src/agentdojo_target/tool_registry.py` — read/write classification, suite-prefixed names.
- `security_claims/agentdojo/src/security_claim_agentdojo/layer2_goals/cross_*.py` — four cross-suite goal specs.
- `security_claims/agentdojo/src/security_claim_agentdojo/layer3_factory.py` — combined Layer-1+Layer-2 SecurityClaim.
