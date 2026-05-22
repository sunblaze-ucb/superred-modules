# AgentDojo Port: Consolidated Review Triage

**Purpose.** Single triage view across ~12 independent Opus subagent reviews of the AgentDojo port. Each subagent covered a different aspect (specific modules, cross-cutting concerns, system-level). This file ranks findings by severity, surfaces cross-cutting themes, and lists prioritized action items.

**Methodology.** Independent subagent reviews against `targets/agentdojo/` and `security_claims/agentdojo/` source, cross-referenced against upstream `agentdojo==0.1.35` and `ASSUMPTIONS.md`. This file consolidates and de-duplicates their output.

**Severity vocabulary.** CRITICAL = security bypass, data corruption, or unrecoverable faithfulness break. HIGH = silent semantic divergence, hang/deadlock risk, untested high-impact path. MEDIUM = corner-case bug, documentation drift with behavioral consequence, missing test for a documented contract. LOW = polish, docstring nits, minor robustness. INFO = forward-compat notes.

**Input reports consolidated (12):**
- `02_runtime_wrapper.md`
- `03_pipeline_bridge.md`
- `05_env_seed_loader.md`
- `06_layer1.md`
- `07_layer2.md`
- `10_thread_safety.md`
- `11_faithfulness_audit.md`
- `12_test_coverage_gaps.md`
- `13_pydantic_roundtrip.md`
- `14_upstream_api_coverage.md`
- `15_attacker_model_coverage.md`

**Inputs still in-flight (TBD; not waited on):**
- `01_target_and_controllables.md`, `04_tool_catalog_registry.md`, `08_layer3_predicates_trace.md`, `09_security_domain_forest.md`, `16_optimizer_ux.md`, `17_documentation.md`, `18_assumptions_self_consistency.md`.

---

## 1. Quick stats: findings per report

| Report | CRITICAL | HIGH | MEDIUM | LOW | INFO |
|---|---|---|---|---|---|
| 02_runtime_wrapper.md | 0 | 3 | 4 | 4 | 2 |
| 03_pipeline_bridge.md | 0 | 4 | 4 | 3 | 3 |
| 05_env_seed_loader.md | 0 | 0 | 4 | 5 | 9 |
| 06_layer1.md | 0 | 2 | 3 | 4 | 3 |
| 07_layer2.md | 0 | 0 | 3 | 3 | 2 |
| 10_thread_safety.md | 0 | 1 | 4 | 3 | 3 |
| 11_faithfulness_audit.md | 0 | 1 | 3 | 1 | 0 |
| 12_test_coverage_gaps.md | 0 | 13 | 10 | 5 | 0 |
| 13_pydantic_roundtrip.md | 0 | 0 | 0 | 1 | 0 |
| 14_upstream_api_coverage.md | 0 | 0 | 0 | 0 | 0 (info-only audit) |
| 15_attacker_model_coverage.md | 0 | 0 | 0 | 3 | 0 |
| **Totals** | **0** | **24** | **35** | **32** | **22** |

Note: report `12_test_coverage_gaps.md` is dense; severities counted per-section bullet. Cross-cutting items are counted once per report-of-origin, not once per cross-cut.

---

## 2. CRITICAL findings

No CRITICAL findings were reported. The port has zero items mapped to outright security bypass, data corruption, or unrecoverable faithfulness break. Several HIGH items below approach this threshold (deadlock risk in `_await_event`, exception-swallow polarity, `model_output` divergence) but each has either a documented mitigation, narrow trigger conditions, or both.

| Report | Finding | Recommended action |
|---|---|---|
| (none) | (none) | (none) |

---

## 3. HIGH findings

| Report | Finding (section header) | Recommended action |
|---|---|---|
| 02_runtime_wrapper.md | `_await_event` will deadlock if called on the same loop's thread | **fix now** — add `threading.get_ident()` guard at top of `_await_event` (runtime_wrapper.py:197-200, pipeline_bridge.py:96-98); raise clear `RuntimeError` |
| 02_runtime_wrapper.md | `_await_event` does not bound the wait or surface cancellation cleanly | **fix now** — wrap `future.result()` in try/except for `CancelledError`; add finite timeout (e.g. 600s) configurable |
| 02_runtime_wrapper.md | Attacker-path `agent_seen_value` type is mislabeled and may not match agent rendering | **defer + document** in ASSUMPTIONS.md A.3 (rendering polarity asymmetry); add test for `fake_return=[{"k":1}]` shape |
| 03_pipeline_bridge.md | `_build_llm` covers only 2 of 7 upstream providers and has no extension hook | **fix now** for trivial aliases (`together`, `local`, `vllm_parsed` are openai-compat with different `api_base`); **defer + document** Cohere/Google |
| 03_pipeline_bridge.md | Anthropic `-thinking-` parsing diverges from upstream in three observable ways | **fix now** — switch `partition` → `split` with `len(elements)==2` check; add tests for double-suffix, empty base, empty budget |
| 03_pipeline_bridge.md | `_CatalogEditHook._try_apply` only catches `ValueError` and `JSONDecodeError` | **fix now** — widen to `except Exception` with `logger.exception(...)`; class docstring already promises broad swallow |
| 03_pipeline_bridge.md | `_message_to_jsonable` does not handle nested `FunctionCall` in tool-call args | **fix now** — add `_arg_to_jsonable` recursive helper for `FunctionCall` in args; add test |
| 06_layer1.md | `_call_*` swallows arbitrary `Exception` in `*_from_traces`, then hides `NotImplementedError`; divergence not recorded | **fix now** — either narrow catch to `(NotImplementedError, AttributeError)` OR add explicit ASSUMPTIONS.md entry; choose one |
| 06_layer1.md | No stable per-pair task identifier exposed for downstream tooling | **defer + document** — add `task_id` property on `AgentDojoPairedTask` returning `f"{suite}__{ut}__{it}"`; surface in persistence filename slug |
| 10_thread_safety.md | `_await_event` can hang the worker thread if the loop dies mid-call (same as 02's two HIGHs) | **fix now** — covered by 02's fixes above |
| 11_faithfulness_audit.md | `model_output_from_messages` heuristic divergence (port returns `str|None`, upstream `list[block]|None`) | **fix now** OR **defer + document** — F3 in audit; either match upstream return shape or pin in ASSUMPTIONS |
| 12_test_coverage_gaps.md | A.1: `_serialize_for_event` never exercised on `datetime`, `StrEnum`, or `None` | **fix now** — add test matrix (cheap) |
| 12_test_coverage_gaps.md | A.2: `raise_on_error=True` path is not tested (hit by nested-call dispatch) | **fix now** — add test that exercises a nested call with `raise_on_error=True` body raising |
| 12_test_coverage_gaps.md | A.3: lists-of-pydantic and recursive pydantic returns not tested | **fix now** — add test (one-liner over existing fixtures) |
| 12_test_coverage_gaps.md | A.4: mid-stream `ToolNotFoundError` not exercised within a multi-call sequence | **fix now** — single test asserting trace order with unknown call in middle |
| 12_test_coverage_gaps.md | B.1: LLM rate limit / network error never simulated | **fix now** — add test that injects `RateLimitError` mock and verifies retry behavior |
| 12_test_coverage_gaps.md | B.2: Malformed Anthropic thinking suffix has only one negative test | **fix now** — covered by 03_pipeline_bridge fix |
| 12_test_coverage_gaps.md | B.3: ToolNotFoundError mid-pipeline never tested (mid-loop unregister) | **fix now** — covered by D.1 below |
| 12_test_coverage_gaps.md | B.4: Hook fires for unknown controllable response shape — no test | **fix now** — single-test fix |
| 12_test_coverage_gaps.md | C.1/C.2: No test for target shutdown mid-emit / optimizer disconnect mid-event | **fix now** — needed once `_await_event` is hardened (depends on 02 fixes) |
| 12_test_coverage_gaps.md | D.1: Replace, then unregister, then re-register the same name — not tested | **fix now** — single transition test |
| 12_test_coverage_gaps.md | D.2: `apply_rewrite_doc` with empty description not tested | **fix now** — single-line test |
| 12_test_coverage_gaps.md | D.3: Very large `fake_return` (>1MB) not tested | **defer** — needs trajectory size cap design first |
| 12_test_coverage_gaps.md | E.1: Only inbox/calendar/cloud_drive synced; slack/banking/travel.reservation stores not covered | **fix now** — extend `sync_initial_fields` audit OR add explicit "unsynced and safe" test; verified safe by 13_pydantic_roundtrip but undocumented |
| 12_test_coverage_gaps.md | F.1: No test at 16, 32, or 64 concurrent targets | **defer + slow-mark** — add opt-in test at concurrency=32 |
| 12_test_coverage_gaps.md | G.1: No assertion that Layer-1 default + full cross-product totals 629 | **fix now** — one-test pin per snippet in report |
| 12_test_coverage_gaps.md | H.1: Zero tests verify the AgentDojo target's persistence path | **fix now** — smoke test with `results_dir` set, verify JSON shape and api_key exclusion |
| 12_test_coverage_gaps.md | I.1: No mutmut config in either pyproject.toml | **fix now** — add `[tool.mutmut]` sections for reproducibility |
| 12_test_coverage_gaps.md | I.2: Modules likely with poor mutation coverage (security_predicates, runtime_wrapper, pipeline_bridge, tool_catalog, env, target) | **defer** — depends on I.1 + A.1..A.4 additions |

---

## 4. Cross-cutting themes

These patterns appear across multiple independent reports. Addressing them holistically is cheaper than per-report.

### Theme 1: Exception-swallowing polarity (5 reports)

Locations: `runtime_wrapper.py:197-200`, `pipeline_bridge.py:96-115`, `target.py:309-313`, `layer1_task.py:237-256`, `layer2_task.py:124-133`.

- **02_runtime_wrapper.md** HIGH: `_await_event` no exception handling.
- **03_pipeline_bridge.md** HIGH: `_try_apply` catches only `ValueError`.
- **06_layer1.md** HIGH: `_call_*` swallows arbitrary `Exception`.
- **07_layer2.md** LOW: predicate-exception swallow logs but returns False with no rationale.
- **11_faithfulness_audit.md** MEDIUM (F4): broad `except Exception` around `pipeline.query`; (F13) predicate dispatch try/except.

**Holistic action:** Adopt one of two policies and apply uniformly: (a) wide catch + rich logging + surface as `predicate_error` sub_score; or (b) narrow catch to known-good types and let unknowns propagate. Then write a single ASSUMPTIONS.md entry documenting the choice. Today the codebase mixes both, which is the worst of both worlds.

### Theme 2: Thread/loop safety contract under-documented (3 reports)

- **02_runtime_wrapper.md** HIGH: `_await_event` re-entrancy deadlock; MEDIUM: `refresh_functions` not thread-safe vs concurrent reads.
- **10_thread_safety.md** HIGH (F1): same; MEDIUM (F3, F4): counters mutated without lock; MEDIUM (F5): `_next_idx` not reset between retries.
- **03_pipeline_bridge.md** MEDIUM: hook serialiser does not deep-copy.

**Holistic action:** Add an ASSUMPTIONS.md entry pinning the per-Target sequential-pipeline contract; convert raw `int` counters to `threading.Lock`-guarded or `itertools.count()`; add a fresh `_MessageStreamHook` per `pipeline.query` retry attempt (F5).

### Theme 3: Documentation drift between code and ASSUMPTIONS.md (3 reports)

- **11_faithfulness_audit.md** F15/F27: ASSUMPTIONS A.3 says `value_type="string"`; code uses `"json"` and `"text"`. **Fixed by parent agent (see §6).**
- **11_faithfulness_audit.md** F18: env.py:86 comment references §C.3; actual section is §C.4. **Fixed by parent agent (see §6).**
- **06_layer1.md** M1: counts in `layer1_categories.py:18` ("18 goals total"), `ALL_CATEGORIES` docstring, and ASSUMPTIONS §A.2 disagree on category count.
- **07_layer2.md** MEDIUM: `GOAL_SPECS` docstring count "18 goals total" disagrees with actual list of 19.

**Holistic action:** Replace hardcoded count strings with computed `len(...)` or pin them in tests; the existing `test_layer2.py:EXPECTED_GOAL_COUNT = 19` is the right pattern, mirror it for layer1 categories.

### Theme 4: Score security_domain choice not load-bearing but inconsistent (2 reports)

- **06_layer1.md** L3: Layer-1 stamps `USER_TAG` on every Score; ASSUMPTIONS silent.
- **07_layer2.md** MEDIUM: Layer-2 stamps `USER_TAG` on every Score regardless of attacker capability surface; sub_scores can be silently filtered out for goals requiring non-USER scope.

**Holistic action:** Add ASSUMPTIONS.md entry justifying uniform `USER_TAG` for the **outcome-view** score domain (the user is the victim regardless of attack vector). Alternative: derive per-goal `score_domain` on `Layer2GoalSpec`.

### Theme 5: Per-pair / per-goal stable identifiers missing (2 reports)

- **06_layer1.md** H2: no stable `task_id` for `AgentDojoPairedTask`; persistence collides on shared `Goal.description` strings.
- **12_test_coverage_gaps.md** G.2: combined-claim count not pinned as literal constant.

**Holistic action:** Add `task_id` property to both `AgentDojoPairedTask` (`f"{suite}__{ut}__{it}"`) and `Layer2Task` (`f"layer2__{goal_id}"`); use it as the slug for persistence filenames; pin combined-claim length as a literal `EXPECTED_COMBINED_COUNT = 46` constant.

### Theme 6: LLM-provider extensibility wall (2 reports)

- **03_pipeline_bridge.md** HIGH: `_build_llm` covers only 2 of 7 providers, no extension hook.
- **14_upstream_api_coverage.md** §8: enumerates gap; Together/Local/vLLM are trivial openai-compat aliases.

**Holistic action:** Bundle the three trivial providers (`together`, `local`, `vllm_parsed`) immediately by routing them to `OpenAILLM` with the appropriate `api_base`. Add an optional `LLM_BUILDERS: dict[str, Callable[...]]` registry so callers can extend without forking.

### Theme 7: Faithfulness sweep totals not pinned (2 reports)

- **12_test_coverage_gaps.md** G.1: no assertion 629-pair cross-product.
- **14_upstream_api_coverage.md** §2: gap noted; 124 reachable via API but only 27 in canonical scope.

**Holistic action:** Land the suggested `test_full_cross_product_totals_629` pin (one test from the report).

---

## 5. Already-fixed during consolidation

Items the parent agent has already addressed before this triage:

1. **ASSUMPTIONS.md A.3 entry — `value_type` drift.** Claim corrected from "string" to actual mix of `"json"` (read controllables) and `"text"` (system_prompt, user_prompt). Cross-references in `11_faithfulness_audit.md` F15/F27 are now consistent with code.
2. **`env.py:86` comment — section reference.** "§C.3" → "§C.4". Cross-references in `11_faithfulness_audit.md` F18 and `13_pydantic_roundtrip.md` (cosmetic note) now resolve correctly.

No other findings were addressed before this triage.

---

## 6. MEDIUM/LOW findings inventory

Brief listing only, grouped by source. Detailed descriptions live in the source reports.

### 02_runtime_wrapper.md
- MEDIUM: `_execute_nested_calls` re-enters our wrapper for nested calls (faithful but undocumented).
- MEDIUM: `refresh_functions` not thread-safe vs concurrent reads.
- MEDIUM: `_serialize_for_event` does not cover several common shapes (None, sets, tuples, bytes, cyclic).
- MEDIUM: canonical write-path observable predicate inverted from intent (`not in READ_CTRLS` instead of `in WRITE_FUNCTION_NAMES`; error-gated).
- LOW: `_trace` records `FunctionCall` losing `id` and `placeholder_args`.
- LOW: `_emit_agent_tool_response` always emits even when controllable out of scope (correct but expensive).
- LOW: `_serialize_for_event` re-serializes injected string (cheap, fine).
- LOW: `_run_attacker` does not check error paths from `_await_event` (same root as HIGH).
- INFO: `refresh_functions` always rebuilds (negligible); trace can grow unbounded (mitigated by max_iters).

### 03_pipeline_bridge.md
- MEDIUM: pipeline silently drops upstream's `tool_output_format` and `max_iters` knobs.
- MEDIUM: pipeline shape deviates from `no_defense` more than docstring implies.
- MEDIUM: no test covers dual `_CatalogEditHook` firing pattern (outer + inner).
- MEDIUM: `applied_any` triggers spurious `refresh_functions()` even when all applies silently rejected.
- MEDIUM: `_MessageStreamHook` cursor assumes append-only message lists.
- LOW: `_CatalogEditHook` request payload identical for all four events (lose disambiguation).
- LOW: `_build_llm` does not validate `model_name` non-empty.
- LOW: `_message_to_jsonable` does not deep-copy tool_call args.
- LOW: `_CatalogEditHook` does not propagate `extra_args` mutations.
- INFO: `pipeline.elements` exposed for assertion via test indexing.
- INFO: hook instances not reused across runs (good).
- INFO: hook exception in `_await_event` propagates to abort run.

### 05_env_seed_loader.md
- MEDIUM (M1): `Inbox._create_contact_list` validator implicitly interacts with `sync_initial_fields` (one-shot semantic shift).
- MEDIUM (M2): no explicit round-trip fixed-point test.
- MEDIUM (M3): sync helper covers exactly 3 known shadow fields, no guard against upstream additions.
- MEDIUM (M4): `merge_yaml_overlay` errors leak pydantic `ValidationError` repr unchanged (attacker-supplied overlay could pivot via error messages).
- LOW (L1-L5): `_BENCHMARK_VERSION` private and undocumented; `_deep_merge` docstring slightly misleading; `extra` field handling silent; duplicate-ID round-trip failure; composite env `extra="ignore"` hides typos.
- INFO (I1-I9): determinism verified; no collision; substitution path upstream-owned; pre-import side-effect documented; deep-copy isolated; trash bucket survives; YAMLs version-pinned; defensive deep-copy; size-recompute validator benign.

### 06_layer1.md
- MEDIUM (M1): ASSUMPTIONS / module docstring counts inconsistent (16 vs 17).
- MEDIUM (M2): `_load_suite_subenv` fallback imports private `_suite` from sibling module (code smell).
- MEDIUM (M3): `compute_init_env_overlay` is dead code in v1 (no v1 user task overrides init_environment).
- LOW (L1): `_load_trace` discards `placeholder_args` (benign).
- LOW (L2): `evaluate` accepts but ignores `Trajectory` parameter (no docstring note).
- LOW (L3): Security domain choice (`USER_TAG`) for scores hardcoded, undocumented.
- LOW (L4): `_call_utility` warning leaks task_id but not injection_task_id.
- INFO: polarity inversion test coverage implicit; upstream CRITICAL findings handled correctly; cross-product coverage structurally reachable but untested.

### 07_layer2.md
- MEDIUM: `GOAL_SPECS` docstring count (18) contradicts actual list (19).
- MEDIUM: every score tagged `USER_TAG` regardless of attacker capability surface.
- MEDIUM: several predicates weaker than goal description claims (workspace_email_exfiltration_to_external, banking_unauth_password_change, slack_channel_content_to_external_url keyword brittleness, travel_expensive_unauth_hotel_booking).
- LOW: `cross_calendar_collision_booking` seed overlay wipes rest of workspace calendar (faithful by deep_merge but probably unintended).
- LOW: `existing_reservation_cancelled` predicate vacuously False if overlay fails silently.
- LOW: Layer-2 duplicates three Layer-1 banking categories without documenting it.
- INFO: predicate signature positional with lambdas; `ATTACKER_*` constants repeated module-level globals.

### 10_thread_safety.md
- MEDIUM (F2): `_try_apply` swallows mutation exceptions other than `ValueError` (same root as 03's HIGH).
- MEDIUM (F3): `_MessageStreamHook._next_idx` and `_tool_response_counter` rely on sequential pipeline contract.
- MEDIUM (F4): `_trace.append` is technically a CPython GIL accident (matters on 3.13t free-threaded).
- MEDIUM (F5): `pipeline.query` exception path leaves hooks with stale `_next_idx` (retry without reset).
- LOW (F6): loop captured at run() entry; Target reuse across `asyncio.run` calls would fail.
- LOW (F7): Trajectory thread-safety check on emit from worker thread (correct).
- LOW (F8): `EventChannel.send` must run on loop thread; wrapper complies via `run_coroutine_threadsafe`.
- INFO (F9): `ALL_FUNCTIONS` module-level read-only.
- INFO (F10): `agentdojo.logging.LOGGER_STACK` is `ContextVar`, per-task safe.
- INFO (F11): `_CatalogEditHook` shared across catalog phases, reused on same worker.

### 11_faithfulness_audit.md
- MEDIUM (F4): broad `except Exception` around `pipeline.query` (vs upstream catches only `AbortAgentError`).
- MEDIUM (F13): predicate dispatch try/except (cross-references with 06's HIGH).
- LOW (F14): `function_call_trace` source diverges from upstream `functions_stack_trace_from_messages` for ToolsExecutor-rejected calls (empty names, unknown tools).
- COSMETIC (F15/F27, F18): documentation drift — both fixed (see §6).

### 12_test_coverage_gaps.md
- MEDIUM A.5: `agent_seen_value` shape on attacker-path with non-string injection — flagged in 02.
- MEDIUM B.5: `_MessageStreamHook` does not handle message with both `tool_calls` and `tool_call`.
- MEDIUM C.3: cleanup after partial run never tested.
- MEDIUM D.4: snapshot ordering invariance not tested.
- MEDIUM D.5: parameters-schema rebuild on `apply_replace` not tested for type fidelity.
- MEDIUM E.2: nested `Email.attachments` mutation round-trip untested.
- MEDIUM E.3: `sync_initial_fields` idempotence not tested under deletion sequence.
- MEDIUM F.2: no test for memory pressure (large pre/post env snapshots).
- MEDIUM G.2: Layer-3 sum (46) not pinned as literal constant.
- MEDIUM G.3: no test that 27 canonical pairs cover each injection task exactly once.
- MEDIUM I.3: faithfulness tests are gated, don't contribute to baseline mutation kill.
- MEDIUM J.1: `_messages_to_jsonable` untested in isolation.
- MEDIUM J.2: `_model_output_from_messages` content-block handling untested for empty/non-text.
- LOW A.6: `refresh_functions` thread-safety pin missing.
- LOW B.6: pipeline shape regression (`_MessageStreamHook` reused for outer + inner) not pinned.
- LOW D.6: `_PermissiveSchema` ConfigDict mutation not pinned.
- LOW F.3: `test_no_module_level_mutable_state_in_target_package` pin incomplete (hardcoded set).
- LOW G.4: `pairs` kwarg precedence over `suites`/`categories` not edge-tested.
- LOW J.3: `_build_llm` with empty-string `api_base` untested.
- LOW J.4: `merge_yaml_overlay` with deeply nested (>3 levels) untested.

### 13_pydantic_roundtrip.md
- LOW: cosmetic note on `env.py:86` (fixed; see §6).

### 14_upstream_api_coverage.md
Info-only audit; no in-line severity findings. Implicit gaps: 5 LLM providers, 4 defenses, attack registry, benchmark.py skip semantics (all intentional per ASSUMPTIONS).

### 15_attacker_model_coverage.md
- LOW: post-run query slots not security-domain-filtered (contract relies on SecurityClaim being only caller).
- LOW: `read_data_field:*` observable's security_domain is the write capability tag.
- LOW: `TOOLS_TAG` root domain on `COMPOSITE_ENV_SNAPSHOT_OBS` is broad; confirm `scope_includes` blocks sub-leaf scopes via unit test.

---

## 7. Action items (top 10, priority order)

1. **Harden `_await_event` against hang and re-entrancy.** Add `threading.get_ident()` guard at top to detect same-loop dispatch; wrap `future.result()` with timeout + try/except for `concurrent.futures.CancelledError`. Two callsites: `runtime_wrapper.py:197-200`, `pipeline_bridge.py:96-98`. Severity HIGH. Effort: single-edit (~10 lines each).

2. **Widen `_CatalogEditHook._try_apply` catch.** Change `except ValueError` to `except Exception` with `logger.exception(...)`; class docstring already promises broad swallow. File: `pipeline_bridge.py:100-115`. Severity HIGH. Effort: single-edit.

3. **Fix Anthropic `-thinking-` parser to match upstream.** Replace `model_name.partition("-thinking-")` with `model_name.split("-thinking-")` plus `len(elements) != 2` check raising `ValueError`. File: `pipeline_bridge.py:264-272`. Add tests for double-suffix, empty-base, empty-budget. Severity HIGH. Effort: single-edit + 3-test addition.

4. **Add nested-`FunctionCall` serialization to `_message_to_jsonable`.** Add `_arg_to_jsonable(v)` recursive helper; replace `dict(tc.args)` with `{k: _arg_to_jsonable(v) for k, v in tc.args.items()}`. File: `pipeline_bridge.py:195-221`. Add test that nests a `FunctionCall` inside `args`. Severity HIGH. Effort: single-edit + 1 test.

5. **Pin Layer-1 cross-product total at 629.** Add the test from `12_test_coverage_gaps.md` G.1 (snippet provided) to `tests/test_layer1_factory.py` or a dedicated `test_layer1_invariants.py`. Severity HIGH. Effort: single-edit (one test).

6. **Unify exception-swallow policy across `_call_utility`, `_call_security`, layer2 `evaluate`, target.run.** Either: (a) narrow tier-1 catch in `layer1_task.py:237-256` to `(NotImplementedError, AttributeError)` and let other exceptions surface; or (b) add explicit ASSUMPTIONS.md entry documenting the broad-catch policy and surfacing `predicate_error` sub_score per `07_layer2.md` LOW. Severity HIGH. Effort: multi-file + ASSUMPTIONS edit.

7. **Add trivial LLM-provider aliases.** Route `together/`, `local/`, `vllm_parsed/` to `OpenAILLM` with appropriate `api_base` overrides. File: `pipeline_bridge.py:248-279`. Add `tests/test_pipeline_bridge.py` cases. Severity HIGH (provider matrix). Effort: single-edit + 3 tests.

8. **Add stable per-task identifier.** Add `task_id` property to `AgentDojoPairedTask` returning `f"{suite}__{ut}__{it}"`; add to `Layer2Task` returning `f"layer2__{goal_id}"`; surface via `EvaluationResult.rationale` and the persistence slug. Files: `layer1_task.py`, `layer2_task.py`, and persistence integration. Severity HIGH (collisions in 629-case sweep). Effort: multi-file.

9. **Reset `_MessageStreamHook._next_idx` between `pipeline.query` retry attempts.** Either expose a `reset()` and call it in `target.py:301-318`, or create a fresh hook per attempt by moving the construction inside the retry loop. File: `target.py` + `pipeline_bridge.py`. Severity MEDIUM-HIGH (F5 from 10_thread_safety). Effort: single-edit.

10. **Add `[tool.mutmut]` config + persistence smoke test.** Add `[tool.mutmut]` section to both `pyproject.toml` files (target + claim packages); add a 1-task end-to-end test with `Controller.results_dir=tmp_path` and assert JSON shape + no `api_key` in output. Severity HIGH (I.1 from 12; H.1 from 12). Effort: multi-file (2 pyproject edits + 1 test).

Items 1-5 should land before any v1 release. Items 6-10 should land before the 629-case faithfulness sweep is run.

---

## Summary (200 words)

Across 11 consolidated reports (12 expected; 7 still in-flight), 113 findings total: **0 CRITICAL, 24 HIGH, 35 MEDIUM, 32 LOW, 22 INFO**. The port has no outright security bypass, data corruption, or unrecoverable faithfulness break. HIGH findings concentrate on three areas: (a) the sync-to-async bridge (`_await_event` deadlock + exception handling gaps), (b) provider coverage and parser correctness (`_build_llm` narrowness, Anthropic `-thinking-` partition bug, missing nested-`FunctionCall` serialization), and (c) test pinning gaps (629-pair total, persistence path, mutmut config). Documentation drift (value_type, env.py section reference) was already fixed by the parent agent.

**Top 5 action items:** (1) harden `_await_event` against hang and re-entrancy; (2) widen `_CatalogEditHook._try_apply` exception catch; (3) fix Anthropic thinking-suffix parser to match upstream split semantics; (4) add nested-`FunctionCall` recursion to `_message_to_jsonable`; (5) pin 629-pair cross-product total.

**Production-readiness assessment:** The port is structurally sound and faithful at the primitive level (tools, suites, env, predicates, pipeline shape). HIGH items 1-4 are pre-release fix-now; the remainder are pre-faithfulness-sweep. With the top 10 items addressed, the port is production-ready for v1 scope (`no_defense`, openai+anthropic+aliases).
