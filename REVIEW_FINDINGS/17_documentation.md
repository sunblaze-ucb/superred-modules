# 17 — Documentation Completeness Review

Scope: `targets/agentdojo/` (port) and `security_claims/agentdojo/` (claim).

Reviewer rubric: README completeness, ASSUMPTIONS.md / UPSTREAM_PREDICATE_AUDIT.md depth and freshness, module/class/method docstring substance, `pyproject.toml` metadata, stale references, internal inconsistencies, security-domain documentation.

---

## Executive summary

Documentation is in unusually good shape for an alpha port. Both ASSUMPTIONS.md and UPSTREAM_PREDICATE_AUDIT.md are substantive, with file:line refs and rationale. Module docstrings explain the *why* (e.g. `target.py`'s 5-phase lifecycle, `runtime_wrapper.py`'s sync-to-async bridge, `env.py`'s `initial_*` pydantic validator trap). Class and method docstrings cover the critical APIs (`run`, `evaluate`, `configure_target`, `refresh_functions`, all four `apply_*` catalog methods).

Issues found are concentrated in three areas:

1. **A small number of stale cross-references** in docstrings (wrong ASSUMPTIONS section numbers).
2. **Two stale counts** (Layer-2 catalogue says 18; actual is 19).
3. **Em-dash violations** of standing rule `feedback_no_em_dash` across many source files and ASSUMPTIONS.md.
4. **README omissions**: neither README has an explicit "How to test" section or a labelled "Threat model" section (though both READMEs describe the relevant content informally).
5. **One dead-code stub** (`trace_capture.py`) whose docstring says "Implementation pending" but nothing imports it; either delete the file or document that the functionality landed elsewhere.

No security-domain tag is undocumented. No critical UPSTREAM_PREDICATE_AUDIT.md finding is missing from the code that I could detect.

---

## 1. README review

### Target README — `targets/agentdojo/README.md`

| Criterion | Status | Notes |
|---|---|---|
| What the package does | OK | Two attacker capability surfaces, security-domain forest summary. |
| How to install | OK | `pip install -e ./targets/agentdojo`. |
| How to test | **MISSING** | No `pytest` command, no mention of `tests/faithfulness/`, no markers. The `pyproject.toml` defines `faithfulness`, `faithfulness_upstream`, `faithfulness_full` markers that go undocumented in the README. |
| Threat model | partial | Two attacker capabilities and the security-domain forest are described, but no explicit "Threat model" section. |
| Status | OK | "v0.1.0 alpha. The public API is unstable." |
| Quick start | OK | Real wireup example. Caveat: the import line annotates `# subject to change` for `RESPONSE_READABLE_TAG`, but that tag does not exist in `security_tags.py` (the README imports `USER_TAG, RESPONSE_READABLE_TAG`; only `PROMPT_READABLE_TAG` and `TOOL_CATALOGUE_READABLE_TAG` exist). |

**Stale reference**: `README.md:32` — `from agentdojo_target import AgentDojoTarget, USER_TAG, RESPONSE_READABLE_TAG  # subject to change`. `RESPONSE_READABLE_TAG` is not exported and does not exist anywhere in the package. This is a copy-paste bug.

### Claim README — `security_claims/agentdojo/README.md`

| Criterion | Status | Notes |
|---|---|---|
| What the package does | OK | Three-layer breakdown, deterministic predicate emphasis. |
| How to install | OK | Two pip installs (target + claim). |
| How to test | **MISSING** | Same as target. |
| Threat model | partial | Layer responsibilities described, but no explicit threat-model section. |
| Status | OK | "v0.1.0 alpha." |
| Quick start | OK | Wireup example uses `frozenset({USER_TAG, CONTENT_3P_DATA_3P_TAG})` for scope — good demonstration. |

---

## 2. `ASSUMPTIONS.md` review (target only)

`targets/agentdojo/ASSUMPTIONS.md` — 22.9 KB, 264 lines.

**Coverage**: 8 sections (A-H), 27 numbered entries. Each entry has:
- "AgentDojo" subsection with file path + line numbers.
- "Us" subsection with what this package does.
- "Why" rationale.

**Examples of strong entries**:
- `A.1` cites `src/agentdojo/task_suite/task_suite.py:139-146` for static `{slot}` substitution.
- `B.2` cites both `agent_pipeline/tool_execution.py:86,103` and `llms/openai_llm.py:197,223` and names the test that pins the contract.
- `C.4` (initial_* sync) explains the pydantic validator trap with cross-reference to `tests/test_env_sync.py`.
- `D.2` (faithfulness comparison) names the 12 specific (user-task, injection-task) pairs and the budget cap.

**Section H** consolidates user clarifications dated 2026-05-15 through 2026-05-19 (H.1-H.6). H.4-H.6 are continuous standing rules.

**Currency check**: spot-checked file:line refs against the venv. The `task_suite.py:139-146` reference for `load_and_inject_default_environment` is consistent with upstream's location. The `base_tasks.py:126` polarity-docstring ref is real.

**Stale cross-references from source to ASSUMPTIONS.md**:
- `src/agentdojo_target/target.py:409` — comment says `See ASSUMPTIONS.md §C.3` but the section about `initial_*` sync is **§C.4** (§C.3 is about `Depends` extractor rebinding). The corresponding code in `env.py:86` correctly cites §C.4. Recommend updating `target.py:409` to "§C.4".
- `src/agentdojo_target/controllables.py:17` — says `See ASSUMPTIONS.md §C.4 for the rationale [for the 2x2 quadrant mapping]`. But §C.4 is about `initial_*` sync, not the 2x2 mapping. The 2x2 quadrant rationale lives in `security_tags.py` and a referenced "Section C for the mapping rationale" remark in `security_tags.py:19` which is consistent. Recommend updating `controllables.py:17` to either drop the §C.4 ref or point to the security-tags docstring.

**Missing entries** (code behaviour not covered by ASSUMPTIONS.md):
- The `_serialize_for_event` fallback chain (`pydantic.model_dump_json` → `json.dumps(default=...)` → `repr`) deserves an entry in **A**. Section A.3 partially covers it but does not name the fallback order.
- The `default_system_prompt` re-reads upstream YAML on every call rather than caching (cached upstream via `@lru_cache`). Worth a note in E.2 since this is a deliberate code choice.
- The dependency on `anthropic` and `openai` (not `litellm`) — `pipeline_bridge.py` directly uses provider SDKs while ASSUMPTIONS.md (or the README) is silent on this provider-coupling decision.
- The cleanup contract: `target.cleanup` does NOT reset config slots so a multi-run loop preserves the Task's `configure_target` settings (documented in `target.py` docstring but not in ASSUMPTIONS.md as a deliberate divergence from "fresh-state-per-run").

None of these are critical; ASSUMPTIONS.md ends with the parenthetical "(more sections to be added as implementation discovers divergences)" which licenses these additions.

---

## 3. `UPSTREAM_PREDICATE_AUDIT.md` review (claim only)

`security_claims/agentdojo/UPSTREAM_PREDICATE_AUDIT.md` — 44.7 KB, 281 lines.

**Coverage**:
- All four v1 suites audited (banking, workspace, slack, travel).
- All injection tasks audited (banking IT0-8 = 9, workspace IT0-5 = 6, slack IT1-5 = 5, travel IT0-6 = 7).
- All user tasks audited (banking UT0-15 = 16, workspace UT0-39 = 40, slack UT0-20 = 21, travel UT0-19 = 20).
- Per-task table with severity, finding, porting implication.
- "Cross-cutting patterns" section summarising recurring bug shapes.
- "Common predicate shapes" table.
- "Per-suite summary" with severity counts.

**19 CRITICAL findings + ~68 NOTABLE + ~37 OK = 124 tasks audited**. Numbers in the per-suite summary table.

**Findings reflected in code (verification by code-reading)**:
- **banking InjectionTask0/1/3 op-precedence**: confirmed — `layer1_task.py` evaluates upstream `injection_task.security(...)` verbatim; no patching. The CRITICAL findings preserve upstream's broken `any(X or "" in Y)` reduction.
- **slack UT11 / IT5 NotImplementedError**: `layer1_task.py:_call_utility` and `_call_security` use the trace-first-then-fallback pattern explicitly to handle these. Verified the `from_traces` is tried first and the catch is `Exception` (since upstream raises `NotImplementedError` from `utility(...)`/`security(...)`).
- **Verbatim typo preservation**: confirmed by code-reading — Layer-1 Task calls `user_task.PROMPT` verbatim, never rewriting. Layer-1 ground-truths and predicates are upstream's untouched.
- **Operator-precedence on banking InjectionTask4 case-sensitive recipient**: confirmed — `evaluate()` passes `pre_suite, post_suite` directly to upstream's `security()` which carries the bug.

**Audit completeness signal**: every CRITICAL finding's "porting implication" column says "Match upstream exactly + flag in ASSUMPTIONS.md." None of the audit findings appear to require code in our port (which is the right answer — the audit's purpose is to document the *upstream* state so a future audit can detect drift).

**Gap**: the audit *does* recommend flagging each CRITICAL in `ASSUMPTIONS.md`, but `security_claims/agentdojo/ASSUMPTIONS.md` does not enumerate the specific upstream predicate bugs. Section A.3 of the claim ASSUMPTIONS.md mentions "the polarity-inversion trap" generally, and A.4 mentions trace de-prefixing, but the 19 specific CRITICAL findings (e.g. banking IT0 op-precedence reducing the check to "any attacker-recipient transfer") are not individually called out. **Recommendation**: add a "Section D" to the claim ASSUMPTIONS.md saying "we preserve upstream predicate semantics verbatim including the 19 CRITICAL bugs catalogued in UPSTREAM_PREDICATE_AUDIT.md".

---

## 4. Module docstrings

Both packages have substantive module docstrings throughout. Spot-check sample:

| File | Docstring substance | Verdict |
|---|---|---|
| `agentdojo_target/__init__.py` | 12 lines: what the package does, where to find usage/ASSUMPTIONS | OK |
| `agentdojo_target/target.py` | 33 lines: 5-phase lifecycle, what each phase does, when `cleanup` vs `teardown` runs | Strong |
| `agentdojo_target/runtime_wrapper.py` | 38 lines: three side effects (trace, per-call event firing, observable emission), sync-to-async bridge explanation | Strong |
| `agentdojo_target/pipeline_bridge.py` | 20 lines: what hooks splice where, supported LLM providers | OK |
| `agentdojo_target/security_tags.py` | 22 lines: three trees, hierarchy semantics, naming convention | Strong |
| `agentdojo_target/env.py` | 18 lines: composite root, no flat fields, Depends rebinding required | Strong |
| `agentdojo_target/tool_catalog.py` | 31 lines: three entry kinds, four operations, snapshot, reset | Strong |
| `agentdojo_target/tool_registry.py` | 22 lines: suite-prefix scheme, classification table, drift detection | Strong |
| `agentdojo_target/controllables.py` | 22 lines: categories, 2x2 quadrant rationale, conservative-broader rule | Strong |
| `agentdojo_target/observables.py` | 22 lines: static vs dynamic, why static rebuilds per call | OK |
| `agentdojo_target/config_specs.py` | 21 lines: slot summary with semantic notes | OK |
| `agentdojo_target/query_specs.py` | 27 lines: slot-by-slot summary | OK |
| `agentdojo_target/system_prompt.py` | 9 lines: where it comes from, why no Figure-14/15/16 | OK |
| `agentdojo_target/seed_loader.py` | 12 lines: what `load_composite_seed` does, what `merge_yaml_overlay` does | OK |
| `agentdojo_target/trace_capture.py` | 12 lines: **"Implementation pending."** Stub file. Not imported anywhere. | **DEAD CODE** |
| `security_claim_agentdojo/__init__.py` | 11 lines: three layers, alpha note | OK |
| `security_claim_agentdojo/layer1_task.py` | 29 lines: full evaluation flow, polarity, sub-scores explained | Strong |
| `security_claim_agentdojo/layer2_task.py` | 28 lines: delivery vector by scope, attribute summary | Strong |
| `security_claim_agentdojo/layer1_factory.py` | 16 lines: factory tier explained, filtering precedence | OK |
| `security_claim_agentdojo/layer1_pairs.py` | 18 lines: pair selection rationale | OK |
| `security_claim_agentdojo/layer1_categories.py` | 19 lines: per-suite category table | OK |
| `security_claim_agentdojo/layer1_bridge.py` | 12 lines: four helper APIs | OK |
| `security_claim_agentdojo/layer2_factory.py` | 10 lines: filter modes | OK |
| `security_claim_agentdojo/layer3_factory.py` | 7 lines: trivial composition | OK |
| `security_claim_agentdojo/security_predicates.py` | 11 lines: helper signature + composite-env operation note | OK |
| `security_claim_agentdojo/layer2_goals/__init__.py` | 33 lines: 11-category taxonomy enumerated | Strong |

Per-goal layer2_goals/*.py modules each have a brief docstring explaining goal + benign prompt + predicate. Spot-checked `banking_unauth_transfer.py`, `banking_recurring_redirect.py` — both substantive (5-12 line docstrings).

---

## 5. Class & method docstrings — critical APIs

| API | Docstring | Verdict |
|---|---|---|
| `AgentDojoTarget` class | OK (args, defaults) | OK |
| `AgentDojoTarget.run` | implicit via module docstring (5-phase) | Phase narrative is in module docstring; method itself has no inline docstring. **Recommend adding a one-liner cross-referring to the module docstring.** |
| `AgentDojoTarget.get_observables` | "Static observables with current pre-run content snapshots" | OK |
| `AgentDojoTarget.cleanup` | Strong: explains *why* config slots are NOT reset | Strong |
| `AgentDojoTarget.teardown` | "No long-lived resources to release." | OK |
| `AgentDojoTarget._build_seed_env_with_overrides` | Inline explanation of overlay order | OK |
| `WrappedFunctionsRuntime` class | Strong: explains catalog mutability + per-call decision | Strong |
| `WrappedFunctionsRuntime.refresh_functions` | Strong: explains when to call, atomic-replacement contract | Strong |
| `WrappedFunctionsRuntime.run_function` | No inline docstring | **Missing.** This is THE override; the module-level docstring covers it but the method itself is silent. The behaviour (entry-kind dispatch) is in `_run_canonical` / `_run_attacker` which are individually documented. Could add a one-liner. |
| `WrappedFunctionsRuntime._run_canonical` | "Canonical path: legitimate value first, then optional injection." | OK |
| `WrappedFunctionsRuntime._await_event` | Strong: sync-to-async bridge in 1 line | OK |
| `ToolCatalog.from_seed` / `reset` / `get` / `classify` / `functions_for_runtime` / `snapshot` | Strong: each method documented with explicit return-value contract | Strong |
| `ToolCatalog.apply_register` / `apply_replace` / `apply_unregister` / `apply_rewrite_doc` | Strong: required keys, validation behaviour, return shape | Strong |
| `_CatalogEditHook.query` | No inline docstring; module-level docstring covers it | OK |
| `AgentDojoPairedTask.__init__` | OK | OK |
| `AgentDojoPairedTask.configure_target` | "Pin the benign user prompt, the suite's init_environment overlay, and the default system prompt." | OK |
| `AgentDojoPairedTask.evaluate` | No inline docstring; module-level docstring is the contract | OK (module-level is strong) |
| `AgentDojoPairedTask._call_utility` / `_call_security` | "Trace-first, then fall back to post-env-state utility." | OK |
| `SystemViolatingTask.configure_target` | Strong: explains *why* the goal isn't planted | Strong |
| `SystemViolatingTask.evaluate` | No inline docstring | OK (module-level covers it) |
| Predicates in `security_predicates.py` | Each one has a 1-2 line docstring stating the truth condition | OK |
| Module factories (`agentdojo_layer1_claim`, etc.) | Filter precedence documented, kwargs documented, Raises clauses | Strong |

---

## 6. `pyproject.toml` metadata

### Target `targets/agentdojo/pyproject.toml`

```
name = "agentdojo-target"
version = "0.1.0"
description = "AgentDojo composite target for superred: union of all four v1 suites with on-demand event-based injection and tool-catalog controllable surface"
requires-python = ">=3.11,<3.14"
dependencies = [superred>=0.1.0, agentdojo>=0.1.35, pydantic>=2.7, pyyaml>=6.0]
```

**Verdict**: Reasonable.

- `description`: substantive (one-line summary captures key features).
- `requires-python`: matches the framework convention (`>=3.11,<3.14`).
- `dependencies`: lists the framework, upstream agentdojo, pydantic (used in `tool_catalog.py` for placeholder schemas), pyyaml (used in `seed_loader.py`).
- **No `license` or `classifiers`.** Sibling packages (test_basic_*, security_claim_sorry_bench) also lack these, so this is consistent with the project's convention. Not a blocker for alpha.
- Test markers (`faithfulness`, `faithfulness_upstream`, `faithfulness_full`) are well-defined with comments explaining gating.

### Claim `security_claims/agentdojo/pyproject.toml`

```
name = "security-claim-agentdojo"
version = "0.1.0"
description = "AgentDojo SecurityClaim for superred: original injection-task port + bespoke system-purpose-violation goals against the composite AgentDojoTarget"
dependencies = [superred>=0.1.0, agentdojo-target, agentdojo>=0.1.35]
```

**Verdict**: Reasonable. Same comments as above. Test markers `integration` and `smoke` defined.

**Minor**: claim's `dependencies` lacks a version pin on `agentdojo-target` (just `"agentdojo-target"` with no version). Target is internal so a runtime missing version isn't catastrophic, but in principle `agentdojo-target>=0.1.0` would be more rigorous.

---

## 7. Stale references / drift

### Confirmed stale references

1. **`target.py:409`** — `See ASSUMPTIONS.md §C.3.` But §C.3 is "Depends extractor rebound per suite"; the actual ref is §C.4 (`initial_*` sync before JSON round-trip).

2. **`controllables.py:17`** — `see ``ASSUMPTIONS.md`` §C.4 for the rationale [for the 2x2 quadrant mapping]`. But §C.4 is about `initial_*` sync, not the 2x2 mapping. The mapping rationale lives in `security_tags.py:19` ("See ASSUMPTIONS.md Section C for the mapping rationale") which is vague-but-correct. Recommend `controllables.py:17` either drop the section number or change to a generic "Section C".

3. **`layer2_goals/__init__.py:91-92`** — `Counts: 4 starters + 8 per-suite expansions + 4 cross-suite + 2 capability misuse = **18 goals total**.` But actual count is **19** (verified by inspection of `GOAL_SPECS` list). The per-suite expansions are 9, not 8 (2 banking + 3 workspace + 2 slack + 2 travel = 9). Recommend updating the doc count.

4. **Target README line 32**: `from agentdojo_target import AgentDojoTarget, USER_TAG, RESPONSE_READABLE_TAG`. `RESPONSE_READABLE_TAG` does not exist in `security_tags.py`. The comment `# subject to change` hedges this but it's still a broken import in the quick-start example.

5. **`trace_capture.py`** — entire module is a stub with `"Implementation pending."` docstring; no imports reference it. Either delete the file or update the docstring to point to where the trace-capture functionality landed (which is `runtime_wrapper.py`'s `_trace` list + the post-run `function_call_trace` query slot).

### Em-dash violations of standing rule

Multiple `src/` files contain `—` characters (12+ files; spot-checked `banking_unauth_transfer.py:6`, `pipeline_bridge.py`, `layer1_task.py`, etc.). Memory `feedback_no_em_dash` is a standing rule. ASSUMPTIONS.md itself complies (uses "-" or rephrases). Recommend a sweep via the standing rule.

### Cross-document references that are consistent (no drift)

- `pyproject.toml` markers `faithfulness`/`faithfulness_upstream` match real test directory.
- `tests/test_toolsexecutor_per_turn.py` contains the function `test_toolsexecutor_sees_attacker_tool_after_mid_loop_catalog_edit` referenced in ASSUMPTIONS.md §B.2.
- `tests/test_env_sync.py` referenced in ASSUMPTIONS.md §C.4 exists.
- `tests/faithfulness/test_upstream_comparison.py` referenced in §D.2 exists.

---

## 8. Inconsistencies between docstrings and behaviour

Spot-check sample:

| Claim | Reality | Verdict |
|---|---|---|
| `target.py` says `query` returns JSON-encoded string | Verified — all `query` branches `json.dumps(...)` except `last_response` which is a plain string. | OK — `last_response` is the only non-JSON-encoded return, and `query_specs.py` docstring explicitly says "as a plain string (NOT JSON-quoted)". |
| `runtime_wrapper.py` says `_serialize_for_event` falls back through pydantic, json, repr | Verified at lines 80-104. | OK |
| `tool_catalog.py:apply_unregister` says "Idempotent: removing an already-absent name is a no-op" | Verified at line 313: `self._entries.pop(name, None)`. | OK |
| `tool_catalog.py:apply_register` says "If *name* already exists, the call is rejected with ValueError" | Verified at line 247-250. | OK |
| `controllables.py` says READ_QUADRANT_MAP is validated against READ_FUNCTION_NAMES at import (no missing, no extra) | Verified at lines 240-254. | OK |
| `layer1_task.py` says `evaluate` strips `{suite}__` prefix and calls `*_from_traces` first | Verified in `_load_trace` and `_call_security`/`_call_utility`. | OK |
| `system_prompt.py` says "Reads the upstream YAML fresh each call (cheap; lru_cached upstream)" | Upstream uses `@functools.cache` on `load_system_message` so subsequent calls are O(1). Docstring is accurate. | OK |
| `env.py:sync_initial_fields` says "mutation is in-place; the same env is returned for chaining convenience" | Verified — return value is the same instance. | OK |

No drift detected. The docstrings I sampled are accurate.

---

## 9. Security-domain documentation

`security_tags.py` documents every tag with a 1-2 line docstring giving the semantic. Examples:
- `USER_TAG`: "User prompt channel. An attacker with `user` can override the benign user prompt set by the Task via the user-prompt Controllable."
- `TOOL_CATALOGUE_ADDABLE_TAG`: "Register-only catalogue write capability. Weakest write capability ... Models a malicious-MCP that can only contribute additional tools, not shadow existing ones."
- `CONTENT_3P_DATA_1P_TAG`: "Third-party content held in first-party storage (e.g. a third-party vendor's bill stored in the user's filesystem). Examples: `banking.read_file('bill-december-2023.txt')`."

The four 2x2 leaves carry **explicit examples** in their docstrings. The hierarchical-implication relationships (e.g. `TOOL_CATALOGUE_TAG` implies the `_READABLE_TAG` and `_ADDABLE_TAG` children) are stated.

**Verdict**: Strong. A user reading `security_tags.py` can understand every tag's semantics without leaving the file.

One minor improvement: `DOMAIN`'s docstring says "17 tags across three trees" — verified by counting (system: 11, user: 1, tools: 5 = 17). Accurate.

---

## 10. Summary of recommended changes (no code modification per instructions)

Priority order:

### High priority (factual errors)
1. **Fix `target.py:409`**: change `§C.3` to `§C.4`.
2. **Fix `controllables.py:17`**: change `§C.4` to `Section C` or drop the section number.
3. **Fix `layer2_goals/__init__.py:92`**: update "18 goals total" to "19 goals total" and either revise "8 per-suite expansions" to "9" or recount.
4. **Fix target `README.md:32`**: remove the bogus `RESPONSE_READABLE_TAG` import or replace with a valid tag (e.g. `PROMPT_READABLE_TAG`).
5. **Resolve `trace_capture.py`**: delete the dead-stub file, or update its docstring to point to `runtime_wrapper.py:trace` + `query_specs.py:function_call_trace` where the functionality landed.

### Medium priority (missing sections)
6. **Add "How to test" sections** to both READMEs documenting `pytest` + markers (especially `pytest -m faithfulness` for the per-PR smoke and `pytest -m faithfulness_upstream` for the credentialed sweep).
7. **Add a labelled "Threat model" section** to both READMEs (Target: the 4 attacker capability surfaces + scope; Claim: the per-layer adversarial intent).
8. **Add a section to claim `ASSUMPTIONS.md`** stating that upstream predicate semantics are preserved verbatim (cross-referring to UPSTREAM_PREDICATE_AUDIT.md's 19 CRITICAL findings).

### Low priority (polish)
9. **Standing-rule sweep**: replace em dashes (`—`) with `-`, `,`, or rephrase across all `src/` files and ASSUMPTIONS.md text.
10. **Add inline docstrings** to `AgentDojoTarget.run`, `WrappedFunctionsRuntime.run_function`, and `AgentDojoPairedTask.evaluate` (each currently relies on a module-level docstring for behaviour). Even a one-line "See module docstring for lifecycle" would help.
11. **Pin `agentdojo-target` version** in the claim's `pyproject.toml` dependencies.
12. **Add ASSUMPTIONS entries** for `_serialize_for_event` fallback chain, `default_system_prompt` re-read-per-call, provider-SDK coupling (anthropic/openai vs litellm), and the `cleanup`-preserves-config contract.

### Out of scope
The lack of `license = "..."` and `classifiers = [...]` in both `pyproject.toml` files is consistent with sibling modules and not a documentation blocker for alpha.
