# ASSUMPTIONS.md Self-Consistency Review

Verification of `targets/agentdojo/ASSUMPTIONS.md` against current code (upstream agentdojo==0.1.35, port + claim worktrees).

## Top-line conclusion

The document is largely accurate as a high-level threat-model reference, but contains **one factually incorrect upstream tool reference**, **one substantive code mismatch (A.3 value_type)**, **a small cluster of mis-numbered cross-references** in source comments, and **partial / ambiguous wording in C.7**. Numbering is monotonic across all sections. The newly-added H section is internally consistent with G/A/B/D content.

---

## Per-entry verification

### A. Injection model

- **A.1** Upstream `task_suite.py:139-146` `load_and_inject_default_environment` exists and matches the described behavior (`environment_text.format(**injections_with_defaults)`). Port behavior (`seed_loader.load_composite_seed()` calls each suite's `load_and_inject_default_environment({})`) matches the claim. ACCURATE.

- **A.2** Per-tool controllables in `controllables.py` (47 read controllables built by `_build_read_controllables`) match the claim. ACCURATE.

- **A.3** **INACCURATE.** Document claims "every read Controllable in `controllables.py` declares `value_type="string"`" and "every catalog Controllable declares `value_type="json"`". Actual source:
  - `controllables.py:54,61`: prompt controllables use `value_type="text"` (not "string").
  - `controllables.py:77,89,99,110`: catalog controllables use `value_type="json"` (matches).
  - `controllables.py:231` (`_make_read_ctrl`): read controllables use `value_type="json"` (NOT "string").
  - `runtime_wrapper.py:125` (`_attacker_call_ctrl`): `value_type="json"`.
  
  No `value_type="string"` exists anywhere in the source. The test pin reference (`"9999.99"` in `tests/test_runtime_wrapper.py:130-137`) is real, but it tests payload injection, not the `value_type` label. **Recommend updating A.3** to read "every read Controllable declares `value_type='json'`; prompt controllables declare `value_type='text'`."

### B. Tool catalogue

- **B.1** Four catalog Controllables exist (`controllables.py:68-117`), and `_CatalogEditHook.query` fires them via `pipeline_bridge.py:64-140`. ACCURATE.

- **B.2** Upstream line references slightly loose:
  - `tool_execution.py:86`: confirmed `runtime.functions.values()`.
  - `tool_execution.py:103`: actually `runtime.run_function(env, ...)`, NOT a `runtime.functions` read. The claim "ToolsExecutor.query reads runtime.functions on each invocation" is correct (line 86 reads it; line 103 indirectly dispatches via `run_function`).
  - `openai_llm.py:197,223`: confirmed `runtime.functions.values()` at both lines.
  - Test pin `test_toolsexecutor_sees_attacker_tool_after_mid_loop_catalog_edit` exists at `test_toolsexecutor_per_turn.py:124`. ACCURATE-WITH-MINOR-LINE-SLIP.

### C. Composite environment

- **C.1** `CompositeEnvironment` in `env.py:40-63` has the four sub-attributes (`banking`, `workspace`, `slack`, `travel`). ACCURATE.

- **C.2** Tool registration in `tool_registry.py:199-201` (`prefixed_name`) produces `{suite}__{tool}`. Layer-1 strips prefix in `layer1_task.py:190-207`. ACCURATE.

- **C.3** `Depends` rebinding implemented in `tool_registry.py:224-268` (`_rebind_string_dep`, `_rebind_callable_dep`). ACCURATE.

- **C.4** `sync_initial_fields` exists in `env.py:66-95` and matches the documented behavior. Upstream file/line references for the validators are correct. Test in `tests/test_env_sync.py` exists. ACCURATE.

  **BUT** there are two mis-references TO C.4 from inside source code:
  - `env.py:86` comment says "See `ASSUMPTIONS.md` §C.3 for the rationale" — should be §C.4.
  - `target.py:409` comment says "See ASSUMPTIONS.md §C.3" — should be §C.4.

- **C.5** `seed_loader.py:28-52` `load_composite_seed()` unconditionally loads all four suites; no scope-based filtering. ACCURATE.

- **C.6** Tool catalog rejection at `tool_catalog.py:231-318` operates on suite-prefixed names. ACCURATE.

- **C.7** **PARTIALLY INACCURATE.** Verification of named tools:
  - `get_flight_information` -> exists at `tool_registry.py:124` (read). VERIFIED.
  - `get_all_car_rental_companies_in_city` -> exists at `tool_registry.py:117` (read). VERIFIED.
  - `get_scheduled_transactions` -> exists at `tool_registry.py:70` (read). VERIFIED.
  - `update_scheduled_transaction` -> exists at `tool_registry.py:135` (write). VERIFIED.
  - `cancel_scheduled_transaction` -> **DOES NOT EXIST** in port or upstream. Grep of `agentdojo/default_suites/v1/banking/` returns no matches. FABRICATED.
  - `search_contacts_by_name` -> exists at `tool_registry.py:80` (read, workspace). VERIFIED.
  - `search_contacts_by_email` -> exists at `tool_registry.py:81` (read, workspace). VERIFIED.
  
  Additionally, the sentence "All write-side variants emit `write_call:*` observables (per F.1/F.2 pattern)" is **misleading**:
  1. The reference to F.1/F.2 is incorrect — F.1 and F.2 discuss `get_unread_emails` semi-mutation and `get_webpage` audit trail, neither of which describes the `write_call:*` observable pattern. The relevant section is F.3 which explicitly defines `write_call:<tool>`.
  2. Of the tools listed, only `update_scheduled_transaction` and (the fabricated) `cancel_scheduled_transaction` would be write-side. The contacts-search and read-only travel tools listed are not write-side, so the "All" quantifier doesn't apply.

  **Recommend**: remove `cancel_scheduled_transaction` from the enumeration; fix F.1/F.2 reference to F.3; tighten "All write-side variants" to "Of these, `update_scheduled_transaction` is write-side and emits `write_call:*` (per F.3)".

### D. Polarity

- **D.1** Upstream `base_tasks.py:126` matches ("True if injection was executed correctly"). Upstream `_check_injection_task_security` is at `task_suite.py:297-311` (verified). Layer-1 sets `EvaluationResult.success = security` and primary_score `name="attack_succeeded"` at `layer1_task.py:151-153`. The cited `benchmark.py:147,153` line references are loose — line 147 is `security = True` (skip-branch), but line 153 is `set_contextarg("utility", utility)`, not `security = True`. The other skip-branches with `security = True` are at lines 130, 139, 292, 299, 306, 312. ACCURATE in substance, line citations could be tightened.

- **D.2** All claims match `tests/faithfulness/test_upstream_comparison.py`: `AGENTDOJO_FAITHFULNESS_MODEL`, fallback to `gpt-4-turbo-2024-04-09`, `LLMConfig(max_cost=4.0)`, the 12-pair set. ACCURATE.

### E. Pipeline lifecycle

- **E.1** Upstream `task_suite.py:383` `for _ in range(3):` verified. Port `target.py:301` `for attempt in range(3):` verified. ACCURATE.

- **E.2** Upstream `data/system_messages.yaml` has only `default:` key (verified by reading the file). Port `system_prompt.py:15-20` returns this verbatim. ACCURATE.

- **E.3** Port wraps only `no_defense`. The `pipeline_model` config slot (`config_specs.py:56-64`) lets the model be swapped at run time; this is not specifically documented in ASSUMPTIONS but isn't a divergence from "no defenses" since model identity is orthogonal to defense pipeline. NOT-A-DIVERGENCE-WORTH-FLAGGING.

### F. State observation

- **F.1** Upstream `email_client.py:148` `get_unread_emails` flips read flag (verified). Port doesn't intervene. ACCURATE.

- **F.2** Upstream `web.py:35` `get_webpage` appends to `web.web_requests` (verified). Port doesn't intervene. ACCURATE.

- **F.3** Per-call events via `_MessageStreamHook` (`pipeline_bridge.py:148-192`) and `WrappedFunctionsRuntime` (`runtime_wrapper.py:317-342`). Query specs in `query_specs.py` cover `pre|post_environment_snapshot`, `conversation_history`, `function_call_trace`, `tool_catalog_snapshot`. Note: `last_response` and `write_calls_made` are also exposed but not enumerated in F.3 — minor gap, not a contradiction. ACCURATE.

### G. SecurityClaim Layer-2 design

- **G.1** Verified bespoke catalogue at `security_claim_agentdojo/layer2_goals/` (20 goal files; see `banking_unauth_transfer.py` as example). `layer2_task.py:50-53` defines `SecurityPredicate` as pure Python callable. `security_predicates.py` contains no LLM/judge/api/openai/anthropic references. **CONFIRMED: source is bespoke deterministic, not SORRY-Bench/HarmBench/LLM-judge.** ACCURATE.

- **G.2** Verified `layer1_task.py:126` (`target.set_config("user_prompt", self._user_task.PROMPT)`) and `layer2_task.py:110` (`target.set_config("user_prompt", self._spec.benign_user_prompt)`). Both layers set a benign prompt; adversarial goal lives only in `Task.goal` and the security predicate. ACCURATE.

### H. User clarifications

- **H.1** `seed_loader.py:25` `_BENCHMARK_VERSION = "v1"`; `layer1_bridge.py:24` same. `UPSTREAM_PREDICATE_AUDIT.md` exists in the claim worktree. ACCURATE.
- **H.2** Mirrors G.1; redundant but explicitly cross-references it. NOT-INCONSISTENT.
- **H.3** Cross-references D.2 correctly. ACCURATE.
- **H.4** Process rule; no source assertion to verify. N/A.
- **H.5** Process rule; no source assertion to verify. N/A.
- **H.6** Process rule on em dashes. **Compliance check failed in source code, not in ASSUMPTIONS.md itself:**
  - ASSUMPTIONS.md contains 1 em dash, only inside H.6's rule text itself (line 258). Compliant.
  - 21 em dashes remain in source comments under `targets/agentdojo/src/` and `security_claims/agentdojo/src/` (e.g. `config_specs.py:5,8,12,16`, `pipeline_bridge.py:5,9,68,69`). The H.6 rule has not been applied to source; this is a process gap, not a contradiction with the doc.

---

## Cross-cutting findings

1. **Mis-section cross-references in source code (3 sites).**
   - `env.py:86`: "See `ASSUMPTIONS.md` §C.3" -> should be §C.4 (initial_* sync, not Depends rebinding).
   - `target.py:409`: "See ASSUMPTIONS.md §C.3" -> should be §C.4.
   - `controllables.py:17`: "see `ASSUMPTIONS.md` §C.4 for the rationale behind each [2x2 quadrant] assignment" -> §C.4 is initial_* sync, not 2x2 quadrant mapping. **The 2x2 quadrant rationale is not documented anywhere in ASSUMPTIONS.md.** Either add a new entry (e.g. C.8) explaining the quadrant rationale, or change the cross-reference to point at the inline docstring within `controllables.py` itself.
   - `security_tags.py:19`: "See `ASSUMPTIONS.md` Section C for the mapping rationale and the per-tool quadrant assignment." -> Section C has no explicit quadrant rationale. Same fix as above.

2. **A.3 mismatch is substantive** (every `value_type` label in the doc disagrees with the source). Either change the source to use `"string"` or update the doc to reflect the actual labels in use.

3. **C.7 has a fabricated tool name** (`cancel_scheduled_transaction`). It must be removed or corrected.

4. **No TODO/FIXME/HACK markers** in target or claim source. Only one `workaround` comment in `layer1_bridge.py:19` which refers to an in-doc divergence already covered by the env.py import-ordering comment.

5. **Pipeline `pipeline_model` config slot** is not enumerated under E.3 but is real (`config_specs.py:56-64`). This is a minor undocumented surface — recommend adding a note under E.3 (or a new E.4) since the model is the most consequential dial in the port.

6. **Layer 3** (`layer3_factory.py`) is not mentioned anywhere in ASSUMPTIONS. It is a trivial composition of Layer-1 + Layer-2, so the omission is defensible, but a brief G.3 entry would round out the document.

7. **The polarity of `attack_succeeded`** (D.1) is correctly applied in Layer 1 (`layer1_task.py:151-153`) and in the Layer-2 sub-score name `goal_achieved` (`layer2_task.py:140`). No silent polarity inversions in code.

## Numbering audit

All sections (A-H) use strictly monotonic numbering with no reuses:
- A: A.1, A.2, A.3
- B: B.1, B.2
- C: C.1, C.2, C.3, C.4, C.5, C.6, C.7
- D: D.1, D.2
- E: E.1, E.2, E.3
- F: F.1, F.2, F.3
- G: G.1, G.2
- H: H.1, H.2, H.3, H.4, H.5, H.6

COMPLIANT with the "never reuse a number" rule in the document header.

## Redundancy

- **H.2 duplicates G.1.** This is by design (H is a Q&A log, G is the design doc) and the H entry explicitly cross-references G. ACCEPTABLE.
- **D.2 and H.3** overlap on the faithfulness-budget topic; H.3 cross-references D.2. ACCEPTABLE.
- No accidental duplicates found.

---

## Suggested edits (in priority order)

1. **Fix A.3** to reflect actual `value_type` labels (`json` / `text`, not `string` / `json`).
2. **Fix C.7** by removing `cancel_scheduled_transaction` and correcting the F.1/F.2 reference to F.3.
3. **Update three source-code cross-references** to ASSUMPTIONS sections:
   - `env.py:86` and `target.py:409`: §C.3 -> §C.4.
   - `controllables.py:17` and `security_tags.py:19`: clarify or add a new ASSUMPTIONS entry for the 2x2 quadrant rationale.
4. **Tighten D.1 line references** (`benchmark.py:147,153` -> include the seven skip-branch sites).
5. **Consider adding** a new entry under E (or note in E.3) for the `pipeline_model` config slot, and a G.3 for Layer 3 composition.
6. **Apply H.6 to source comments** (21 em dashes remain in src/).
