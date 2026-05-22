# Tool Catalog and Registry Review

## Summary

`tool_catalog.py` and `tool_registry.py` implement the per-run attacker-mutable tool catalog and its suite-prefixed seed. The registry build is solid: covers all 74 AgentDojo v1 tools, partitions read vs. write exhaustively (a `RegistryMismatchError` at import catches upstream drift in either direction), and rebinds every `Depends` extractor to navigate the composite environment (verified for every dep). The catalog's four mutation methods enforce sensible per-call invariants (no duplicate register, replace only on canonical, etc.). However: the mutation API is not transactional and not thread-safe (relies on an implicit invariant that all callers run on the pipeline worker thread), the attacker-supplied JSON-schema parser has fuzz-tolerant quirks (string `required` becomes per-character set; reserved pydantic names silently dropped), seed `Function` instances are aliased into catalog entries without defensive copies, and `apply_unregister` is idempotent for absent names. None are critical; several are documentation gaps the SecurityClaim layer should know about.

## Findings

### [HIGH] Catalog mutations have no locking; safe today only because of an implicit single-thread invariant

`tool_catalog.py:159-348`: `ToolCatalog` is a mutable dataclass with no locking. The `_entries` dict is mutated by `apply_register` / `apply_replace` / `apply_unregister` / `apply_rewrite_doc` / `reset` and read by `get` / `classify` / `functions_for_runtime` / `snapshot`. Today all mutations come from `_CatalogEditHook.query` (`pipeline_bridge.py:117-140`), which runs from a single pipeline worker thread; all reads (in `WrappedFunctionsRuntime.run_function` and `refresh_functions`) also come from the same worker thread; only `target.query("tool_catalog_snapshot")` reads from the event-loop thread, and that read happens *after* the run completes (`target.py:190-193`). So in the current architecture there is no concurrent access.

The implicit invariant is undocumented and brittle. If the Target ever runs internal branches concurrently (CLAUDE.md "Target supports internal parallelism (concurrent branches each calling `send_event` independently)") or a future change ever has the optimizer's `on_event` directly invoke a catalog method, two threads can race on `self._entries`. Worst case: `apply_replace` reads the canonical's `existing.function.parameters`, the canonical entry is concurrently rewritten to a `registered` entry whose parameters are `_PermissiveSchema`, and the resulting replaced entry holds a stale reference to the canonical's parameters class. No data corruption (Python dict is GIL-atomic at the bucket level) but a kind/parameters mismatch.

Recommended fix: add a `threading.RLock` field on `ToolCatalog` and acquire it inside each `apply_*` and inside the read-then-rebuild paths of `apply_replace` / `apply_rewrite_doc`. Alternatively, document the single-threaded contract in the class docstring and add an assertion in the `apply_*` methods (e.g. `_check_owner_thread()`) that fails loud on misuse.

### [MEDIUM] `_build_parameters_class` silently mis-parses string `required` as a per-character set

`tool_catalog.py:125`: `required = set(schema.get("required", []) or [])`. If the attacker supplies `"required": "abc"` (a string instead of a list), `set("abc")` yields `{"a", "b", "c"}`. Combined with the iteration over `properties`, this means any property whose name is a single character that appears in the string is marked required. A pathological payload `{"properties": {"a": {}, "b": {}, "n": {}}, "required": "ban"}` requires `a`, `b`, and `n` — none of which the attacker likely intended. The catalog accepts the payload without raising; the agent's downstream `model_validate` then rejects calls that omit those properties, breaking the attacker's own tool. Not a security boundary, but a fuzz-tolerant behaviour that hides bugs.

Recommended fix: validate `required` is a list before set conversion: `required_raw = schema.get("required") or []; if not isinstance(required_raw, list): raise ValueError(...)`. Or coerce: `required = set(required_raw) if isinstance(required_raw, list) else set()`. Add a test exercising both list and non-list shapes.

### [MEDIUM] `_build_parameters_class` silently drops pydantic-reserved property names

`tool_catalog.py:124-135`: when an attacker registers a tool with `"properties": {"model_dump": {}, "model_config": {}, "__class__": {}}`, pydantic's `create_model` silently ignores fields whose names collide with reserved attributes. The resulting class has `model_fields == {}` even though the attacker requested four fields. If those fields were in `required`, the required-field invariant is also silently dropped (the test `test_build_placeholder_function_honors_required_fields` would fail if the required field's name were reserved).

This is not a security issue (the attacker controls the payload, so dropping their own request just hurts them) but it can mask attacker errors and is undocumented. Worth a comment in `_build_parameters_class` saying that pydantic strips reserved field names, and (optionally) emitting a warning when this happens. Also worth testing: register with a `"properties": {"model_dump": {}}, "required": ["model_dump"]` payload and assert the resulting class accepts `{}` (since `model_dump` was stripped).

### [MEDIUM] Seed `Function` instances are aliased into catalog entries with no defensive copy

`tool_catalog.py:177-187`: `reset()` (and `from_seed` via `reset()`) builds `CatalogEntry(function=f, ...)` where `f` is the exact `Function` instance from `_seed`. The same `Function` is also in the registry's `ALL_FUNCTIONS` list. `Function` is a pydantic `BaseModel` *without* `model_config = ConfigDict(frozen=True)` — verified mutable: `f.description = 'pwned'` succeeds at runtime.

In normal flow no code mutates a `Function` in place: the apply_* methods always construct a new `Function`. But if any future code or test were to mutate `entry.function.description` directly (instead of via `apply_rewrite_doc`), the change would persist into `ALL_FUNCTIONS` and across all future `ToolCatalog.from_seed(ALL_FUNCTIONS)` constructions in the same process. The module-level `TOOL_REGISTRY` and `ALL_FUNCTIONS` are global, mutable state with no per-instance copy.

Recommended fix: either (a) make `Function`'s `model_config = ConfigDict(frozen=True)` (a one-line upstream PR, see `agentdojo.functions_runtime.Function` def) and accept that mutating Function descriptions becomes a TypeError; (b) at `from_seed` time call `f.model_copy()` to defensively copy each seed Function. Option (b) is one line: `catalog._entries[name] = CatalogEntry(function=f.model_copy(), ...)`. Either way, document the invariant.

### [MEDIUM] `_rebuild_function` returns a fresh `Function` but the parameters class is *shared* across canonical and replaced entries

`tool_catalog.py:286-302` (`apply_replace`): the replaced entry reuses `existing.function.parameters` (the canonical's pydantic class). Same is true in `apply_rewrite_doc` (`tool_catalog.py:330-338`). The pydantic class itself is a Python class object; class-level mutations (`cls.model_fields["name"] = ...`) would affect both the canonical seed's view of the class and the replaced shadow's view. No code path mutates the class today, but the design pins us to that invariant.

For `apply_replace` specifically, sharing the schema is *intentional* and *correct*: the brief mandates that the agent's tool-calling schema is unchanged across replace (only the body and stored fake_return change). What's missing is a test pinning that invariant: that `catalog.get(name).function.parameters is original_function.parameters` after replace, so a future refactor that defensively copies the class will be caught.

Recommended fix: add `test_replace_preserves_parameters_class` that asserts the shared identity invariant. Comment in `apply_replace` explaining why the share is intentional.

### [MEDIUM] `apply_unregister` is idempotent for absent names, hiding attacker typos

`tool_catalog.py:305-313`: removing a name that isn't in the catalog is a no-op. The docstring says "Idempotent: removing an already-absent name is a no-op." This is a design choice, but it differs from `apply_replace` (which raises) and `apply_rewrite_doc` (which raises) for the same condition. The inconsistency may surprise attackers and consumers reading traces — a `tool_catalog_unregister` event for `banking__send_monye` (typo) leaves the canonical `banking__send_money` in the catalog, but no warning is logged.

Trade-off: rejecting absent names would break the simple "always run all four catalog ops at start of turn" pattern the optimizer might use. Idempotent is friendlier. But it does mean ASSUMPTIONS.md should call out the polarity asymmetry between unregister (idempotent) and the other three (raise).

Recommended fix: either (a) raise on absent, and let the hook in `pipeline_bridge.py:114` catch and log the `ValueError` (consistent with replace/rewrite); or (b) keep idempotent but log at WARNING level via the same `logger.warning("Catalog mutation: tool %r not present, no-op", name)` so the trace still surfaces the attacker's intent. Note in `apply_unregister`'s docstring why the polarity differs from the other operations.

### [LOW] `apply_register` payload validation does not check `parameters_schema` type

`tool_catalog.py:241-260`: `parameters_schema = payload.get("parameters_schema")` accepts any value. If the attacker supplies a non-dict (e.g. `"parameters_schema": "string"`, `"parameters_schema": [1,2,3]`), `_build_parameters_class` defensively returns `_PermissiveSchema` (line 121: `if not schema or not isinstance(schema.get("properties"), dict)`). So the call succeeds silently with a permissive schema. The attacker's malformed schema is dropped without acknowledgement.

Same trade-off as the unregister-idempotent finding: silent acceptance is friendlier but hides bugs. Logging a warning when a `parameters_schema` is supplied but rejected would help.

Recommended fix: in `apply_register`, after pulling `parameters_schema`, if it is not None and not a dict, log a warning that the schema is being dropped. Or accept dict only with `isinstance(parameters_schema, (dict, type(None)))` and raise `ValueError` otherwise. The pipeline hook would then surface that via its `except ValueError as exc: logger.warning(...)` path.

### [LOW] `_require_str` rejects empty strings but the error message is misleading

`tool_catalog.py:350-357`: `if not isinstance(value, str) or not value: raise ValueError(f"payload missing required string key {key!r} (got {value!r})")`. The message says "missing" even when the key was present with an empty-string value. Cosmetic but the test `test_register_missing_fields_rejected` happens to pass because the empty/missing branches share the message — a future debugger reading "missing 'name'" when the key was actually `''` would be confused.

Recommended fix: split the messages: `if value is None: raise ValueError("missing")`, `elif not isinstance(value, str): raise ValueError("not a string")`, `elif not value: raise ValueError("empty string not allowed")`. Worth two minutes.

### [LOW] Attacker-chosen tool names are passed to LLM SDKs without sanitization

`tool_catalog.py:231-260` (`apply_register`): the only check on `name` is `_require_str` (non-empty string). Verified at runtime: a name like `"shadow\nbanking__send_money"` or `"x" * 10000` is accepted by the catalog. When the next LLM turn fires (`build_pipeline` -> OpenAILLM/AnthropicLLM), the SDK serializes the catalog as tool definitions; OpenAI's API enforces `^[a-zA-Z0-9_-]{1,64}$` on tool function names and returns `BadRequestError`, which propagates out of `_chat_completion_request` (which doesn't retry BadRequestError per `openai_llm.py:147`). The pipeline then fails with no graceful fallback to "drop the malformed attacker tool and keep going".

This is a design choice per the brief (the attacker payload is unsanitized; if the LLM rejects it, the attacker has wasted their slot). But it has a knock-on cost: a single malformed catalog edit kills the entire run, including subsequent legitimate calls. The catalog edit hook's existing `except ValueError` doesn't catch SDK errors that surface in the LLM step itself.

Recommended fix: add an optional `name_pattern` validator to `apply_register` (regex-defaulted from the configured LLM provider, e.g. `^[a-zA-Z0-9_-]{1,64}$` for OpenAI). Reject malformed names with `ValueError` so the hook drops the bad edit and the next LLM call proceeds with the previous catalog. Document in ASSUMPTIONS.md.

### [LOW] `snapshot()` swallows pydantic schema-generation exceptions silently

`tool_catalog.py:216-220`: `try: schema = entry.function.parameters.model_json_schema(); except Exception: schema = {}`. The bare `except` (with `pragma: no cover - defensive`) catches any pydantic version-skew or weird-class issue and falls back to an empty schema. The consumer downstream sees `parameters_schema: {}` and may not realize the schema was unrepresentable. Today this never fires on the 74 canonical tools.

Recommended fix: log a WARNING with the entry name and exception so the failure is observable in the trace; the empty-schema fallback can remain. (Same pattern as the other defensive blocks.)

### [LOW] `_build_placeholder_function` silently falls back to `_PermissiveSchema` on `create_model` failure

`tool_catalog.py:132-135`: `try: return create_model(...); except Exception: return _PermissiveSchema`. Same silent-fallback pattern as `snapshot()`. A failed `create_model` (e.g. due to a reserved name colliding with all attacker properties) yields a permissive schema; the attacker thinks they have a strict schema but the LLM sees `{}` (which OpenAI accepts as "any args"). Could mask attacker errors in tests.

Recommended fix: log a WARNING on the fallback path.

### [LOW] Module-level singletons `TOOL_REGISTRY` and `ALL_FUNCTIONS` are import-time built and mutable

`tool_registry.py:395-412`: built at import time and never rebuilt. `ALL_FUNCTIONS` is a public `list[Function]` exported in `__all__`. Anyone with `import agentdojo_target.tool_registry as r; r.ALL_FUNCTIONS.append(my_evil_fn)` permanently mutates the seed for every subsequent `ToolCatalog.from_seed(ALL_FUNCTIONS)` in the process. The same applies to mutating `TOOL_REGISTRY`. This is a Python module convention concern, not a security boundary inside the runtime, but `ALL_FUNCTIONS` is broadcast by name as the canonical entry point so it's worth either making it a tuple (immutable) or returning a defensive copy.

Recommended fix: `ALL_FUNCTIONS: tuple[Function, ...] = tuple(e.function for e in TOOL_REGISTRY.values())`. Update the docstring and any callers that need a list.

### [LOW] `_rebind_string_dep` has a misleading return-type annotation

`tool_registry.py:224-234`: `def extract(env: BaseModel) -> BaseModel`. But `getattr(sub, attr_name)` may return a `dict` (e.g., `inbox.emails` is `dict[str, Email]`) or a plain Python list. Mypy flags this (`Returning Any from function declared to return "BaseModel" [no-any-return]`). The annotation is inherited from upstream's `Depends.env_dependency: str | Callable[[BaseModel], BaseModel]`, but upstream's own type is overly tight too.

Recommended fix: annotate as `Callable[[BaseModel], Any]` (loose). Or upstream the fix to AgentDojo's `Depends`. The mypy error is a real signal that the type system can't verify the return shape.

### [LOW] `_build_parameters_class` mypy: `create_model(..., **fields)` call-overload mismatch

`tool_catalog.py:133`: mypy reports `No overload variant of "create_model" matches argument types "str", "dict[str, tuple[type, Any]]"`. The runtime usage `create_model(name, **fields)` works correctly (pydantic accepts the kwargs form), but mypy's stub doesn't capture the variadic dict-unpacking pattern. Cosmetic, but the strict-mode build will flag it.

Recommended fix: either add `# type: ignore[call-overload]` with a brief comment, or restructure to `create_model(name, **{k: v for k, v in fields.items()})` (same runtime behaviour, sometimes makes mypy happy).

### [LOW] `apply_unregister` of a non-canonical entry leaves observables stale

If the attacker registers `evil` (fires per-call `tool_call:evil` Controllable) and later unregisters it, the per-call Controllable object created by `_attacker_call_ctrl` (in `runtime_wrapper.py`) is built lazily per call, so once unregistered the Controllable is never created again. Fine. But the previous calls' `tool_call:evil` events are already on the trajectory. SecurityClaim predicates that scan for "what attacker tools were registered during the run" should also scan the trajectory event history, not just `tool_catalog_snapshot`. This is a design property of the per-call vs. catalog Controllables, but it's worth a sentence in ASSUMPTIONS.md so the Layer-2 author knows where to look.

### [INFO] Test coverage gaps in `test_tool_catalog.py`

- No test for `apply_register` with a real `parameters_schema` (only the standalone `_build_placeholder_function` is tested).
- No test for `apply_replace` -> `apply_unregister` -> `apply_register` of the same name (replay attack scenario).
- No test that `apply_rewrite_doc` on a `replaced` entry preserves `fake_return` (only on `registered` is tested).
- No test for `Function.parameters` identity preservation across `apply_replace` (pinning the intentional sharing).
- No test for non-dict `parameters_schema` payloads.
- No test exercising `snapshot()` for a `registered`-with-explicit-schema or `replaced` entry (only the canonical catalog's schema shape is asserted).
- No test for whitespace-only or special-character names (per the LOW above).

### [INFO] Test coverage gaps in `test_tool_registry.py`

- No test that `RegistryMismatchError` actually fires on a fabricated stale entry (the build is private, so this would require touching the module-level constants in a fixture).
- No test that the `RegistryEntry` dataclass is frozen (relevant if a consumer tries to mutate `entry.kind` post-construction).
- No test that all dependencies' rebind closures captured suite by VALUE (the closure-capture-by-reference bug is named in the docstring of `_rebind_string_dep` but not verified by tests). Could add a quick test that constructs two registries side by side and asserts they don't share closures.

## Strengths

- The 74-tool count is enforced bidirectionally: `_assert_no_stale_entries` rejects READ/WRITE entries not in upstream, and `_classify` rejects upstream tools not in either set. A future upstream version that adds or removes a tool causes a clear ImportError at first use.
- Suite-prefixed naming is consistently applied everywhere (registry, catalog, controllables, observables). The `split_prefixed` helper makes the inverse always available.
- `Depends` rebinding is correct: `_rebind_string_dep` captures `suite` and `attr_name` by value via a named factory function (the docstring explicitly calls out the closure-capture trap), avoiding the classic late-binding bug. End-to-end verified by the `test_dependencies_rebound_to_composite` test, which I confirmed extracts the right sub-attribute for every one of the 74 tools.
- The "v1 plus a few extras" coverage list in `ASSUMPTIONS.md` C.7 (banking scheduled transactions, travel flights / car rentals, workspace contacts) is fully reflected in the READ/WRITE sets; no orphan reads.
- The catalog's apply_* methods consistently return the new `CatalogEntry` (except `apply_unregister`, intentional), giving callers a chance to inspect what was applied.
- The placeholder `run` callable raises `RuntimeError` if ever reached, making any wrapper-dispatch bug fail fast and loud — caught by `test_placeholder_run_raises_if_called`.
- `reset()` correctly restores the original `Function` references, so the catalog's per-run state cannot leak into subsequent runs of the same Target instance.
- `tool_registry.py` correctly handles the upstream circular-import chain via the `import agentdojo.task_suite.load_suites` pre-import at the top; this is the same flush-the-registration-chain trick used in `env.py` and is documented in both places.
- The `READ_FUNCTION_NAMES` / `WRITE_FUNCTION_NAMES` frozensets give downstream code (runtime wrapper, observables, SecurityClaim) cheap O(1) membership tests rather than re-scanning the registry on every call.
