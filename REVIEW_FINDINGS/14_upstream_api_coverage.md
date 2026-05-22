# Upstream API Coverage Audit

Scope: enumerate everything the AgentDojo port consumes from `agentdojo==0.1.35` and identify coverage gaps.

Port roots audited:
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/targets/agentdojo/`
- `/Users/simonsure/research/superred/superred-modules/.claude/worktrees/agentdojo-port/security_claims/agentdojo/`

Upstream root:
- `/Users/simonsure/research/superred/.venv/lib/python3.13/site-packages/agentdojo/`

---

## 1. Tools: 74 confirmed

The v1 suites expose exactly 74 tools, broken down as:

| Suite | Tools | Notes |
| --- | --- | --- |
| workspace | 24 | includes `get_unread_emails` (semi-mutating) |
| travel | 28 | includes 4 calendar tools shared shape with workspace |
| banking | 11 | |
| slack | 11 | includes `get_webpage` (audit-trail write) |
| TOTAL | **74** | |

Verified by importing `agentdojo.task_suite.load_suites` and summing `task_suite.tools` lengths per suite (sourced via `get_suites('v1')`).

The port's `tool_registry.ALL_FUNCTIONS` is built from `_build_registry()` which walks each suite's `.tools` list. The R/W classification (`READ_TOOLS` / `WRITE_TOOLS`) explicitly enumerates each tool by name and asserts exhaustiveness in `_assert_no_stale_entries` and `_classify`. Adding up the R/W sets:
- banking: 6 read + 5 write = 11
- workspace: 14 read + 10 write = 24
- slack: 5 read + 6 write = 11
- travel: 22 read + 6 write = 28

Total = **74**, matches upstream. The registry will raise `RegistryMismatchError` at import time if upstream's tool list ever drifts in either direction, so coverage is statically pinned.

Relevant files:
- Port: `targets/agentdojo/src/agentdojo_target/tool_registry.py` lines 65-167 (R/W tables), 365-392 (registry build).
- Upstream: `agentdojo/default_suites/v1/{banking,workspace,slack,travel}/task_suite.py`.

**Coverage gap: NONE.** All 74 tools are covered. The semi-mutating `get_unread_emails` and audit-trail `get_webpage` are intentionally classed as reads (documented in port docstring lines 18-22).

---

## 2. Tasks: 97 user + 27 injection = 124 confirmed at upstream; partial scope in canonical claim

**Upstream count (verified via `get_suites('v1')`):**

| Suite | User tasks | Injection tasks |
| --- | --- | --- |
| workspace | 40 | 6 |
| travel | 20 | 7 |
| banking | 16 | 9 |
| slack | 21 | 5 |
| TOTAL | **97** | **27** |

Both totals match the brief (97 + 27 = 124 classes). Note: the file-level `^class UserTaskN+` grep gives 86 user tasks, but `get_suites('v1')` returns 97 because v1_1, v1_1_1, v1_1_2, v1_2 versions are layered into the v1 registry by `load_suites.py` (the port pins `v1` only; per ASSUMPTIONS H.1, the additions are folded back into v1's `user_tasks` dict via registration decorators).

**Port reference: PARTIAL.** The canonical pair scope `CANONICAL_PAIRS` in `layer1_pairs.py` references:
- All **27** injection tasks (one pair per injection task)
- Only **12** of **97** user tasks (12 distinct user-task IDs across the 27 pairs)

This is by design (per the docstring at `layer1_pairs.py` lines 13-17): "this canonical scope is just the per-claim default - callers can pass an explicit `pairs=...` to `agentdojo_layer1_claim` for any other scope including the full 629-case cross-product." The `agentdojo_layer1_claim(pairs=...)` accepts arbitrary pair tuples and resolves them via `get_user_task` / `get_injection_task` in `layer1_bridge.py`, which delegates to `get_suite(...).get_user_task_by_id(...)` and `.injection_tasks[...]`. So all 124 task classes are reachable through the bridge; the default scope just doesn't pin them.

**Gap analysis:**
- 85 of 97 user tasks (87%) are NEVER referenced by the default Layer-1 claim. They are still listed in upstream's `task_suite.user_tasks` and discoverable via the bridge, but no port-side fixture or test currently sweeps them.
- The full 629-pair cross-product (97 user x 27 injection per suite, totalling ~600+ for the suite-respecting variant) is reserved for the faithfulness test runner, which today contains only a smoke test (`tests/faithfulness/test_faithfulness_smoke.py`) and a TODO for the full sweep at `scripts/run_faithfulness_full.py` (per the file docstring lines 13-16).

Files:
- Port: `security_claims/agentdojo/src/security_claim_agentdojo/layer1_pairs.py` (canonical scope), `layer1_factory.py` (filter API), `layer1_bridge.py` (lookup).
- Upstream: `agentdojo/task_suite/load_suites.py`, `agentdojo/task_suite/task_suite.py`.

---

## 3. Suites and YAML files: 4 confirmed

Upstream ships 4 `environment.yaml` files in `agentdojo/data/suites/{banking,slack,travel,workspace}/environment.yaml` plus the matching `injection_vectors.yaml` per suite. The port's `seed_loader.load_composite_seed()` calls `get_suite(v1, name).load_and_inject_default_environment({})` for each of the 4 suite names, building a `CompositeEnvironment` with all four sub-envs populated (per ASSUMPTIONS C.5).

Files:
- Port: `targets/agentdojo/src/agentdojo_target/seed_loader.py` lines 24, 43-46.
- Upstream: `agentdojo/data/suites/{banking,workspace,slack,travel}/environment.yaml` (4 files); the workspace suite also has 3 `include/` files (`inbox.yaml`, `calendar.yaml`, `cloud_drive.yaml`) loaded transitively.

**Coverage gap: NONE.** All 4 environment YAMLs load.

---

## 4. Pipeline elements used vs. available

**Upstream `BasePipelineElement` subclasses (full list from `agent_pipeline/__init__.py`):**

| Class | File | Port uses? |
| --- | --- | --- |
| `AgentPipeline` | `agent_pipeline.py` | **YES** (`pipeline_bridge.py:31`) |
| `BasePipelineElement` | `base_pipeline_element.py` | **YES** (subclass `_CatalogEditHook`, `_MessageStreamHook`) |
| `InitQuery` | `basic_elements.py` | **YES** (`pipeline_bridge.py:33`) |
| `SystemMessage` | `basic_elements.py` | **YES** (`pipeline_bridge.py:33`) |
| `ToolsExecutor` | `tool_execution.py` | **YES** (`pipeline_bridge.py:36`) |
| `ToolsExecutionLoop` | `tool_execution.py` | **YES** (`pipeline_bridge.py:36`) |
| `OpenAILLM` | `llms/openai_llm.py` | **YES** (`pipeline_bridge.py:35`) |
| `AnthropicLLM` | `llms/anthropic_llm.py` | **YES** (`pipeline_bridge.py:34`) |
| `CohereLLM` | `llms/cohere_llm.py` | **NO** (NotImplementedError at `_build_llm`) |
| `GoogleLLM` | `llms/google_llm.py` | **NO** (NotImplementedError) |
| `LocalLLM` | `llms/local_llm.py` | **NO** (NotImplementedError) |
| `PromptingLLM` / `BasePromptingLLM` | `llms/prompting_llm.py` | **NO** |
| `OpenAILLMToolFilter` | `llms/openai_llm.py` | **NO** (used only by `tool_filter` defense) |
| `TransformersBasedPIDetector` | `pi_detector.py` | **NO** |
| `PromptInjectionDetector` (base) | `pi_detector.py` | **NO** |
| `GroundTruthPipeline` | `ground_truth_pipeline.py` | **NO** |
| `ToolSelector` | `planner.py` | **NO** |
| `ToolUsagePlanner` | `planner.py` | **NO** |

**Port-defined `BasePipelineElement` subclasses (in `pipeline_bridge.py`):**
- `_CatalogEditHook` (lines 64-140): fires the four tool-catalog Controllables before every LLM turn; implements the catalog editability surface defined in ASSUMPTIONS B.1.
- `_MessageStreamHook` (lines 148-192): emits per-turn `agent_trace_message_NNNN` observables.

**Worth wrapping? Brief assessment:**
- `GroundTruthPipeline`: useful as an oracle baseline (it returns the ground-truth function calls); not consumed today but might be valuable for evaluating optimizer regressions without LLM cost. Not blocking for v1.
- `ToolSelector` / `ToolUsagePlanner`: paper baselines for agent decomposition; out of scope for v1's `no_defense` pipeline.
- `PromptingLLM`: enables non-tool-calling models via JSON-formatted prompts; valuable if a future v2 targets local/non-tool-calling models.

---

## 5. Defense pipeline elements: only `no_defense` baseline supported

Upstream `agent_pipeline.py` exposes `DEFENSES = ["tool_filter", "transformers_pi_detector", "spotlighting_with_delimiting", "repeat_user_prompt"]` (lines 43-49).

**Port supports: 0 defenses.** Only the `no_defense` baseline shape is bundled (`pipeline_bridge.build_pipeline` builds `[SystemMessage, InitQuery, hook, llm, msg_hook, tools_loop]`). ASSUMPTIONS E.3 explicitly accepts this and defers defenses to v2.

**Defenses NOT supported (and the upstream elements they require):**
| Defense | Required upstream element | Notes |
| --- | --- | --- |
| `tool_filter` | `OpenAILLMToolFilter` | LLM-driven pre-filter; OpenAI-only upstream |
| `transformers_pi_detector` | `TransformersBasedPIDetector` (+ `protectai/deberta-v3-base-prompt-injection-v2`) | Inserts a detector inside the tools loop, escalates with `AbortAgentError` |
| `repeat_user_prompt` | `InitQuery` (re-used inside the loop) | No new element; just pipeline rewiring |
| `spotlighting_with_delimiting` | none (system message + delimited formatter) | Pure prompt-engineering; cheapest to port |

The brief's E.3 calls these out as v2 follow-ups and there are no port-side stubs. Adding `spotlighting_with_delimiting` and `repeat_user_prompt` is mostly mechanical (no new external deps); `tool_filter` requires reusing the same OpenAI client and `transformers_pi_detector` would pull in `transformers` + a model checkpoint.

---

## 6. `task_suite.load_suites` integration

The port references `agentdojo.task_suite.load_suites` from 5 files (per `grep -rn "load_suites"`):

| File | Reason |
| --- | --- |
| `targets/agentdojo/src/agentdojo_target/env.py:32` | Pre-import side effect: flush the registration chain so v1_1/v1_2 layered tasks register against the v1 suites BEFORE `BankingEnvironment`/etc. are imported, avoiding the circular-import bug documented at lines 23-31 |
| `targets/agentdojo/src/agentdojo_target/tool_registry.py:31` | Same pre-import side effect |
| `targets/agentdojo/src/agentdojo_target/seed_loader.py:20` | Uses `get_suite(version, name)` to call `load_and_inject_default_environment({})` per suite |
| `security_claims/agentdojo/src/security_claim_agentdojo/layer1_bridge.py:20,22` | Pre-import + uses `get_suite` to look up `BaseUserTask` / `BaseInjectionTask` instances by ID |
| `targets/agentdojo/tests/test_concurrent_isolation.py:33` and `tests/test_toolsexecutor_per_turn.py:38` | Same pre-import side effect for test setup |

**Integration health: GOOD.** The circular-import workaround (verified at session start via direct repro) is consistently applied at every module that touches `agentdojo.default_suites.v1.*` imports. The pattern is: `import agentdojo.task_suite.load_suites  # noqa: F401` BEFORE any per-suite import. Without it, the v1_1_1 layered tasks try to import `WorkspaceDeepDiff` from a partially-initialised `v1.workspace.task_suite` module and crash. The port respects this in every entry point.

---

## 7. `benchmark.py` features deliberately ignored

The upstream `benchmark.py` orchestrates the full cross-product and adds error-handling fallbacks. The port does NOT call `benchmark_suite_with_injections`, `run_task_with_injection_tasks`, or `run_task_without_injection_tasks`; instead the superred Controller drives per-task execution. The features intentionally NOT carried over:

| Upstream feature | Lines | Port behavior |
| --- | --- | --- |
| `SuiteResults` aggregate | `benchmark.py:23-34` | Replaced by superred `EvaluationResult` + per-suite/per-category sub-scores |
| Skip-on-error semantic: `BadRequestError` (context length, max_tokens) sets `security=True` | `benchmark.py:120-132` | NOT replicated; the port re-raises and the Controller records `stop_reason='error'` with the traceback. Per ASSUMPTIONS D.1: "the skip-branches set `security=True` for error cases which is a separate fallback semantic" - explicitly flagged and avoided to prevent polarity confusion. |
| Skip-on-error: Cohere `ApiError` internal-server-error sets `security=True` | `benchmark.py:133-141` | NOT replicated (Cohere unsupported anyway) |
| Skip-on-error: Google `ServerError` sets `security=True` | `benchmark.py:142-147` | NOT replicated (Google unsupported anyway) |
| DoS-attack semantic: `security = not utility` | `benchmark.py:149-150` | NOT replicated (no DoS attacks ported; the upstream attack registry is also not consumed) |
| `force_rerun` / `load_task_results` log caching | `benchmark.py:85-103, 253-270` | NOT replicated; superred's `Controller` writes its own per-task JSON via `results_dir` |
| `TraceLogger` context manager | `benchmark.py:107-115` | Replaced by superred's event/trajectory infrastructure |
| `injection_tasks_utility_results` (run injection tasks as user tasks first, warn on failure) | `benchmark.py:200-209` | NOT replicated |
| Benchmark version param | `benchmark.py:49, 213-219` | Hardcoded to `v1` in the port (ASSUMPTIONS H.1) |

**Coverage gap noted:** The `BadRequestError`/`ApiError`/`ServerError` -> `security=True` fallback is upstream's way of recording "we couldn't verify, so assume safe." The port's choice to surface errors via `stop_reason='error'` is more transparent (and aligns with superred's threat-model semantics) but means a direct numerical comparison with upstream's published `runs/` results will count error pairs differently. The faithfulness sweep needs to either re-implement this skip semantic in the comparator or filter out error pairs before computing agreement.

Also NOT consumed from upstream:
- `agentdojo.attacks.*`: the port does not use `BaseAttack`, `attack_registry`, `important_instructions_attacks`, `dos_attacks`, or `baseline_attacks`. Attacker capability in superred is modelled as the optimizer's `ControllableInjection` payload, not a pre-canned attack.
- `agentdojo.scripts.benchmark` / `agentdojo.scripts.check_suites`: CLI entry points; not used.
- `agentdojo.logging.Logger` / `TraceLogger`: replaced by trajectory recording.
- `agentdojo.models.MODEL_PROVIDERS` / `ModelsEnum`: the port maps providers manually in `_build_llm` (`pipeline_bridge.py:248-280`).

---

## 8. LLM provider support

**Upstream `get_llm` dispatch table (`agent_pipeline.py:70-125`):**
- `openai` -> `OpenAILLM`
- `anthropic` -> `AnthropicLLM` (with `-thinking-N` suffix parsing)
- `together` -> `OpenAILLM` against `https://api.together.xyz/v1`
- `together-prompting` -> `PromptingLLM` against Together's OpenAI-compat endpoint
- `cohere` -> `CohereLLM` (uses `cohere.Client()`)
- `google` -> `GoogleLLM` (Vertex AI via `google.genai.Client(vertexai=True, ...)`)
- `local` -> `LocalLLM` (port-driven local OpenAI-compat endpoint)
- `vllm_parsed` -> `OpenAILLM` against the local vLLM endpoint

**Port `_build_llm` (`pipeline_bridge.py:229-279`):**
- `openai/<model>` -> `OpenAILLM`. Accepts `api_base`/`api_key` overrides (relevant for LiteLLM proxy).
- `anthropic/<model>` -> `AnthropicLLM`. Parses optional `-thinking-N` suffix.
- All other prefixes -> `NotImplementedError` with an explicit pointer to the upstream dispatch table.

**Unsupported in port (in priority order of likely demand):**
| Provider | Required upstream element | Effort to add |
| --- | --- | --- |
| Together (OpenAI-compat) | `OpenAILLM` + Together base URL | Trivial; uses existing element |
| Local (OpenAI-compat / vLLM) | `OpenAILLM` + localhost base URL OR `LocalLLM` for tool-delimited models | Trivial-to-moderate (LocalLLM needs `tool_delimiter` config) |
| Google (Vertex) | `GoogleLLM` + `google-genai` client | Moderate; needs `GCP_PROJECT` / `GCP_LOCATION` env wiring |
| Cohere | `CohereLLM` + `cohere.Client` | Moderate; needs a separate client field |
| `together-prompting` | `PromptingLLM` | Moderate; non-tool-calling agent flow |

The port's choice to lock to openai+anthropic in v1 aligns with what the LiteLLM proxy supports and what the user clarified for faithfulness testing (D.2 / SORRY-Bench substitution). Adding Together via `OpenAILLM` would be a 5-line addition.

---

## 9. Other upstream surfaces consumed by the port (completeness check)

For an audit trail, the full set of `from agentdojo.X import ...` symbols (excluding the LLM/pipeline ones above):

| Symbol | File | Used by |
| --- | --- | --- |
| `agentdojo.functions_runtime.Function` | `tool_registry.py`, `tool_catalog.py` | Build catalog entries |
| `agentdojo.functions_runtime.Depends` | `tool_registry.py` | Rebind extractors |
| `agentdojo.functions_runtime.FunctionsRuntime` | `runtime_wrapper.py`, `pipeline_bridge.py` | Parent class + type hint |
| `agentdojo.functions_runtime.FunctionCall` | `target.py`, `layer1_task.py`, `layer2_*.py`, `security_predicates.py` | Trace records |
| `agentdojo.functions_runtime.TaskEnvironment` | `env.py`, `layer1_bridge.py`, `layer1_task.py` | Env base class |
| `agentdojo.functions_runtime.EmptyEnv`, `Env` | `pipeline_bridge.py` | Type hints in element signatures |
| `agentdojo.base_tasks.BaseUserTask`, `BaseInjectionTask` | `layer1_bridge.py`, `layer1_task.py` | Look up + dispatch upstream predicates |
| `agentdojo.types.ChatMessage` | `target.py`, `pipeline_bridge.py`, test files | Type hint |
| `agentdojo.default_suites.v1.{banking,slack,travel,workspace}.task_suite` | `env.py`, `tool_registry.py` | Source-of-truth tools and Environment classes |

No imports from `agentdojo.ast_utils`, `agentdojo.strenum`, `agentdojo.yaml_loader`, `agentdojo.attacks.*`, `agentdojo.scripts.*`, `agentdojo.benchmark`, `agentdojo.logging`, or `agentdojo.models`. All are either superseded by superred infrastructure or out of v1 scope.

---

## 10. Summary of coverage gaps

| Surface | Status | Notes |
| --- | --- | --- |
| 74 tools | **Full** | `RegistryMismatchError` pins exhaustiveness |
| 27 injection tasks (all 4 suites) | **Full** (canonical scope hits all 27) | |
| 97 user tasks | **Partial** (12 referenced by default scope; 97 reachable through bridge API) | Full 629-pair sweep is TODO at `scripts/run_faithfulness_full.py` |
| 4 environment YAMLs | **Full** | |
| `task_suite.load_suites` circular-import workaround | **Full** | Applied at every entry point |
| Pipeline elements (the 5 core: `SystemMessage`, `InitQuery`, `ToolsExecutor`, `ToolsExecutionLoop`, `OpenAILLM`/`AnthropicLLM`) | **Full** | |
| 4 defense pipelines | **None** | ASSUMPTIONS E.3 defers to v2 |
| 6 LLM providers beyond openai+anthropic | **None** | `NotImplementedError` with documented path forward |
| `benchmark.py` skip-on-error semantic | **None** (intentional) | Per ASSUMPTIONS D.1: rename to `attack_succeeded` to avoid polarity inversion |
| `benchmark.py` DoS-attack `security = not utility` semantic | **None** (intentional) | No DoS attacks ported |
| `benchmark.py` log-cache / `force_rerun` | **None** | Superseded by Controller's `results_dir` |
| `agentdojo.attacks.*` registry | **None** (intentional) | Attacker capability lives on optimizer side as `ControllableInjection` |
| `GroundTruthPipeline`, `ToolSelector`, `ToolUsagePlanner`, `PromptingLLM`, PI detectors | **None** | Out of v1 scope; worth revisiting for v2 |

**Net assessment:** the port is faithful to upstream's *primitives* (tools, tasks, suites, environments, the no-defense pipeline shape, and the upstream `security`/`utility` predicates) but is deliberately narrow on the *orchestration* surface (no defenses, no attack registry, no benchmark.py skip semantics, no broader LLM provider matrix). Every omission is enumerated either in `ASSUMPTIONS.md` or in the `NotImplementedError` paths of `_build_llm`. The 124-class coverage claim is true at the *upstream lookup* level (all reachable via `layer1_bridge.get_user_task` / `get_injection_task`) but the *default Layer-1 claim* only pins 27 of 124 (the 27 injection tasks + 12 hand-picked user tasks). A future faithfulness sweep that exercises the full 629-pair cross-product will exercise all 124 classes at least once.
