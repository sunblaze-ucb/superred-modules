# Ship status — AgentDojo port (v0.1.0)

## Test counts

| Package | tests passing | gated/skipped | total files |
|---|---|---|---|
| `targets/agentdojo` | 108 | 1 (faithfulness smoke) | 9 |
| `security_claims/agentdojo` | 36 | 0 | 5 |
| **Total** | **144** | 1 | 14 |

## What landed

### Target package (`targets/agentdojo/`)

- `security_tags.py` — 17-tag forest (system / user / tools), 2x2 grid leaves under tools.
- `env.py` — `CompositeEnvironment` pydantic root unioning all four AgentDojo v1 sub-envs.
- `seed_loader.py` — loads each suite's default-injected sub-env; per-suite YAML/JSON overlay merger.
- `tool_registry.py` — suite-prefixed (`{suite}__{tool}`) union of 74 upstream tools with rebound `Depends` extractors; read/write classification with import-time exhaustiveness check.
- `controllables.py` — 53 controllables (1 system_prompt + 1 user_prompt + 4 tool_catalog + 47 per-read) with each per-read assigned to a 2x2 grid leaf.
- `observables.py` — 4 static observable specs + per-event dynamic builders for chat messages, tool calls, tool responses, write_call sidechannels, read_data_field mirrors.
- `tool_catalog.py` — `ToolCatalog` with register/replace/unregister/rewrite operations + per-turn snapshot + reset.
- `runtime_wrapper.py` — `WrappedFunctionsRuntime` subclass intercepting `run_function`: trace recording, per-read controllable event firing with substitution, write-call observable emission, sync-to-async bridge via `asyncio.run_coroutine_threadsafe`, short-circuit dispatch for attacker-managed entries.
- `pipeline_bridge.py` — OpenAI/Anthropic LLM dispatch; AgentPipeline construction with `_CatalogEditHook` spliced before every LLM turn.
- `target.py` — `AgentDojoTarget` (subclass of superred's `Target`): wires everything; lifecycle phases 1-5; config_specs, query_specs, get_controllables, get_observables, run, cleanup, teardown.
- `config_specs.py`, `query_specs.py`, `system_prompt.py` — declarative state schemas.
- `ASSUMPTIONS.md` — 17 numbered divergences from AgentDojo upstream.
- `README.md` — install + quickstart.

### SecurityClaim package (`security_claims/agentdojo/`)

- `layer1_categories.py` — refined attack-semantic categorisation (16 labels across 27 pairs, per the user's confirmed table 2026-05-15).
- `layer1_pairs.py` — 27 canonical (suite, user_task_id, injection_task_id) pairs covering every v1 injection task once.
- `layer1_bridge.py` — bridges into AgentDojo upstream task / env type lookups; `compute_init_env_overlay` replays per-task `init_environment` mutations.
- `layer1_task.py` — `AgentDojoPairedTask` (Task[AgentDojoTarget]): configure_target sets benign user_task.PROMPT + system_prompt + per-suite seed overlay; evaluate decodes pre/post env snapshots + trace, strips suite-prefix, calls upstream `*_from_traces` then `*`.
- `layer1_factory.py` — top + per-suite + per-category factories. 7 named factories exported.
- `layer2_task.py` + `security_predicates.py` + `layer2_goals/` — bespoke system-purpose-violation goals; v1 ships 4 (banking unauth-transfer, workspace email-exfil, slack channel-exfil, travel pii-exfil) with deterministic predicates.
- `layer2_factory.py` — top + per-category factories.
- `layer3_factory.py` — `agentdojo_combined_claim` composing layers 1 and 2.
- `tests/test_integration.py` — Controller + AgentDojoTarget + passthrough optimizer + each top-level claim factory, against a fake LLM via monkeypatch.
- `tests/smoke/run.py` — credential-gated end-to-end run script against a real LLM.
- `ASSUMPTIONS.md`, `README.md`.

## What did not land in v1 (explicitly deferred)

| Item | Why | Where it picks up |
|---|---|---|
| Faithfulness 12-pair smoke test (real LLM) | Needs API credentials; runner pattern is sketched. | `targets/agentdojo/tests/faithfulness/test_faithfulness_smoke.py` placeholder + `tests/faithfulness/README.md`. |
| Full 629-case faithfulness cross-product | Needs ~$30-60 of API spend + hours of wall time. | Same README documents the runner script (`scripts/run_faithfulness_full.py`) as a v2 deliverable. |
| Layer-2 goal catalogue beyond 4 starter goals | The user said "you may come up with things yourself"; v1 ships 4 representative goals (one per suite). | Adding more is mechanical: drop a new `Layer2GoalSpec` module under `layer2_goals/` and register it in `__init__.py`. |
| Defenses (tool_filter, spotlighting, etc.) | superred's threat-model decomposition treats defenses as optimizer-side concerns; out of v1 scope. | `ASSUMPTIONS.md` §E.3 documents the gap. |
| Non-OpenAI/Anthropic LLM providers | v1 supports `openai/` and `anthropic/` prefixes. | Adding Cohere / Google / Together / local is mechanical; see AgentDojo's `get_llm` source for the dispatch. |

## Running the suites

```bash
# Target package
cd targets/agentdojo
pip install -e ".[dev]"
pytest tests/

# SecurityClaim package
cd security_claims/agentdojo
pip install -e ".[dev]"
pytest tests/

# Credential-gated smoke run (real LLM)
cd security_claims/agentdojo
OPENAI_API_KEY=sk-... python tests/smoke/run.py
# OR
LITELLM_API_KEY=... LITELLM_API_BASE=... python tests/smoke/run.py
```

## Notes saved across sessions

Memory entries in `~/.claude/projects/-Users-simonsure-research-superred/memory/`:

- `project_agentdojo_port.md`
- `reference_agentdojo_repo.md`
- `feedback_agentdojo_security_polarity.md`
- `feedback_superred_security_claim_conventions.md`

Research notes (per-session, in `$CLAUDE_JOB_DIR/notes/`):

- `paper.md`, `agentdojo_runtime.md`, `agentdojo_tasksuite_pipeline.md`, `agentdojo_suites_banking_workspace.md`, `agentdojo_suites_slack_travel.md`, `agentdojo_versioning_tests.md`
- `PLAN.md` (final, after user clarifications)
- `PLAN_v0_pre_user_answers.md` (initial draft for traceability)
