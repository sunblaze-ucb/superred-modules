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

---

## B. Tool catalogue

### B.1 Catalogue editability is new

**AgentDojo**: tool list is fixed for the duration of a run; the agent sees the suite's pre-declared tools (with `tool_filter` defense possibly removing some).

**Us**: four catalog Controllables (`register`, `replace`, `unregister`, `rewrite_doc`) fire at the start of each agent turn via a spliced `BasePipelineElement` hook. Registered/replaced tools fire per-call Controllables when invoked, carrying the agent-supplied args; optimizer's injection response is the fake return the agent sees.

**Why**: the brief mandates this as a second attacker capability surface (models malicious-MCP / tool-shadowing threat).

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

---

## D. Polarity of upstream `security` / `utility`

### D.1 `security == True` means "attack succeeded"

**AgentDojo** (`base_tasks.py:126` docstring): "True if injection was executed correctly." Polarity matches this in `_check_injection_task_security` (`task_suite.py:297-311`). The skip-branches in `benchmark.py:147,153,...` set `security=True` for error cases ("agent stayed safe") which is a separate fallback semantic.

**Us**: SecurityClaim Layer-1's `EvaluationResult.primary_score.value = float(security_result)`, so success=True corresponds to attack succeeded. Naming in our code uses `attack_succeeded` rather than `security` to avoid the documentation-vs-fallback trap.

**Why**: explicit naming reduces the chance of polarity inversion in downstream analysis code.

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

(more sections to be added as implementation discovers divergences)
