# ASSUMPTIONS: asb_target

Every deviation from upstream Agent Security Bench (ASB, `agiresearch/ASB`,
pinned commit `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`, MIT). The ASB agent
loop itself (plan-then-execute logic, prompts, plan format, simulated-tool
returns, success strings, the system-instruction scaffolding) is reproduced
verbatim; the deviations below concern packaging, the LLM transport,
re-expressing ASB's argparse-flag-driven injection as superred events, the
trust-boundary model, the durable memory store, and the trajectory.

This is a **bare runtime**: it exposes injection opportunities as
Controllables but performs **no injection by default** (with no attacker every
run is a clean, upstream-faithful baseline), and it has **no defense
infrastructure**. Specific attacks are an attacker's concern, not the target's.

## A. Vendoring & dependencies

- **A.1** The ASB source is vendored verbatim under `_asb_vendor/`
  (`pyopenagi/` + `aios/`), pinned to the SHA above (see
  `_asb_vendor/NOTICE.md`). ASB is not a pip package, so vendoring is the only
  way to run its real loop.
- **A.2** ASB's `AgentFactory` / `Interactor` (download + pip-install agents
  on demand) are **not used**; the target instantiates `SuperredReactAgent`
  directly and loads the scenario `config.json` from the vendored `example/`
  tree. No network agent download occurs.
- **A.3** De-dependency surgery, each marked in-code, no behavioural change:
  - `aios/llm_core/llm_classes/model_registry.py`: pruned to the
    OpenAI-compatible `GPTLLM`; dropped gemini/bedrock/claude entries.
  - `aios/llm_core/llms.py`: local-backend imports (`HfNativeLLM`/`OllamaLLM`/
    `vLLM`) made lazy inside the local-model branch.
  - `pyopenagi/agents/react_agent_attack.py`: `langchain`/`chroma` imports
    guarded (optional); the port uses its own memory store (D), not Chroma.
  - `aios/context/simple_context.py`: `import torch` made lazy (it is only used
    by GPU context snapshot/recover, which the port never calls; a top-level
    import needlessly pulled torch in, slowing imports and leaving torch's
    interpreter-exit handlers active under pytest).

## B. LLM transport

- **B.1** All inference is routed through the user's litellm proxy.
  `asb_target.llm_proxy.ProxyLLM` is a faithful copy of `GPTLLM.process` with
  the model-name assertion (`assert re.search('gpt', self.model_name,
  re.IGNORECASE)`, `gpt_llm.py`) **removed** so any proxy-served model id
  works, and with explicit `base_url`/`api_key` (rather than env vars). The
  configured model name is registered to `ProxyLLM` in the kernel registry.
- **B.2** ASB's hard-coded `time.sleep(2)` between LLM calls is configurable
  (`request_delay_seconds`, default `2.0` = ASB-faithful; `0` for fast tests).
- **B.3** The output-token cap is **pinned** (`max_output_tokens`, default
  `1024` = upstream `GPTLLM.max_new_tokens`) rather than the vendored argparse
  default of 256, so a kernel-wiring change cannot silently shrink it.
- **B.4** A dead / misconfigured endpoint **fails loudly**: a connection /
  auth / status / bad-request / unexpected error is recorded and the target
  aborts the run with a `RuntimeError`, instead of swallowing the provider
  error text into the agent transcript (which could silently contaminate the
  attack-success substring check). Only a neutral marker reaches the
  transcript. **Throttling is retried, then fatal.** The OpenAI client retries a
  429 with exponential backoff and Retry-After (`ProxyConfig.max_retries`,
  default 6, above the SDK's own 2 because a sweep runs many cells against one
  gateway). A 429 that survives those retries is sustained, and is recorded as
  a hard failure like any other. This port previously TOLERATED it, which is
  worse than it sounds: the agent's turn silently became the neutral marker and
  the run was then scored on a transcript the model never produced. Failing
  costs only time, because the task is simply re-run.

## C. Injection model (the core adaptation)

- **C.1** ASB selects an attack method via argparse flags and self-injects
  fixed strings. The port instead fires a superred controllable event at each
  of the four injection sites and applies the attacker's injection (or nothing):
  a `ControllablePreCallEvent` for the three input-side sites (DPI/PoT/MP) and a
  `ControllablePostCallEvent` for OPI (output-side; see C.3). **Clean by
  default**: with no attacker every site declines and the run is benign.
- **C.2 DPI**: the injection is appended to the benign `task_input`
  (`task_input += " " + value`, matching ASB's `+=`).
- **C.3 OPI**: fired on **every** non-final tool return **including the
  attacker tool's own observation**: the `function_name != self.tool_name`
  guard the earlier port added is **removed**, restoring upstream's
  provenance/name-blind behaviour (`react_agent_attack.py:188-189`). Each OPI
  event is tagged to the firing tool's scenario sub-boundary (E). OPI is
  delivered as a `ControllablePostCallEvent` (its `answer` carries the genuine
  observation, its `request` the call `{tool, params}`) because it tampers a
  tool's RETURNED observation -- the output side. This is both semantically
  correct and what lets generic content/observation-injection optimizers (e.g.
  AgentVigil) auto-target the surface via the post-call answer, with no
  per-experiment configuration.
- **C.4 PoT**: the plan-format **scaffolding** (`instruction_base`) is the
  target's own and is **always present**, including on the PoT path (upstream
  `pot_bkd_instruction = "".join(instruction_base + examples)`); the attacker's
  demonstration block is appended after it. The PoT **example content** (the
  per-agent backdoor examples) is attacker-supplied via the Controllable, not
  baked into the target. The trigger (default `"in light of extant
  literature"`) is appended to the user task when PoT injects.
- **C.5 attacker_tool_injection** (forcing the attacker tool into every plan
  step) is gated on the experiment-set `attacker_tool_forcing` ConfigSpec
  (Task-set, default off), the equivalent of upstream's
  `direct_prompt_injection OR observation_prompt_injection` flag (upstream
  forces for BOTH DPI and OPI). The event model cannot know at plan time
  whether OPI will fire, so forcing is an explicit experiment switch rather
  than inferred from a DPI injection; this keeps DPI and OPI consistent and a
  clean baseline (config off) never forces. PoT/MP name the tool via the
  plan/memory instead, so forcing is excluded under them, matching ASB.
- **C.6 Attacker tool registration**: registered iff the task configures one
  (`attacker_tool` non-empty), a precondition for any attack, rather than
  ASB's per-flag gating.
- **C.7 Defenses removed**: the target has **no** defense infrastructure
  (no `defense_type` config, no delimiters/instructional/ob-sandwich/paraphrase/
  pot-shuffle code). ASB's defenses are out of scope for this bare runtime.
- **C.8 Tool-catalogue editing (beyond upstream)**: a superred extension with no
  upstream analog. Upstream ASB ships a single fixed Task-set attacker tool
  (C.6); the port additionally exposes attacker-driven catalogue editing as four
  Controllables fired **once** before planning, mirroring the agentdojo and
  inspect-agent targets (same names + the add/edit/remove capability split):
  `tool_catalog_register` (add a new tool, scope `tool_catalogue_add`),
  `tool_catalog_replace` (shadow an existing tool's behavior, `tool_catalogue_edit`),
  `tool_catalog_rewrite_doc` (edit a tool's description only, `tool_catalogue_edit`),
  and `tool_catalog_unregister` (remove a tool, `tool_catalogue_remove`). An
  ADDED or behavior-REPLACED tool is **shadowed**: when the agent calls it the
  original `.run()` is **not** invoked; instead a per-call event is fired
  directly to the attacker carrying `{tool, params}`, and the attacker's response
  (or a static `fake_return`) is used as the return. That per-call event is
  tagged at the `tool_catalogue_add` / `tool_catalogue_edit` **capability**, not
  the tool's `tools.*` leaf, so an attacker holding the catalogue-edit capability
  controls the call even when the tool's own leaf is out of scope. A rewrite-doc
  edit changes only the catalogue listing the model reads (influencing
  selection), leaving behavior intact. **Clean by default**: with no attacker all
  four events decline and the catalogue is untouched. This is purely additive
  surface gated to no-op when unused; it alters no upstream attack path.

## D. Durable memory (restored)

- **D.1** ASB's vector memory store is **restored** as durable target state.
  Upstream uses `langchain_chroma.Chroma` with `OpenAIEmbeddings` and top-1
  `similarity_search_with_score`; the port reproduces the **mechanism** with a
  lightweight in-process top-1 **cosine** store (`memory_store.py`) over the
  same proxy embeddings (`text-embedding-3-small`, 1536-d), avoiding the
  heavy `chromadb`/`langchain` dependency. Embeddings are identical (same
  model, same proxy); top-1 retrieval over the small per-task corpus is
  metric-robust, so the retrieved content (hence the agent's behaviour) is
  faithful.
- **D.2** Memory is **durable**: held on the target, it survives
  `reset_ephemeral_state` (the per-run reset) and is discarded only when a
  fresh target is built per task. This lets one attacker, across multiple runs
  of one task, **write in an early run and read in a later run** (ASB's
  two-phase write-then-read).
- **D.3** Memory mode is activated by a `memory_mode` **ConfigSpec** (Task-set,
  never the attacker), replacing upstream `--read_db`/`--write_db`. Default off
  = clean run that neither reads nor writes (matching upstream's no-read_db/
  no-write_db baseline). The read retrieves the genuine top-1 record; the
  `mp_retrieved_workflow` Controllable lets an attacker substitute it (default
  = genuine). The wrapper instruction and the written `Agent/Task/Workflow/
  Tools` record are byte-faithful to upstream.
- **D.4** Divergences: (a) the store is **per task** (discarded between tasks),
  not upstream's suite-level persisted database; (b) the embedding model is
  `text-embedding-3-small` substituting upstream's `OpenAIEmbeddings` default
  (`ada-002`), same dimensionality, top-1 over a tiny corpus makes the ranking
  difference immaterial; (c) an empty store or a record without a `Workflow:`
  block **degrades gracefully** (no injection / whole-record fallback) where
  upstream would crash on an unbound variable (required by the never-crash
  directive). (d) On a memory run with an empty store and no injection the port
  appends **no** memory message, whereas upstream's `read_db` branch always
  appends an assistant message (the literal string `None` when the search
  returns nothing); the port omits that vacuous `None` turn. (e) PoT takes
  **precedence** over the memory read (as upstream's `pot_backdoor elif
  read_db` ordering does): when a PoT injection lands on a memory run the read
  is skipped (no `memory_found` scored), though the end-of-run write still
  occurs so a later run can retrieve it.

## E. Trust-boundary forest (redesigned)

- **E.1** Four root boundaries: `user` (DPI), `system` (with `system_prompt`
  for PoT and a read-only `agent_trace` subtree), `tools` (OPI), `memory` (MP).
  The earlier external/internal **provenance split is removed**, and **all
  read/write (`_readable`) tags are removed** (the Controller's native
  read-only scope grants see-but-not-inject on any tag). The `system` tree also
  carries `model_identity` (read-only knowledge of which model powers the agent,
  exposed as an observable for cross-target consistency with the agentdojo and
  inspect-agent targets; never a controllable, as the model is a construction
  concern), `detailed_system_specification` (a read-only leaked free-text
  system-specification observable, a sibling of `model_identity`), and
  `tool_catalogue` (the tool-REGISTRY capability: a grouping root
  over `tool_catalogue_add` / `tool_catalogue_edit` / `tool_catalogue_remove`,
  see C.8). `tool_catalogue` lives under `system`, deliberately SEPARATE from the
  `tools` tree, so granting read of the catalogue listing (which tools exist)
  does NOT imply reading every tool's returned observation (those are the `tools`
  tree); `tool_catalogue` is not an ancestor of any `tools.*` node.
- **E.2** The `tools` tree is **fully per-tool granular**, three levels:
  `tools` (the whole ecosystem) -> `tools.<scenario>` (one scenario's tool
  environment, ten of them) -> `tools.<scenario>.<tool>` (a single tool/service,
  twenty leaves), built mechanically from the dataset's `Corresponding Agent`
  (no hand-authored grouping). An attacker can be scoped to one tool, one
  scenario's tools, or the whole ecosystem; the upstream-faithful,
  provenance-blind OPI scope is the `{tools}` root (a faithful uniform-OPI run
  uses it). The per-tool/per-scenario nodes are a beyond-upstream additive
  refinement (ASB draws no per-tool distinction). The framework's
  `distinct_combinations` antichain enumeration is exponential in child count
  and is impractical on this granular forest; it is a standalone research
  utility never called on the run path (which uses the cheap `scope_includes`),
  so enumerate antichains on a chosen subtree, or build scopes directly.

## F. Trajectory / observables

- **F.1** Emissions are once-each, at true provenance, in causal order, with a
  strict producer split. The agent's OWN generations (the plan JSON and each
  per-step [Thinking] output) are observables under `agent_trace` (a leaf of
  `system`). A whole tool interaction (the call, its params, and the returned
  observation) is the tool's data, carried under that tool's `tools.*` leaf:
  non-final interactions as the OPI controllable event (its `request` is
  `{tool, params, observation}`, and the attacker may change the return), the
  final interaction as a `tool_interaction` observable. There is **no** separate
  `agent_trace` tool-call or tool-response record (a tool interaction belongs to
  the tool, not the agent), so a `{system}` attacker cannot read any tool's
  interaction; each interaction is emitted exactly once, under its tool. The
  earlier bulk message-list re-emission (which double-emitted injected text
  under the system subtree, a cross-domain leak) is removed. The grader still
  reads the raw transcript via the `messages` query, so scoring is unchanged.
- **F.2** No attack payload is exposed as an observable (the target only
  exposes injection points; attacks are an attacker concern). The earlier
  `asb_attack_reference__*` observables are removed.
- **F.3** Static observables (read once at init): `system_prompt` (persona),
  `model_identity` (the litellm model id; added for cross-target consistency,
  read-only), `detailed_system_specification` (a read-only leaked free-text
  system-spec brief: purpose, architecture, runtime logic, hardcoded
  prompts/code), and `tool_catalog_listing` (the registry: which tools exist + their
  descriptions, including the attacker tool once registered). The listing is
  tagged at the `tool_catalogue` registry boundary (E.1), **not** the `tools`
  tree, so reading it does not grant reading any tool's returned observation.

## G. Execution model

- **G.1 `concurrency=1` per target, but any number of targets per process**:
  ASB drives its agent through an `LLMRequestQueue` drained by a
  `FIFOScheduler` thread. Upstream both are process-wide (the queue is a CLASS
  attribute, the kernel a singleton), which makes two ASB targets in one
  process **silently wrong**: a queued request names no model, so whichever
  scheduler pops it answers with ITS kernel's model, and provider failures pile
  into one shared list that cannot say which run they belong to. Worse,
  building a second runtime for a different model STOPS the first one's
  scheduler, leaving its in-flight request with no consumer and its agent
  spinning in `listen()` forever. This port therefore moves every piece of that
  state onto an instance: `AsbRuntime` owns the queue, the scheduler thread,
  the kernel and the `ProxyConfig` (credentials, pacing, failure record), and
  each `AsbTarget` builds exactly one on first run and stops it in `teardown`.
  Several ASB Controllers can then run concurrently in one process, including
  under `superred.run_all`. `concurrency=1` still holds WITHIN a target: one
  agent per target at a time. Scheduler threads stay daemons and a process-exit
  hook stops any runtime whose owner did not.

  Five deviations from the verbatim vendored code implement this, each marked
  in place: (i) `BaseQueue` holds its queue per instance, with `default()`
  preserving the shared-queue behaviour for any caller that supplies none;
  (ii) `FIFOScheduler` takes its queue as an argument; (iii) the scheduler loop
  no longer lets an exception kill its thread, because a dead scheduler hangs
  its agent forever, which was survivable when a run owned its process and is
  not now; (iv) `FIFOScheduler.stop()` wakes the queue with a sentinel instead
  of waiting out its 1s read timeout, because a superred run tears a scheduler
  down once PER TASK and that second, paid 43,000 times, is hours of pure
  waiting (measured: 1.010s -> 0.005s per cycle); and (v) `_Kernel` in
  `runtime.py` replaces `LLMKernel`'s constructor, since `LLMKernel` resolves
  the model through the global `MODEL_REGISTRY` and passes only
  `llm_name`/`log_mode`, leaving nowhere to inject this runtime's
  `ProxyConfig`. Its dispatch body is reproduced exactly. At one runtime per
  process all five are behaviour-identical to upstream.

  **The leaked system-spec brief changed wording.** `system_specification.md`
  is not documentation: it is read at import and handed to the attacker as the
  `detailed_system_specification` observable, so its bytes are experiment
  input under any scope that includes `{system}`. Three sentences describing
  scheduler topology were corrected here, because the old ones now state the
  opposite of what the code does. They describe process structure, not an
  injection surface, so no attack strategy depends on them; but a results tree
  spanning this commit contains tasks measured against both wordings, and the
  brief is not part of the measurement identity, so nothing in the record
  distinguishes them. Disclosed rather than avoided: keeping a knowingly false
  brief was judged worse than a wording change no attacker can act on.

  **Request pacing is now per runtime, not per process.** ASB's inter-call
  delay (`request_delay_seconds`, upstream's hardcoded `time.sleep(2)`) used to
  serialize every call in the process because one scheduler served them all.
  With N runtimes the effective request rate is N times higher, so gateway rate
  limits, not the code, bound concurrency. Size `run_all(concurrency=)` against
  the provider's limits.
- **G.1a** ASB's `AgentProcessFactory` hands out pids from a pool of 10000 and
  never reclaims them on the agent path, so a long experiment with one factory
  would exhaust the pool and crash. The target builds a **fresh
  `AgentProcessFactory` per run**, keeping the per-run pid count tiny. The
  factory only mints pids; a request is served by the scheduler that owns the
  queue it was submitted to, never by whichever factory created it.
- **G.2** The model is a **construction concern** (constructor arg), not a
  config slot. Generation settings (seed 0, temperature 0, the pinned token
  cap) are fixed per experiment.
- **G.2a Target model vs optimizer model.** The target runs all inference
  through its **own** litellm `ProxyLLM` client (registered in `runtime.py`),
  never the optimizer's constrained `LLMClient` (`target.py` imports no
  `LLMClient`). Consequently the optimizer's model-lock and token budget apply
  **only to the attacker**; the target uses whatever model it is constructed
  with, and its token spend is out of band. The default is `gpt-4o-mini`: it is
  ASB's de-facto GPT model, hardcoded in upstream's memory-db path and as the
  refusal judge (`main_attacker.py`), and is verified reachable on this
  project's litellm proxy. The vendored argparse default `--llm_name
  gemma-2b-it` is **rejected**: it is an inherited AIOS placeholder for a local
  HuggingFace model (that backend is removed in this port), not ASB's
  experimental run model, whose real sweeps use GPT models.
- **G.3** Only the **automatic** planning mode is supported (the bare general
  runtime); ASB's manual workflow mode is removed.
- **G.4** `reset_ephemeral_state` (the framework's renamed per-run reset; the
  earlier port wrongly named it `cleanup`, which made the target
  non-instantiable against current framework main) clears only per-run state
  and **preserves the durable memory store** and per-task config.

## H. Out of scope for the target

- **H.1** The refusal judge and the attack-success / utility predicates live in
  `security_claim_asb`; the target only exposes the message trace + ground
  truth via queries.
