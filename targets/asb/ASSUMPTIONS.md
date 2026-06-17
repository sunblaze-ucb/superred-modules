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
  transcript. Transient rate-limit errors stay tolerated (ASB-faithful).

## C. Injection model (the core adaptation)

- **C.1** ASB selects an attack method via argparse flags and self-injects
  fixed strings. The port instead fires a superred `ControllablePreCallEvent`
  at each of the four injection sites and applies the attacker's injection (or
  nothing). **Clean by default**: with no attacker every site declines and the
  run is benign.
- **C.2 DPI**: the injection is appended to the benign `task_input`
  (`task_input += " " + value`, matching ASB's `+=`).
- **C.3 OPI**: fired on **every** non-final tool return **including the
  attacker tool's own observation**: the `function_name != self.tool_name`
  guard the earlier port added is **removed**, restoring upstream's
  provenance/name-blind behaviour (`react_agent_attack.py:188-189`). Each OPI
  event is tagged to the firing tool's scenario sub-boundary (E).
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
  read-only scope grants see-but-not-inject on any tag). `model_identity` and
  `tool_catalog` tags are removed (model is construction-only; the attacker
  tool is Task setup, not an attacker-controlled write).
- **E.2** `tools` has one **mechanical** sub-boundary per scenario
  (`tools.<scenario>`, ten of them; each scenario's two tools share its tag,
  derived from the dataset's `Corresponding Agent`). The upstream-faithful,
  provenance-blind OPI scope is the `{tools}` root. Per-tool leaves were
  **deliberately not used**: the framework's `distinct_combinations` antichain
  enumeration is exponential in a node's child count, so 20 sibling leaves are
  infeasible (~10^6 antichains) while ten scenario leaves are cheap. This is a
  beyond-upstream additive refinement (ASB draws no per-tool distinction).

## F. Trajectory / observables

- **F.1** The agent's genuine generations are emitted **once each at their true
  provenance and in causal order** (plan JSON, per-step model output, each
  executed tool call under `agent_trace`; the final tool return as an
  observable; memory read/write at the memory boundary). The earlier bulk
  re-emission of the whole message list (which double-emitted injected text and
  re-tagged it onto the system subtree, a cross-domain leak) is **removed**.
  Non-final tool returns are not re-emitted as observables; they are on the
  trajectory via their OPI controllable event. The grader still reads the raw
  transcript via the `messages` query, so scoring is unchanged.
- **F.2** No attack payload is exposed as an observable (the target only
  exposes injection points; attacks are an attacker concern). The earlier
  `asb_attack_reference__*` observables are removed.

## G. Execution model

- **G.1 `concurrency=1`, single Controller per process**: ASB uses a
  process-global `LLMRequestQueue` drained by one `FIFOScheduler` thread, plus
  other process globals (the singleton kernel/scheduler `_RUNTIME` and the
  proxy `PROXY_CONFIG`). `concurrency=1` serializes tasks within one Controller,
  and only ONE ASB Controller may run per process: sweep multiple ASB threat
  models sequentially, not via a concurrent `asyncio.gather` of ASB Controllers
  (they would race on the shared globals). The scheduler thread is a daemon,
  stopped at process exit.
- **G.1a** ASB's `AgentProcessFactory` hands out pids from a pool of 10000 and
  never reclaims them on the agent path, so a long experiment with one factory
  would exhaust the pool and crash. The target builds a **fresh
  `AgentProcessFactory` per run** (requests still flow through the global
  queue), keeping the per-run pid count tiny.
- **G.2** The model is a **construction concern** (constructor arg), not a
  config slot. Generation settings (seed 0, temperature 0, the pinned token
  cap) are fixed per experiment.
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
