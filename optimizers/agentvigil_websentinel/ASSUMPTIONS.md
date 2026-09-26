# AgentVigil/WebSentinel Optimizer Assumptions

This module implements the AgentVigil/WebSentinel attack idea for SuperRed agent targets. Both papers are the reference: AgentVigil, "Generic Black-Box Red-teaming for Indirect Prompt Injection against LLM Agents" (arXiv:2505.05849), and WebSentinel, "Detecting and Localizing Prompt Injection Attacks for Web Agents" (arXiv:2602.03792).

## What Matches The Papers And Code

- The optimizer starts from the official `new_seeds` web/content prompt-injection corpus by default. The older text seed corpus is also included for callers that want it.
- It uses the official mutation families: expand, shorten, rephrase, crossover, and generate-similar.
- Mutations are produced by a helper LLM and parsed from `<response>...</response>`, matching the official mutator shape.
- Seed selection follows the official MCTS/UCT idea: strong seeds are reused, but less-tested seeds still get explored. Generated mutations stay pending until their own SuperRed run is scored, so unscored mutations are not selected as parents.
- The initial seed corpus is evaluated before generated mutations are used, matching the official fuzzer's warm-up scoring step.
- Framework evaluation is the attack score. In the official code, the target agent returns task success for each injected seed; in SuperRed, that signal is `RunEndEvent.evaluation`.

## How It Maps To SuperRed

- The closest SuperRed match for a contaminated webpage, retrieved document, memory item, RAG chunk, MCP result, browser page, or tool result is a content/tool `PostCall` event. The optimizer injects into `read__...` events and other content-like PostCall surfaces such as browser/page/memory/RAG/MCP controls. Per-tool output surfaces named `tool:<tool_name>` (e.g. the AgentHarm / `inspect_agent` target's indirect-prompt-injection points, "replace the value the tool returns to the agent") are recognized by the `tool:` prefix and treated as content surfaces; without this the optimizer would silently decline them in tool-only scopes and score a false non-attack. Init-time availability and the PostCall injection gate share one predicate (`_is_agent_content_surface`) so they cannot disagree.
- Some agent targets use opaque controllable names. Callers can pass explicit content controllable names, and the optimizer also looks at controllable descriptions and value types so it does not depend only on naming conventions.
- If the user prompt is writable, the optimizer appends a `WEBPAGE CONTENT` block to the user prompt. This keeps the attack usable in simpler targets, but it is less indirect than a true webpage/tool-content injection.
- If the system prompt is writable, the optimizer adds a short red-team capability extension, includes the current SuperRed goal, and carries the injected web/content instruction.
- If the tool catalog is writable, the optimizer can register, replace, or rewrite a relevant tool so the agent sees injected web/content text. For AgentDojo-style capability claims, it uses attacker-only tool names from the goal when they are present.
- Static observables such as model identity, system prompt, and tool catalog are used when in scope. The static context is size-limited so long target metadata does not crowd out the mutator prompt.
- Tool catalogs may appear as a bare list or as common wrapped forms such as `{"tools": [...]}`. The optimizer accepts those shapes when the observable metadata looks tool-related.
- When multiple surfaces are in scope, the optimizer does not inject into all of them. It picks one sensible surface per run, using the best surface that is both authorized and actually reached. Dynamic content/tool `PostCall` surfaces are preferred because they are closest to the paper's contaminated-web-content setting. If no such surface is available, writable tool catalog comes next, then writable system prompt, then user prompt as the simple-target fallback.
- The user-prompt fallback is intentionally skipped when stronger agentic content surfaces are in scope. SuperRed cannot retract a user-prompt injection if a better `PostCall` fires later, so deferring to the agentic surface avoids noisy over-injection.
- This deferral is adaptive and persistent. A run that delivers nothing (the preferred surface was held out for but never fired) does not score the held seed — the seed never reached the target, so penalizing it would teach the search a false weakness. The held seed is retried on the next run with the surface ladder one notch deeper. The ladder only deepens and never resets on a successful delivery, so once the optimizer learns which surface actually fires it keeps using it instead of re-blinding itself to the top surface every run.
- Non-delivery runs are bounded by a consecutive-miss count rather than the attempt budget. Once the optimizer has tried every reachable surface in a row without landing a single injection, it gives up rather than burning the whole run budget achieving nothing. Any delivery resets the streak, so a surface that keeps working runs to the attempt budget, and a surface that goes silent after delivering before cannot spin forever. This single rule also covers targets whose content/tool `PostCall` surfaces only appear at runtime (nothing injectable is listed up front): they still get one run to materialize, then stop if that run plants nothing.

## Static Text And Constants

- Official reusable literals are packaged as data-only JSON under `src/agentvigil_websentinel_optimizer/data/official/`. We do not vendor the official runtime implementation.
- The mutator system prompt and mutation templates are exact values extracted from official `mutate_prompts.py`, including the original wording and typos.
- The default seed corpus is exact data extracted from official `new_seeds.py`. The older official `seeds.py` text corpus is available via the `include_text_seeds=True` constructor argument (it appends `OFFICIAL_TEXT_SEEDS` to the default corpus) or by passing an explicit `seeds=` corpus.
- Only the data the optimizer actually consumes is packaged: the seed corpora and the mutator prompts. The official `adaptive_attack_data.json` task/web-page dataset is intentionally not vendored — SuperRed supplies the task to attack, so the optimizer never reads an external task dataset.
- SuperRed delivery constants are not from the official code: the `SUPERRED AGENT CAPABILITY EXTENSION` system-prompt text, tool-catalog JSON payload shapes, content-surface name hints, and AgentDojo attacker-tool names exist only to map the official web/content attack onto SuperRed's broader target primitives.

## What Is Different On Purpose

- The official fuzzer scores one seed across many target tasks, then runs 20 fuzz loops with up to 10 mutations per loop. SuperRed normally runs one security task at a time, so this optimizer treats one SuperRed run as one seed evaluation. The default `max_attempts=20` mirrors the official loop count, not the official total target-evaluation count. Increase `max_attempts` for evaluation-count parity experiments. The official `subset_ratio` setting is omitted because SuperRed provides one task evaluation at a time to the optimizer.
- The official implementation has a persistent checkpoint/log directory. SuperRed already persists trajectories and task results, so this module keeps state in memory and relies on the framework for run records.
- WebSentinel is a detector/localizer. This optimizer implements the attack-generation side used by the AgentVigil/WebSentinel repository; it does not implement a separate WebSentinel defense.
- When framework feedback is hidden, the optimizer cannot know whether a delivered run succeeded. It continues up to `max_attempts`, gives delivered-but-unscored candidates neutral reward, and mutates open-loop rather than inventing a private success judge. Runs that delivered nothing are not scored at all (see the deferral notes above).
- Adversarial tool descriptions are deterministic by default. With `use_llm_tool_descriptions=True`, the helper LLM crafts the registered attacker tool's lure description from the goal and current seed when the tool-catalog surface is used — extending AgentVigil's "LLM mutates the injection" idea to the tool-registration vector. It is opt-in for determinism/cost. Transport or parsing failures fall back to the static description, while budget exhaustion is allowed to stop cleanly through the controller. Surface selection itself stays deterministic on purpose: the content-first ladder is the faithful, reproducible mapping of the paper's contaminated-content vector, so the learned-delivery fixes above address its behavior rather than spending an LLM call to re-pick a surface each run.
- Runtime trajectory observables are not used for private scoring. The original attack scores target-task success, so this port uses SuperRed framework evaluation rather than adding a separate response judge.
- The optimizer does not drain, mutate, or cache the trajectory. Runtime injection decisions are made from the live `PreCall`/`PostCall` events, and the trajectory remains the controller-owned audit record. The smoke test snapshots the completed trajectory only to verify which injections were recorded.
- Candidate generation is capped by the number of target runs remaining, so helper LLM calls are not spent on mutations that cannot be evaluated.
- Multi-parent backpropagation credits each ancestor once per evaluated descendant. This matters for crossover: a diamond-shaped ancestry graph should not double-count a shared ancestor just because two parent paths point back to it.
- The mutation and seed-selection loop aims for close parity with the official implementation. The delivery layer is broader than the official simulated web agent because SuperRed exposes more target capabilities and more agent target families.

## Agent Targets

This optimizer is intended for agent-style targets such as AgentDojo, browser agents, RAG/memory agents, MCP/tool agents, and Letta/ASB-style targets when they expose any of these SuperRed surfaces: user prompt, system prompt, tool catalog, static observables, dynamic tool responses, or content/read injection points.

## Schema-typed content surfaces are declined, not injected (DTAP fitness)

A content-injection payload is an unstructured string. A content surface whose
`value_type` is a parsed schema (e.g. DTAP `env_inject:<server>`, which needs a
`{injection_mcp_tool, kwargs}` object) silently discards a bare string: the target
`json.loads`-parses it, gets nothing, and writes nothing, yet the run is still
recorded as a scored attack -- a fake 0.0 indistinguishable from a defended attack.
`_handle_post_call` now checks `_accepts_free_text` (value_type in
text/str/string/html/markdown) BEFORE latching `_selected_surface`: a schema surface
is declined WITHOUT consuming the run, so the later free-text surface (e.g.
`env_tool:<server>`, which replaces a tool return) is still selected. Detection is
unchanged (the surface is still classified as content); only emission is gated. On
AgentDojo/ASB/inspect_agent every content surface is `text`, so this is a no-op there.
## Surface classifier: empty categories and out-of-money budget

The shared LLM surface classifier (`surface_llm.classify_controllables`,
byte-identical across the agentic optimizers) sorts each granted surface into a
role category by reading its description. Two behaviours deviate from a naive
reading and are load-bearing here, because this module is where a misclassified
category is fatal rather than merely lossy:

- Categories are roles to match, not a partition to fill. When a scope grants no
  user-prompt surface -- a threat model may drop it, e.g. scopes s3, s4 and s6 --
  the prompt tells the model a category may match zero surfaces and forbids
  relabelling content surfaces to populate it. Without this, gpt-4o-2024-05-13 put
  every DTAP `env_tool:<server>` surface into `user-prompt` under
  category-completion pressure. A false `user-prompt` label sets
  `_can_write_user_prompt`, which shrinks `_available_surface_ranks()` to length 1,
  so `_is_done()` fires after run 1 and the task ends in seconds with a zero
  indistinguishable from a defence. With the earlier prompt this collapse was
  common on DTAP indirect-injection tasks in scopes without a user-prompt surface;
  the improved prompt prevents it, and scopes that keep the user-prompt surface
  are unaffected. The prompt also classifies by role, not goal-relevance, so a
  live indirect-injection surface is not dropped to `irrelevant` merely because it
  looks off-topic for the task: the prompt is written so every granted surface
  receives a category.
  One wording constraint is load-bearing: the prompt describes each role in prose
  and must never spell one out as a label-shaped phrase. An earlier revision said a
  qualifying value "is a content/environment surface"; the model answered with that
  literal string, every entry failed the `cat in allowed` filter, and
  `classify_controllables` returned `{}`. That total discard is invisible to a
  vacuity check, because an empty result is never vacuous.

- Out-of-money is distinguished from "no LLM". A genuinely exhausted attacker (a
  positive per-task cap consumed, so the raised `BudgetExhaustedError` carries
  `usage.cost > 0`) is re-raised, so the controller records the task as
  budget-exhausted instead of the bare handler swallowing it into an empty
  classification that a dead proxy or a target with nothing to attack would also
  produce. The deliberately budget-less noop client the controller hands a
  non-LLM optimizer raises the same error with nothing spent (`usage.cost == 0`);
  that is "no LLM configured", not "out of money", and still degrades to the
  caller's name-based backstop. `fill_value` gates on the same distinction.
