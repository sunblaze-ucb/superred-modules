# AgentVigil/WebSentinel Optimizer Assumptions

This module implements the AgentVigil/WebSentinel attack idea for SuperRed agent targets.

## What Matches The Papers And Code

- The optimizer starts from the official `new_seeds` web/content prompt-injection corpus by default. The older text seed corpus is also included for callers that want it.
- It uses the official mutation families: expand, shorten, rephrase, crossover, and generate-similar.
- Mutations are produced by a helper LLM and parsed from `<response>...</response>`, matching the official mutator shape.
- Seed selection follows the official MCTS/UCT idea: strong seeds are reused, but less-tested seeds still get explored. Generated mutations stay pending until their own SuperRed run is scored, so unscored mutations are not selected as parents.
- The initial seed corpus is evaluated before generated mutations are used, matching the official fuzzer's warm-up scoring step.
- Framework evaluation is the attack score. In the official code, the target agent returns task success for each injected seed; in SuperRed, that signal is `RunEndEvent.evaluation`.

## How It Maps To SuperRed

- The closest SuperRed match for a contaminated webpage, retrieved document, memory item, RAG chunk, MCP result, browser page, or tool result is a content/tool `PostCall` event. The optimizer injects into `read__...` events and other content-like PostCall surfaces such as browser/page/memory/RAG/MCP controls.
- Some agent targets use opaque controllable names. Callers can pass explicit content controllable names, and the optimizer also looks at controllable descriptions and value types so it does not depend only on naming conventions.
- If the user prompt is writable, the optimizer appends a `WEBPAGE CONTENT` block to the user prompt. This keeps the attack usable in simpler targets, but it is less indirect than a true webpage/tool-content injection.
- If the system prompt is writable, the optimizer adds a short red-team capability extension, includes the current SuperRed goal, and carries the injected web/content instruction.
- If the tool catalog is writable, the optimizer can register, replace, or rewrite a relevant tool so the agent sees injected web/content text. For AgentDojo-style capability claims, it uses attacker-only tool names from the goal when they are present.
- Static observables such as model identity, system prompt, and tool catalog are used when in scope. The static context is size-limited so long target metadata does not crowd out the mutator prompt.
- When multiple surfaces are in scope, the optimizer does not inject into all of them. It picks one sensible surface per run, using the best surface that is both authorized and actually reached. Dynamic content/tool `PostCall` surfaces are preferred because they are closest to the paper's contaminated-web-content setting. If no such surface is available, writable tool catalog comes next, then writable system prompt, then user prompt as the simple-target fallback.
- The user-prompt fallback is intentionally skipped when stronger agentic content surfaces are in scope. SuperRed cannot retract a user-prompt injection if a better `PostCall` fires later, so deferring to the agentic surface avoids noisy over-injection.
- This deferral is adaptive. If a higher-priority surface is advertised but does not actually fire in a run, the next run may fall back to the next lower-priority surface rather than staying blind forever.

## Static Text And Constants

- The mutator system prompt and mutation templates match the official `mutate_prompts.py` text, including the original wording and typos.
- The default seed corpus is the official `new_seeds.py` web/content corpus. The older official text seed corpus is available separately for callers that want it.
- SuperRed delivery constants are not from the official code: the `SUPERRED AGENT CAPABILITY EXTENSION` system-prompt text, tool-catalog JSON payload shapes, content-surface name hints, and AgentDojo attacker-tool names exist only to map the official web/content attack onto SuperRed's broader target primitives.

## What Is Different On Purpose

- The official fuzzer scores one seed across many target tasks, then runs 20 fuzz loops with up to 10 mutations per loop. SuperRed normally runs one security task at a time, so this optimizer treats one SuperRed run as one seed evaluation. The default `max_attempts=20` mirrors the official loop count, not the official total target-evaluation count. Increase `max_attempts` for evaluation-count parity experiments. The official `subset_ratio` setting is omitted because SuperRed provides one task evaluation at a time to the optimizer.
- The official implementation has a persistent checkpoint/log directory. SuperRed already persists trajectories and task results, so this module keeps state in memory and relies on the framework for run records.
- WebSentinel is a detector/localizer. This optimizer implements the attack-generation side used by the AgentVigil/WebSentinel repository; it does not implement a separate WebSentinel defense.
- When framework feedback is hidden, the optimizer cannot know whether a run succeeded. It continues up to `max_attempts`, gives tried candidates neutral reward, and mutates open-loop rather than inventing a private success judge.
- Runtime trajectory observables are not used for private scoring. The original attack scores target-task success, so this port uses SuperRed framework evaluation rather than adding a separate response judge.
- The optimizer does not drain, mutate, or cache the trajectory. Runtime injection decisions are made from the live `PreCall`/`PostCall` events, and the trajectory remains the controller-owned audit record. The smoke test snapshots the completed trajectory only to verify which injections were recorded.
- Candidate generation is capped by the number of target runs remaining, so helper LLM calls are not spent on mutations that cannot be evaluated.
- Multi-parent backpropagation credits each ancestor once per evaluated descendant. This matters for crossover: a diamond-shaped ancestry graph should not double-count a shared ancestor just because two parent paths point back to it.
- The mutation and seed-selection loop aims for close parity with the official implementation. The delivery layer is broader than the official simulated web agent because SuperRed exposes more target capabilities and more agent target families.

## Agent Targets

This optimizer is intended for agent-style targets such as AgentDojo, browser agents, RAG/memory agents, MCP/tool agents, and Letta/ASB-style targets when they expose any of these SuperRed surfaces: user prompt, system prompt, tool catalog, static observables, dynamic tool responses, or content/read injection points.
