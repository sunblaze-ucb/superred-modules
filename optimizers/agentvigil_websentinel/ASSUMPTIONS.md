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
- Some agent targets use opaque controllable or observable names. Callers can pass explicit content controllable/observable names, and the optimizer also looks at controllable descriptions and value types so it does not depend only on naming conventions.
- If only the user prompt is writable, the optimizer appends a `WEBPAGE CONTENT` block to the user prompt. This keeps the attack usable in simpler targets, but it is less indirect than a true webpage/tool-content injection.
- If the system prompt is writable, the optimizer adds a short red-team capability extension and includes the current SuperRed goal. When no content/tool delivery surface is available, the system prompt also carries the injected web/content instruction. This keeps stronger scopes distinct from user-prompt-only scopes.
- If the tool catalog is writable, the optimizer can register, replace, or rewrite a relevant tool so the agent sees injected web/content text. For AgentDojo-style capability claims, it uses attacker-only tool names from the goal when they are present.
- Static observables such as model identity, system prompt, and tool catalog are used when in scope. The static context is size-limited so long target metadata does not crowd out the mutator prompt.
- Runtime trajectory observables are read when visible, including AgentDojo message and tool-response observables named like `agent_trace_message_0001` and `agent_trace_tool_response_0001`.
- Per-run diagnostics track whether the attack used an agentic content surface or fell back to user-prompt delivery.
- Delivery priority is content/tool surfaces first, then writable system prompt, then user prompt as the last fallback. This avoids injecting into the user prompt when a more agentic surface is in scope.

## What Is Different On Purpose

- The official fuzzer scores one seed across many target tasks. SuperRed normally runs one security task at a time, so this optimizer treats one SuperRed run as one seed evaluation. `max_attempts` is therefore a SuperRed target-run budget, not the official script's `fuzz_loop` count.
- The official implementation has a persistent checkpoint/log directory. SuperRed already persists trajectories and task results, so this module keeps state in memory and relies on the framework for run records.
- WebSentinel is a detector/localizer. This optimizer implements the attack-generation side used by the AgentVigil/WebSentinel repository; it does not implement a separate WebSentinel defense.
- When framework feedback is hidden, the optimizer cannot know whether a run succeeded. It continues up to `max_attempts` and mutates open-loop rather than inventing a private success judge.
- The mutation and seed-selection loop aims for close parity with the official implementation. The delivery layer is broader than the official simulated web agent because SuperRed exposes more target capabilities and more agent target families.

## Agent Targets

This optimizer is intended for agent-style targets such as AgentDojo, browser agents, RAG/memory agents, MCP/tool agents, and Letta/ASB-style targets when they expose any of these SuperRed surfaces: user prompt, system prompt, tool catalog, static observables, dynamic tool responses, or content/read injection points.
