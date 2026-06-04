# Chord/XTHP Optimizer Assumptions

This module implements Chord's Cross-Tool Harvesting and Polluting attack for SuperRed agent targets.

## What Matches The Paper And Official Code

- The default attack is tool-control-flow hijacking, not memory poisoning.
- The optimizer supports Chord's predecessor and successor helper-tool variants.
- The official malicious helper tool names/descriptions are loaded from `malicious_tools.json`.
- The official sensitive-argument mappings are loaded from `malicious_tool_arguments.json`.
- The official target queries are loaded from `query.json` and are used by default when the user prompt is writable.
- Hijack success follows the official control-flow idea: predecessor helpers should appear before the victim tool; successor helpers should appear after it.
- Framework feedback is authoritative when visible. Otherwise, the optimizer reads the trajectory for tool-call order and observable responses.

## SuperRed Mapping

- Chord's scanner runs a generated malicious tool inside its own LangChain/LlamaIndex agent loop. SuperRed already owns the controller and target loop, so this optimizer injects Chord-style helper tools through writable tool-catalog surfaces instead of running Chord's scanner inside SuperRed.
- `tool_catalog_register` is the closest paper-equivalent surface: it adds the helper tool to the agent's tool pool.
- `tool_catalog_replace` and `tool_catalog_rewrite_doc` are used when those are the granted writable surfaces.
- If only user prompt or system prompt is writable, the optimizer can still exercise the granted capability, but that path is a SuperRed extension rather than Chord's mainline tool-pool setting.
- Chord's official scanner evaluates several queries inside one scanner run. SuperRed runs one target execution at a time, so this optimizer schedules those official queries across optimizer runs.

## Deliberate Limits

- The module does not vendor Chord's full LangChain/LlamaIndex runtime. That runtime would duplicate SuperRed's controller, target, trajectory, and security-domain mechanics.
- The module does not require shared memory or connected-run state. XTHP is about tool control flow and cross-tool data movement during an agent task.
- LLM-generated helper descriptions use Chord's official prompt text, but official static helper descriptions are preferred when the victim tool appears in the packaged data.
