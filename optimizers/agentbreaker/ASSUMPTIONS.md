# Assumptions and deviations

Provenance and every deliberate departure from the reference implementation.

## Upstream

| | |
| --- | --- |
| Project | [NVIDIA garak](https://github.com/NVIDIA/garak) |
| Code | [`garak/probes/agent_breaker.py`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/agent_breaker.py) |
| Data | [`garak/data/agent_breaker/prompts.yaml`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/data/agent_breaker/prompts.yaml) |
| Commit | `2233d51bedf6bfb634872ef9efbcd94eae38cc1a` |
| Licence | Apache-2.0 |
| Risk | [OWASP LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/) |

## Copied byte-for-byte

- `data/upstream/prompts.yaml` — the whole upstream file. `scripts/sync_upstream.py
  --check` re-downloads and fails on drift.
- `analysis_prompt` / `exploit_prompt` — upstream's `.format()` calls, same
  named fields (exploit uses `target_tool`, `functionality`, `vulnerabilities`,
  `exploit_strategies`, `history_str`).
- `format_attack_prompt` — upstream `_format_attack_prompt`, verbatim.
- `parse_analysis` / `build_tool_plans` — upstream `_analyze_attackable_tools`
  JSON handling and `_build_tool_configs` priority ordering (priority entries by
  `"tool - why"`, substring match, then remaining tools in analysis order).
- Attack sequence per tool — upstream's order: the analysis's `attack_prompts`
  first, then feedback-driven `EXPLOIT` generations, up to
  `max_attempts_per_tool` (upstream default 5).

## Deviations

### 1. Tools come from observables; discovery is dropped

Upstream, when its YAML has no tools, sends `DISCOVERY_*` prompts to the target
agent to ask what tools it has, then `PARSE_*` prompts to structure the reply.
SuperRed targets already expose the tool catalogue as an **observable**
(`tool_catalog_listing` on ASB, `tool_list` on OpenClaw, …) and the purpose as
a system observable, so the whole discovery/parse phase is redundant and not
ported. The `DISCOVERY_*` / `PARSE_*` keys remain in the vendored
`prompts.yaml` (kept byte-identical) but are unused.

### 2. The task goal is folded into the analysis

Upstream has no external objective — it exploits tools generically. A superred
`Optimizer` pursues a `Goal`, so the goal is appended to the agent-purpose text
passed into the `ANALYSIS` template's `{agent_purpose}` slot (as
`ATTACKER OBJECTIVE: <goal>`). The byte-faithful template is unchanged; only
the value placed in its slot is enriched.

### 3. Verification is dropped

Upstream's `VERIFY` phase runs a judge LLM to decide success. In superred,
judging belongs to the `SecurityClaim`, so `VERIFY` is not ported (its key
stays in the vendored file, unused). Success is whatever the claim decides.

### 4. One attack per run; feedback across runs

Upstream loops attempts against one generator, resetting per tool. Here each
attempt is one run: the optimizer injects once, reads the agent's reply via the
`ControllablePostCallEvent`, and stores `(prompt, response)` so the next run's
`EXPLOIT` call sees the history (upstream's last-5-attempts window). Both turns
of a tool stay on the surface that received the first injection.

### 5. No separate red-team model or NIM plumbing

Upstream loads its own red-team and parse models (`nim`, `openai/gpt-oss-120b`
by default) via garak's plugin system. Superred provides exactly one attacker
`LLMClient`, budget-tracked and scope-locked, so all analysis and exploit calls
go through `self.llm`. The `red_team_model_*` / `parse_model` configuration has
no analogue and is not ported.

### Injection-surface selection (superred adaptation)

Upstream aims a single prompt at a chat box; superred targets expose arbitrary
named surfaces. Following the house pattern used by `muzzle`, `chord_xthp`,
`eia_agent`, `gepa_agentic`, `poisonedrag` and `agentvigil_websentinel`, this
module ships a byte-identical copy of the shared `surface_llm.py` and calls
`classify_controllables` once at `initialize`: the attacker's own LLM reads
each surface's description and says which one is the user's prompt, so the
payload is not spent on a content surface that merely fired first.

The classifier returns `{}` on any failure (no budget, malformed reply), and a
surface the target raises at run time without listing it is unknown to the
classification; both fall back to the previous name/value-type backstop, so
behaviour is unchanged when no LLM is available.

Note: the shared `surface_llm.py` pins `temperature=0.0`, which the repo's
`test_no_temperature.py` guard forbids. The six merged modules that ship the
classifier resolve this by not carrying the guard at all; this module keeps the
guard and skips only that one pinned shared file, so the guard still covers all
first-party code here.

## Upstream coverage

`garak/probes/agent_breaker.py` defines one probe (`AgentBreaker`) plus its
`AttackState` helper. The probe is ported whole: the ANALYSIS -> EXPLOIT
feedback loop, the vendored `prompts.yaml`, the tool-plan construction
(`_build_tool_configs`, including the priority-target ordering and the
deduplicated remainder) and the per-tool attempt cap.

Upstream marks it `active = False` ("Requires red team model configuration");
here the attacker LLM is always configured by the framework, so it is simply
available.

## Known limitations

- Needs an agentic target that exposes a tool catalogue observable. Against a
  bare chatbot it finds no tools and declines every injection.
- Spends attacker inference: one analysis call per task, then up to
  `max_attempts_per_tool` exploit calls per tool once seeds are spent. Use
  `max_tools` and `max_attempts_per_tool` to bound cost.
- Tool/purpose observable names are matched from a fixed candidate set; a target
  using a different name for its catalogue would need that name added.
- Faithfulness is pinned to the prompt catalogue and the control-flow, not to
  upstream's exact prompts-per-tool counts, which depend on the red-team model's
  analysis output at runtime.
