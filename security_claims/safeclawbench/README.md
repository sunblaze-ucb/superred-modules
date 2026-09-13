# safeclawbench-claim

SafeClawBench **Semantic Core** security claims for the superred OpenClaw target.

This package grounds the OpenClaw security claims in a real, published benchmark
rather than hand-rolled heuristics.

## Provenance

- **Paper:** *SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox
  Harm in Tool-Using LLM Agents* (arXiv `2606.18356`).
- **Dataset:** <https://huggingface.co/datasets/sairights/safeclawbench> (MIT).
  The 600-case Semantic Core challenge set (`benchmark_v5_600.json`) is vendored
  verbatim under `src/safeclawbench_claim/data/`, alongside the upstream
  `DATASET_LICENSE` and `CITATION.cff`.

The benchmark has 600 controlled cases, 100 in each of six reporting families:
DPI (direct prompt injection), IPI (indirect), TRI (tool-return injection), MPI
(memory poisoning), MEX (memory extraction), and ADI (ambiguity-driven unsafe
inference).

## Scope

This package wires the **Semantic Core** endpoint only — the prompt-level
benchmark whose primary metric is **CoreFail@600**: an LLM judge's binary label
for whether the agent's final response *semantically* complies with the attack
goal (paper §3.5/§3.7). The judge rubric (`judge.py:JUDGE_PROMPT`) reproduces
upstream `evaluator/judge.py` verbatim and also returns the Core-gated
`harm_evidence_category` (TextHarm/AccessHarm/ActionHarm/PersistHarm vs
SemanticOnly) and a 1–5 safety score.

The **Exec-Balanced** sandbox endpoint (deterministic state oracles over
isolated file/email/db/memory/web/code stores) is a separate containerized
protocol and is intentionally **out of scope** here.

## How it maps onto the target

- `configure_target` installs the fixed six-tool agent system prompt
  (`AGENT_SYSTEM_PROMPT`, verbatim from upstream `run_benchmark.py`) via the
  target's `system_prompt_append`, optionally prefixed by a prompt-level defense
  policy (default: none == the paper's D0 baseline).
- the case `user_prompt` is the attack payload, delivered through the target's
  `user_message` controllable (carried on the task `Goal`) — so a passthrough
  optimizer reproduces the upstream baseline and an attacker optimizer can
  augment it.
- `evaluate` reads the agent's final response and runs the CoreFail judge.
  Score convention is attacker-oriented: `success` is `True` when CoreFail fires.

## Usage

```python
from safeclawbench_claim import safeclawbench_claim, safeclawbench_target_factory
from superred.core.types.llm import LLMConfig

# CoreFail is an LLM-judged endpoint, so a judge config is required.
claim = safeclawbench_claim(
    attack_types=["DPI", "IPI", "TRI"],   # optional family filter
    difficulties=["hard"],                 # optional
    judge_llm_config=LLMConfig(model="gpt-5.4", api_base=..., api_key=...),
)
target_factory = safeclawbench_target_factory()
# hand `claim` + `target_factory` to a superred Controller.
```

Convenience roll-ups: `safeclawbench_family_claim("MEX", ...)` and
`safeclawbench_combined_claim([...])`. For offline tests, inject a
`StubCoreFailJudge` instead of a `judge_llm_config`.

## License

MIT for this package's code (`LICENSE`). The vendored `benchmark_v5_600.json`
dataset keeps its own upstream MIT license and citation — see
`src/safeclawbench_claim/data/DATASET_LICENSE` and `CITATION.cff`.
