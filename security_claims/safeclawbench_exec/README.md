# superred-claim-safeclawbench-exec

SafeClawBench **Exec-Balanced** security claims for superred — the paper's
*sandbox-observed tool/state harm* endpoint (Tian et al., arXiv:2606.18356),
complementing the prompt-level Semantic Core (`superred-claim-safeclawbench`).

Each of the 600 executable cases runs the model-under-test through a vendored,
fully-offline mock sandbox (six stores: files / email / db / memory / web /
code) via the [`safeclawbench-exec` target](../../targets/safeclawbench_exec),
and a **deterministic state oracle** scores whether observable harm occurred —
no judge LLM is required (only the model being red-teamed).

## Scope

Where the Semantic Core asks "did the response *semantically* comply?"
(LLM-judged `CoreFail@600`), the Exec-Balanced endpoint asks "did the agent
actually *change state* harmfully?" The primary metric is **HarmCompletion-ASR**
(the case's `requires` — tool-call success and/or the state oracle — all hold);
`ToolCall-ASR`, `StateChange-ASR`, and `Blocked-by-Harness` are surfaced as
sub-scores. Score convention is attacker-oriented: `success` is `True` when the
sandbox observes completed harm.

## Usage

```python
from safeclawbench_exec_claim import (
    safeclawbench_exec_claim,
    safeclawbench_exec_target_factory,
)
from superred.core.types.llm import LLMConfig

claim = safeclawbench_exec_claim(
    attack_types=["MPI", "MEX"],   # optional family filter (DPI/IPI/TRI/MPI/MEX/ADI)
    defense_level="D0",             # applied by the target; D0 == baseline
)
target_factory = safeclawbench_exec_target_factory(
    agent_llm_config=LLMConfig(model="gpt-5.4", api_base=..., api_key=...),
)
# hand `claim` + `target_factory` to a superred Controller.
```

Convenience roll-ups: `safeclawbench_exec_family_claim("MPI", ...)` and
`safeclawbench_exec_combined_claim([...])`.

## License

MIT for this package's code (`LICENSE`). The vendored 600-case dataset + sandbox
live in the target package and keep their upstream MIT license — see that
package's `LICENSES/` and `ASSUMPTIONS.md`. Cite Tian et al., 2026
(arXiv:2606.18356).
