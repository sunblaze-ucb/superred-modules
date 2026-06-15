# PoisonedRAG Optimizer Assumptions

## Original Attack Shape

PoisonedRAG is a RAG knowledge-base corruption attack. The attacker injects a
small number of malicious texts into a retrieval corpus so that, when the target
question is asked, those texts are retrieved and the generator outputs an
attacker-chosen incorrect answer.

The defaults follow the released code where they map to SuperRed:

- `attack_method="LM_targeted"` is the default path.
- `adv_per_query=5`, matching the five poison texts used per target query.
- `top_k=5`, matching the generator context size in the main experiment.
- Success is the released `clean_str(incorrect_answer) in clean_str(response)`
  check when framework feedback is not visible.
- Each poison document is the official black-box adversarial text:
  `question + "." + corpus` (released `src/attack.py` `get_attack`,
  `LM_targeted` path). The question prefix is the retrieval-optimizing "S"
  component that makes the poison rank for the target query; the LLM-crafted
  corpus is the generation-steering "I" component. The prefix is applied to
  both generated corpuses and caller-supplied `poison_texts`, since the released
  code stores the corpus and prepends the question at attack time.
- The official RAG wrapper prompt and the current (JSON, multi-corpus) joint
  generation prompt are preserved in `prompts.py`. The older single-corpus
  `ADV_PROMPT` templates are not used by this optimizer (the released code now
  recommends the joint prompt) and are not vendored.

## Mapping To SuperRed

SuperRed owns the target run, retrieval system, security scope, and trajectory.
This optimizer therefore does not run the official BEIR/Contriever experiment
loop inside SuperRed. Instead, it injects PoisonedRAG documents into whichever
writable RAG surface the target exposes.

The optimizer treats names containing `knowledge_base`, `vector_store`,
`document`, `corpus`, `retrieved_context`, `rag_context`, or `context` as RAG
poison surfaces. JSON-like surfaces receive a payload with the attack name,
question, target answer, and poison documents. Text context surfaces receive the
poison documents as retrieved context text.

When a user-like controllable is writable, the optimizer asks the official target
question there. When `system_prompt` is writable, it adds a short instruction to
prefer retrieved context answers. That system-prompt path is a SuperRed
capability extension; the paper-mainline attack is still the corpus/context
poisoning path.

If no RAG corpus or retrieved-context surface is writable but the user prompt is
writable, the optimizer falls back to the official PoisonedRAG RAG wrapper and
places the poison texts in the `Contexts:` block. This is not database
corruption, but it is the closest user-only SuperRed capability mapping and
keeps the same question/context/answer format used by the released code.

Static observables can provide `correct_answer`, `target_answer`, or related
answer hints. This helps the optimizer generate poison text without requiring a
separate benchmark file.

## Feedback And Success

Framework feedback from `RunEndEvent.evaluation` is authoritative when visible.
A failed framework evaluation is not overridden by local substring scoring.

If framework feedback is not visible, the optimizer reads response observables
from the filtered trajectory first, then falls back to a paired post-call answer
when that is in scope. If neither response nor feedback is visible, the optimizer
runs open-loop and stops only by attempt budget.

If no poison or prompt was actually injected and no feedback/response is visible,
the optimizer does not spend an attempt. This avoids penalizing a candidate that
never reached the target.

If no injectable surface is granted at all (no corpus/context, system-prompt, or
user-prompt controllable), the attack can never be delivered, so the optimizer
gives up immediately rather than spinning runs that inject nothing.

## Deliberate Limits

The official HotFlip white-box path requires direct access to retriever model
weights, tokenizers, gradients, CUDA execution, and BEIR retrieval scores.
SuperRed optimizers do not receive that internal retriever state from targets, so
this module does not implement HotFlip as the default optimizer path.

The released code computes retrieval precision/recall/F1 by directly embedding
poison texts and original corpus documents. SuperRed targets may expose retrieved
contexts on the trajectory, but they do not expose a universal retriever score
API. This optimizer records attack success through framework feedback or target
answer containment, and leaves retriever-specific metrics to target-specific
claims when available.
