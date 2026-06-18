# Vendored Agent Security Bench (ASB) source

This directory contains a verbatim copy of selected source from
**Agent Security Bench (ASB)** by AGI Research, used to run ASB's real
agent loop faithfully inside the `asb_target` superred module.

- Upstream: https://github.com/agiresearch/ASB
- Pinned commit: `1f561dccf92d55302368fa67679b4ba9d9c8fdc4`
- License: MIT (Copyright (c) 2024 AGI Research) — see `LICENSE` in this directory.

## What was copied

- `pyopenagi/` (agents, tools, utils, queues, context) — the agent runtime,
  the plan-then-execute `ReactAgentAttack` loop, and the simulated tools.
- `aios/` (scheduler, llm_core, and the rest of the tree) — the FIFO
  scheduler and the LLM kernel.

The copy is wholesale; modules the simulated-agent path never imports
(real tools, image/diffusers tools, the AgentFactory/Interactor download
machinery) are present but unused.

## Modifications (de-dependency surgery only; behavior preserved)

To run on the user's litellm proxy without ASB's heavy optional
dependencies, four files were edited (all marked in-code, all documented
in `targets/asb/ASSUMPTIONS.md`):

1. `aios/llm_core/llm_classes/model_registry.py` — pruned to the
   OpenAI-compatible `GPTLLM` only; dropped the gemini/bedrock/claude
   entries (which imported `google-generativeai` / `boto3` / `anthropic`).
2. `aios/llm_core/llms.py` — made the local-backend LLM imports
   (`HfNativeLLM`/`OllamaLLM`/`vLLM`, which pull `torch`/`transformers`/
   `vllm`/`ollama`) lazy, inside the local-model branch.
3. `pyopenagi/agents/react_agent_attack.py` — made the `langchain`/`chroma`
   imports optional (guarded), since the port reproduces ASB's vector memory
   with its own lightweight in-process store (`asb_target.memory_store`) over
   the proxy's embeddings, rather than `langchain_chroma`.
4. `aios/context/simple_context.py` — made `import torch` lazy (it is only
   used by GPU context snapshot/recover, which the port never calls), so a
   top-level import does not needlessly pull in `torch`.

No agent-behavioral logic (prompts, plan format, injection sites, simulated
tool returns, success strings) was changed.
