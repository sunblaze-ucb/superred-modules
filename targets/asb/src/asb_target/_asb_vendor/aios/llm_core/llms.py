# This file just wraps around the LLM classes in aios/llm_kernel/llm_classes
# and provides an easy interface for the rest of the code to access
# All abstractions will be implemented here

from .llm_classes.model_registry import MODEL_REGISTRY

# Local-backend LLM classes (HfNative/Ollama/vLLM) pull heavy deps
# (torch, transformers, vllm, ollama). They are imported lazily inside the
# local-model branch of LLMKernel.__init__ so the superred port, which routes
# every model through the OpenAI-compatible proxy, needs none of them
# (see ASSUMPTIONS.md).

class LLMKernel:
    def __init__(self,
                 llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 256,
                 log_mode: str = "console",
                 use_backend: str = None
        ):

        # For API-based LLM
        if llm_name in MODEL_REGISTRY.keys():
            self.model = MODEL_REGISTRY[llm_name](
                llm_name = llm_name,
                log_mode = log_mode
            )
        # For locally-deployed LLM
        else:
            from .llm_classes.hf_native_llm import HfNativeLLM
            from .llm_classes.ollama_llm import OllamaLLM
            from .llm_classes.vllm import vLLM

            if use_backend == "ollama" or llm_name.startswith("ollama"):
                self.model = OllamaLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )

            elif use_backend == "vllm":
                self.model = vLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )
            else: # use huggingface LLM without backend
                self.model = HfNativeLLM(
                    llm_name=llm_name,
                    max_gpu_memory=max_gpu_memory,
                    eval_device=eval_device,
                    max_new_tokens=max_new_tokens,
                    log_mode=log_mode
                )

    def address_request(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request(agent_process,temperature)

    def address_request_list(self,
                        agent_process,
                        temperature=0.0):
        self.model.address_request_list(agent_process,temperature)
