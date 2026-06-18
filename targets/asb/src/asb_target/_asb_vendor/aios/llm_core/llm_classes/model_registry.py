# Registry of closed/proprietary LLM classes.
#
# superred port: pruned to the OpenAI-compatible GPTLLM only. The
# gemini/bedrock/claude classes pulled google-generativeai / boto3 /
# anthropic and are unused here because every model is routed through the
# litellm proxy via the OpenAI API. asb_target.llm_proxy registers a
# ProxyLLM for the configured model name at runtime (see ASSUMPTIONS.md).

from .gpt_llm import GPTLLM

MODEL_REGISTRY = {
    'gpt-3.5-turbo': GPTLLM,
    'gpt-4-turbo': GPTLLM,
    'gpt-4o': GPTLLM,
    'gpt-4o-2024-08-06': GPTLLM,
    'gpt-4o-mini': GPTLLM,
}
